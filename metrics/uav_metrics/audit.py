"""Independent-reference audit categories and target-negative evidence.

Trust and independence are explicit user declarations, never model-inferred.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from metrics.uav_metrics.research import VERSION, bounded, finite, object_records, bin_name, size_bins, distribution

CATEGORIES = ('missing_objects', 'incorrect_classes', 'localization_errors', 'duplicate_annotations',
              'spurious_annotations', 'incorrect_negative_images', 'ignored_uncertain_objects', 'invalid_syntax_images')


def iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    intersection = max(0, min(x+w/2, xx+ww/2)-max(x-w/2, xx-ww/2))*max(0, min(y+h/2, yy+hh/2)-max(y-h/2, yy-hh/2))
    return intersection/(w*h+ww*hh-intersection) if w*h+ww*hh-intersection > 0 else 0.


def match_pairs(annotations, references, threshold, *, class_aware=False, used_a=None, used_r=None):
    used_a = set() if used_a is None else used_a
    used_r = set() if used_r is None else used_r
    candidates = []
    for i, a in enumerate(annotations):
        for j, r in enumerate(references):
            if i in used_a or j in used_r or class_aware and a['class'] != r['class']:
                continue
            overlap = iou(a['bbox'], r['bbox'])
            if overlap >= threshold:
                candidates.append((-overlap, i, j))
    matches = []
    for negative_overlap, i, j in sorted(candidates):
        if i not in used_a and j not in used_r:
            used_a.add(i)
            used_r.add(j)
            matches.append((i, j, -negative_overlap))
    return matches


def annotation_audit(ds, g):
    audit = ds.metadata.get('annotation_audit')
    unavailable = {'version': VERSION, 'status': 'insufficient_evidence', 'semantic_correctness': 'unverified',
                   'reason': ds.metadata.get('annotation_audit_unavailable_reason','Provide independently human-reviewed or trusted external reference images; unverified model predictions are ineligible.'),
                   'source_scope':ds.metadata.get('annotation_audit_source_scope')}
    if not isinstance(audit, dict) or 'images' not in audit:
        return unavailable
    if audit.get('independent') is not True or audit.get('evidence_type') not in ('human_verified', 'trusted_external') or not audit.get('reviewer') or not audit.get('protocol') or not audit.get('source_id'):
        return unavailable
    if audit.get('dataset_sha256') and audit['dataset_sha256'] != ds.sha256:
        return {**unavailable, 'reason': 'Audit dataset fingerprint does not match this population'}
    images = audit['images']
    if not isinstance(images, list) or len(images) > 512 or not all(isinstance(x,dict) for x in images) or len({x.get('image') for x in images}) != len(images):
        raise ValueError('Audit requires at most 512 uniquely identified images')
    match_iou = audit.get('match_iou', .5)
    localization_iou = audit.get('localization_iou', .1)
    if not finite(match_iou) or not finite(localization_iou) or not 0 < localization_iou < match_iou <= 1:
        raise ValueError('Require 0 < localization_iou < match_iou <= 1')
    matching_mode = audit.get('matching_mode', 'class_agnostic')
    if matching_mode not in ('class_agnostic', 'class_aware'):
        raise ValueError('Unsupported audit matching_mode')
    rows = ds.frame.set_index('image').to_dict('index')
    annotations = defaultdict(list)
    for obj in object_records(ds, g):
        annotations[obj['image']].append(obj)
    counts = Counter({k: 0 for k in CATEGORIES})
    per_class, per_size = defaultdict(Counter), defaultdict(Counter)
    totals = Counter(reference_objects=0,annotations_reviewed=0)
    reviewed_images, verified_negatives, records = [], [], []
    semantic_images=0
    pairs_budget = 200000
    for entry in images:
        name = entry.get('image')
        if name not in rows or entry.get('reference_complete') is not True or not rows[name].get('readable'):
            raise ValueError('Audit images must be readable members with explicit reference_complete=true')
        row = rows[name]
        reference = entry.get('reference_objects')
        if not isinstance(reference, list) or len(reference) > 500 or len(annotations[name]) > 500:
            raise ValueError('Audit bounded to 500 references/annotations per image')
        active, ignored = [], []
        for obj in reference:
            box = obj.get('bbox', [])
            if not obj.get('class') or len(box) != 4 or not all(finite(x) for x in box):
                raise ValueError('Reference objects require class and normalized center bbox')
            x, y, w, h = box
            if min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-9 or max(x+w/2, y+h/2) > 1+1e-9:
                raise ValueError('Invalid independent reference geometry')
            obj = {**obj, 'size_bin': bin_name(w*row['width']*h*row['height'], size_bins(g))}
            (ignored if obj.get('ignored') or obj.get('uncertain') else active).append(obj)
        annotated = annotations[name]
        valid_labels = bool(row.get('annotation_present') and row.get('annotation_valid'))
        if not valid_labels:
            # Missing/unparseable annotations do not establish semantic omission.
            # Their syntax/presence diagnostics remain independently observable.
            if row.get('annotation_present'):
                counts['invalid_syntax_images'] += 1
            reviewed_images.append(name)
            records.append({'image':name,'semantic_status':'unassessable_annotation_syntax_or_presence','reference_objects':len(active)})
            if not active and not ignored and entry.get('negative_verified') is True:
                verified_negatives.append(name)
            continue
        semantic_images+=1
        pairs_budget -= len(annotated)*(len(active)+len(ignored))*3
        if pairs_budget < 0:
            raise ValueError('Independent audit comparison budget exceeded; submit a smaller audited subset')
        # Predictions/annotations intersecting independently ignored reference regions
        # are excluded from error claims, not counted as verified correct.
        uncertain = [a for a in annotated if any(iou(a['bbox'], r['bbox']) >= match_iou for r in ignored)]
        annotated = [a for a in annotated if a not in uncertain]
        used_a, used_r = set(), set()
        events = []
        exact = match_pairs(annotated, active, match_iou, class_aware=matching_mode == 'class_aware', used_a=used_a, used_r=used_r)
        for i, j, overlap in exact:
            if annotated[i]['class'] != active[j]['class']:
                events.append(('incorrect_classes', active[j]))
        # A second class-agnostic pass identifies wrong-class boxes when the
        # requested primary matching is class-aware, without double assignment.
        for i, j, overlap in match_pairs(annotated, active, match_iou, used_a=used_a, used_r=used_r):
            events.append(('incorrect_classes', active[j]))
        for i, j, overlap in match_pairs(annotated, active, localization_iou, class_aware=True, used_a=used_a, used_r=used_r):
            events.append(('localization_errors', active[j]))
        for i, a in enumerate(annotated):
            if i not in used_a:
                duplicate = any(j in used_r and a['class'] == r['class'] and iou(a['bbox'], r['bbox']) >= match_iou for j, r in enumerate(active))
                events.append(('duplicate_annotations' if duplicate else 'spurious_annotations', a))
        events.extend(('missing_objects', r) for j, r in enumerate(active) if j not in used_r)
        if valid_labels and not annotations[name] and active:
            counts['incorrect_negative_images'] += 1
        counts['ignored_uncertain_objects'] += len(ignored)
        for category, obj in events:
            counts[category] += 1
            per_class[obj['class']][category] += 1
            per_size[obj['size_bin']][category] += 1
        for population, items in (('reference_objects', active), ('annotations_reviewed', annotated)):
            totals[population] += len(items)
            for obj in items:
                per_class[obj['class']][population] += 1
                per_size[obj['size_bin']][population] += 1
        if not active and not ignored and entry.get('negative_verified') is True:
            verified_negatives.append(name)
        reviewed_images.append(name)
        records.append({'image': name, 'errors': dict(Counter(k for k, _ in events)), 'reference_objects': len(active),
                        'annotations_reviewed': len(annotated), 'ignored_annotations': len(uncertain)})
    denominators = {k: len(reviewed_images) if k.endswith('images') else totals['reference_objects'] if k in ('missing_objects', 'incorrect_classes', 'localization_errors') else totals['annotations_reviewed'] for k in CATEGORIES}
    denominators['ignored_uncertain_objects'] = totals['reference_objects']+counts['ignored_uncertain_objects']
    def strata(groups):
        output={}
        for key,values in groups.items():
            result=dict(values)
            result['rates']={}
            for category in ('missing_objects','incorrect_classes','localization_errors','duplicate_annotations','spurious_annotations'):
                denominator=values['reference_objects'] if category in ('missing_objects','incorrect_classes','localization_errors') else values['annotations_reviewed']
                result['rates'][category]={'count':values[category],'denominator':denominator,'fraction':values[category]/denominator if denominator else None}
            output[key]=result
        return output
    return {'version': VERSION, 'status': 'measured' if reviewed_images else 'insufficient_evidence', 'evidence_type': audit['evidence_type'],
            'semantic_correctness': 'audited_subset_only' if semantic_images else 'unverified', 'images_audited': len(reviewed_images),
            'semantic_images_audited':semantic_images,'semantic_image_coverage':semantic_images/len(ds.frame) if len(ds.frame) else None,
            'reference_evidence_type':audit['evidence_type'],'error_measurement_method':'algorithmic comparison against declared independent references under the recorded matching protocol',**dict(totals),
            'error_counts': dict(counts), 'error_rates': {k: {'count': counts[k], 'denominator': denominators[k], 'fraction': counts[k]/denominators[k] if denominators[k] else None} for k in CATEGORIES},
            'per_class': strata(per_class), 'per_size': strata(per_size),
            'image_coverage': len(reviewed_images)/len(ds.frame) if len(ds.frame) else None, 'image_population': len(ds.frame),
            'reviewed_annotation_coverage':totals['annotations_reviewed']/len(object_records(ds,g)) if object_records(ds,g) else None,
            'sampling_design':audit.get('sampling_design','unspecified; audited subset only'),
            'source_scope':ds.metadata.get('annotation_audit_source_scope'),
            'audited_images': reviewed_images, 'verified_negative_images': verified_negatives, 'image_results': records,
            'protocol': {'reviewer': audit['reviewer'], 'source_id': audit['source_id'], 'protocol': audit['protocol'], 'matching_mode': matching_mode,
                         'match_iou': match_iou, 'localization_iou': localization_iou, 'matching': 'descending IoU; deterministic index tie break; one-to-one; high-IoU class mismatch before low-IoU same-class localization'},
            'audit_sha256': hashlib.sha256(json.dumps(audit, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'limitations': ['Independence/trust are declared, not authenticated.', 'No extrapolation beyond audited images; selection/sampling design must be supplied by the reviewer.',
                            'Categories have different denominators and may overlap at image level; do not sum rates.', 'Localization errors below localization_iou appear as unmatched missing/spurious objects.']}


def negative_coverage(ds, g, audit):
    by_image = defaultdict(set)
    for obj in object_records(ds, g):
        by_image[obj['image']].add(obj['class'])
    targets = set(g.required_classes)
    explicit, unknown, other_classes, target_free = [], [], [], []
    for row in ds.frame.to_dict('records'):
        name = row['image']
        if not row.get('readable') or not row.get('annotation_present') or not row.get('annotation_valid'):
            unknown.append(name)
        elif not by_image[name]:
            explicit.append(name)
            target_free.append(name)
        elif targets and not targets & by_image[name]:
            other_classes.append(name)
            target_free.append(name)
    verified = set(audit.get('verified_negative_images', [])) & set(ds.frame.image)
    per_class = {}
    for cls in sorted(targets):
        valid = set(ds.frame.image)-set(unknown)
        absent = [name for name in valid if cls not in by_image[name]]
        per_class[cls] = {'annotated_target_free_images': len(absent), 'denominator': len(valid), 'unknown_images': len(unknown)}
    scenarios = []
    from metrics.uav_metrics.research import scenario_evidence
    for s in g.scenarios[:32]:
        # Image metadata is retained; target-free membership is an explicit,
        # derived observation, not an inference from missing annotations.
        evidence = scenario_evidence(ds, g, s.conditions)
        conditions = s.conditions
        from metrics.uav_metrics.research import condition, conjunction
        image_only = s.required_class is None and not set(conditions) & {'class', 'object_size', 'occlusion', 'truncation', 'minimum_side_pixels', 'area_pixels', 'relative_area'}
        count = sum(name in target_free and conjunction(condition(row.get(g.uav_parameters.get('columns', {}).get(k, k)), v) for k, v in conditions.items()) is True for row in ds.frame.to_dict('records') for name in [row['image']]) if image_only else None
        scenarios.append({'name': s.name, 'negative_image_count': count, 'scenario_population': evidence['population'], 'unknown_count': evidence['unknown_count'],
                          'reason': 'Image-metadata conjunction' if image_only else 'Object-presence scenarios do not define a target-free image population'})
    candidates = ds.metadata.get('hard_negative_candidates', [])
    accepted = []
    if not isinstance(candidates, list) or len(candidates) > 512:
        raise ValueError('At most 512 hard-negative candidate records are supported')
    for item in candidates:
        if not isinstance(item, dict) or item.get('image') not in target_free or not item.get('reason') or not item.get('source_id'):
            raise ValueError('Hard-negative candidates require a known annotated target-free image, reason and source_id')
        if item.get('evidence_type') == 'model_prediction' and (not item.get('model_id') or not finite(item.get('confidence_threshold')) or not 0 <= item['confidence_threshold'] <= 1):
            raise ValueError('Model candidates require identity and finite confidence threshold')
        accepted.append({**item, 'status': 'unverified_candidate'})
    valid = len(ds.frame)-len(unknown)
    return {'version': VERSION, 'status': 'measured' if valid else 'insufficient_evidence', 'evidence_type': 'observed_annotations_and_optional_independent_audit',
            'target_classes': sorted(targets), 'explicit_negative_images': len(explicit), 'unknown_label_images': len(unknown),
            'other_class_only_images': len(other_classes) if targets else None, 'annotated_target_free_images': len(target_free) if targets else None,
            'negative_proportion': len(explicit)/valid if valid else None, 'denominator': valid, 'image_population': len(ds.frame),
            'verified_negative_images': len(verified), 'verified_negative_coverage': len(verified & set(target_free))/len(target_free) if targets and target_free else None,
            'verified_negative_population_coverage':len(verified)/len(ds.frame) if len(ds.frame) else None,
            'per_class': per_class, 'per_scenario': scenarios, 'hard_negative_candidates': accepted,
            'candidate_count': len(accepted), 'limitations': ['Empty valid annotations are explicit negatives; missing labels remain unknown.',
                'Target-free annotations are not independently verified completeness.', 'Candidates do not establish hard-negative difficulty or measured false positives.']}
