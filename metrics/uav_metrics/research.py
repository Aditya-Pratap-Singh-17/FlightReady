"""Versioned descriptive evidence shared by existing UAV registry properties.

No scores, learned thresholds or model-performance conclusions are produced.
"""
from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd

VERSION = 'research-evidence-2.0'
DEFAULT_BINS = {'small': [0, 1024], 'medium': [1024, 9216], 'large': [9216, None]}


def finite(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def bounded(value, low, high, name):
    if not finite(value) or int(value) != value or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return int(value)


def distribution(values, unit):
    a = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(a):
        return {'status': 'insufficient_evidence', 'count': 0, 'unit': unit}
    q = np.quantile(a, [0, .05, .25, .5, .75, .95, 1])
    counts, edges = np.histogram(a, bins=min(12, max(1, len(a))))
    return dict(zip(('minimum', 'q05', 'q25', 'median', 'q75', 'q95', 'maximum'), map(float, q)),
                status='measured', count=len(a), unit=unit, mean=float(a.mean()),
                histogram=[{'lower': float(edges[i]), 'upper': float(edges[i+1]), 'count': int(n)} for i, n in enumerate(counts)])


def size_bins(g):
    bins = g.uav_parameters.get('object_size_bins', DEFAULT_BINS)
    if not isinstance(bins, dict) or not 1 <= len(bins) <= 16:
        raise ValueError('object_size_bins requires 1..16 nonoverlapping native-area intervals')
    previous = -math.inf
    for name, interval in bins.items():
        if not isinstance(name, str) or not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError('Invalid object size interval')
        low, high = interval
        if not finite(low) or low < 0 or high is not None and (not finite(high) or high <= low) or low < previous:
            raise ValueError('Size intervals must be ordered, finite, nonnegative and disjoint')
        previous = math.inf if high is None else high
    return bins


def bin_name(area, bins):
    return next((name for name, (lo, hi) in bins.items() if lo <= area and (hi is None or area < hi)), 'outside_declared_bins')


def object_records(ds, g):
    """Validate normalized center boxes again, including reloaded/legacy evidence."""
    rows = ds.frame.set_index('image').to_dict('index') if 'image' in ds.frame else {}
    bins = size_bins(g)
    output = []
    for index, obj in enumerate(ds.objects):
        if not isinstance(obj.get('class'), str) or not obj['class'].strip():
            continue
        row = rows.get(obj.get('image'), {})
        box = obj.get('bbox', [])
        if not row.get('readable') or len(box) != 4 or not all(finite(x) for x in box):
            continue
        x, y, w, h = box
        iw, ih = row.get('width'), row.get('height')
        if not finite(iw) or not finite(ih) or min(iw, ih, w, h) <= 0 or min(x-w/2, y-h/2) < -1e-9 or max(x+w/2, y+h/2) > 1+1e-9:
            continue
        width, height = w*iw, h*ih
        output.append({**obj, 'object_index': index, 'width_pixels': width, 'height_pixels': height,
                       'area_pixels': width*height, 'minimum_side_pixels': min(width, height),
                       'relative_area': w*h, 'aspect_ratio': width/height,
                       'size_bin': bin_name(width*height, bins), 'image_width': iw, 'image_height': ih})
    return output


def scale_analysis(ds, g):
    objects = object_records(ds, g)
    bins = size_bins(g)
    spec = g.uav_parameters.get('detector_preprocessing')
    model_status = 'insufficient_evidence'
    if spec:
        if not isinstance(spec, dict) or set(spec)-{'mode','input_size','scaleup'} or spec.get('mode') not in ('letterbox', 'stretch'):
            raise ValueError('detector_preprocessing requires explicit mode letterbox or stretch')
        shape = spec.get('input_size')
        if isinstance(shape, int):
            shape = [shape, shape]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError('input_size must be [height,width] or a square side')
        ih, iw = [bounded(x, 16, 8192, 'input_size') for x in shape]
        if 'scaleup' not in spec or not isinstance(spec['scaleup'], bool):
            raise ValueError('Explicit boolean scaleup required; augmentation geometry is not inferred')
        if spec['mode']=='stretch' and spec['scaleup'] is False:
            raise ValueError('Stretch mode always fills the declared shape; use letterbox for scaleup=false')
        for o in objects:
            sx, sy = iw/o['image_width'], ih/o['image_height']
            if spec['mode'] == 'letterbox':
                sx = sy = min(sx, sy, math.inf if spec['scaleup'] else 1)
            o.update(model_width_pixels=o['width_pixels']*sx, model_height_pixels=o['height_pixels']*sy,
                     model_minimum_side_pixels=min(o['width_pixels']*sx, o['height_pixels']*sy),
                     model_area_pixels=o['area_pixels']*sx*sy)
        model_status = 'measured' if objects else 'insufficient_evidence'
    fields = ['width_pixels', 'height_pixels', 'area_pixels', 'minimum_side_pixels', 'relative_area', 'aspect_ratio']
    if spec:
        fields += ['model_width_pixels', 'model_height_pixels', 'model_minimum_side_pixels', 'model_area_pixels']
    def stats(items):
        return {key: distribution([o[key] for o in items], 'ratio' if key in ('relative_area', 'aspect_ratio') else 'pixel^2' if 'area' in key else 'pixel') for key in fields}
    classes = sorted(set(g.required_classes) | {str(o['class']) for o in objects})
    counts = Counter(o['size_bin'] for o in objects)
    all_bins = list(bins) + (['outside_declared_bins'] if counts['outside_declared_bins'] else [])
    per_class = {c: {b: sum(o['class'] == c and o['size_bin'] == b for o in objects) for b in all_bins} for c in classes}
    criteria = g.uav_parameters.get('small_object_criteria')
    coverage = {'status': 'insufficient_evidence', 'reason': 'Confirm small_object_criteria with classes, area range, optional minimum_objects; descriptive bins are not task requirements.'}
    if criteria:
        classes_requested = criteria.get('classes', g.required_classes)
        area_range = criteria.get('native_area_range')
        if not classes_requested or not isinstance(area_range, list) or len(area_range) != 2 or not all(finite(v) for v in area_range) or not 0 <= area_range[0] < area_range[1]:
            raise ValueError('Small-object criteria require classes and an explicit finite [low,high) native area range')
        matching = [o for o in objects if o['class'] in classes_requested and area_range[0] <= o['area_pixels'] < area_range[1]]
        minimum = criteria.get('minimum_objects')
        if minimum is not None:
            minimum = bounded(minimum, 1, 10**9, 'minimum_objects')
        incomplete = len(objects) != len(ds.objects) or not all(ds.frame.get('annotation_present', pd.Series(False, index=ds.frame.index)) & ds.frame.get('annotation_valid', pd.Series(False, index=ds.frame.index)))
        coverage = {'status': 'pass' if minimum is not None and len(matching) >= minimum else 'insufficient_evidence' if incomplete or not objects else 'fail' if minimum else 'measured',
                    'criteria': criteria, 'object_count': len(matching), 'image_count': len({o['image'] for o in matching}),
                    'per_class': dict(Counter(o['class'] for o in matching)), 'denominator': len(objects),
                    'incomplete_annotation_population': incomplete, 'minimum_objects': minimum}
    return {'version': VERSION, 'status': 'measured' if objects else 'insufficient_evidence', 'evidence_type': 'observed_annotation_geometry',
            'object_count': len(objects), 'image_population': len(ds.frame), 'excluded_objects': len(ds.objects)-len(objects), 'annotated_image_count': len({o['image'] for o in objects}),
            'size_bins': bins, 'bin_policy': 'Native pixel area [lower,upper); descriptive defaults, not detectability thresholds',
            'object_counts': {b: counts[b] for b in all_bins}, 'object_percentages': {b: 100*counts[b]/len(objects) if objects else None for b in all_bins},
            'image_counts': {b: len({o['image'] for o in objects if o['size_bin'] == b}) for b in all_bins},
            'per_class_bin_counts': per_class, 'native_and_input_statistics': stats(objects),
            'per_class_statistics': {c: stats([o for o in objects if o['class'] == c]) for c in classes},
            'model_input': {'status': model_status, 'configuration': spec, 'reason': 'Ideal deterministic box transform; padding does not change box dimensions. Integer raster rounding, stride padding and training augmentations may differ.' if spec else 'Input size and preprocessing are unspecified.'},
            'small_object_coverage': coverage, 'limitations': ['Bounding-box geometry does not establish detectability.', 'Only valid annotations on readable, dimensioned images contribute.']}


def condition(value, expected):
    """Three-valued atomic comparison; false dominates unknown in conjunctions."""
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)) or isinstance(value, str) and not value.strip():
        return None
    if isinstance(expected, dict):
        if not expected or set(expected) - {'min', 'max'}:
            raise ValueError('Numeric conditions use min/max bounds')
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(value):
            return None
        for bound in expected.values():
            if bound is not None and not finite(bound):
                raise ValueError('Condition bounds must be finite')
        if expected.get('min') is not None and expected.get('max') is not None and expected['min']>expected['max']:
            raise ValueError('Condition minimum exceeds maximum')
        return (expected.get('min') is None or value >= expected['min']) and (expected.get('max') is None or value <= expected['max'])
    values = expected if isinstance(expected, list) else [expected]
    return str(value).lower() in [str(v).lower() for v in values]


def conjunction(states):
    states = list(states)
    return False if False in states else None if None in states else True


def scenario_evidence(ds, g, conditions, required_class=None, minimum=None):
    if not isinstance(conditions, dict) or len(conditions) > 12:
        raise ValueError('A scenario supports at most 12 conjunctive conditions')
    if minimum is not None:
        minimum = bounded(minimum, 1, 10**9, 'minimum_samples')
    objects = object_records(ds, g) if ds.modality == 'image' else []
    by_image = defaultdict(list)
    for obj in objects:
        by_image[obj['image']].append(obj)
    obj_keys = {'class', 'object_size', 'occlusion', 'truncation', 'minimum_side_pixels', 'area_pixels', 'relative_area'}
    object_conditions = {k: v for k, v in conditions.items() if k in obj_keys} if ds.modality == 'image' else {}
    if required_class and ds.modality == 'image' and 'class' not in object_conditions:
        object_conditions['class'] = required_class
    matched, unknown, nonmatching, missing, matching_objects = [], [], [], [], []
    for index, row in ds.frame.iterrows():
        image = row.get('image', f'row:{index}')
        states = [condition(row.get(g.uav_parameters.get('columns', {}).get(k, k)), v) for k, v in conditions.items() if k not in object_conditions]
        if ds.modality != 'image' and required_class:
            states.append(condition(row.get(g.prediction_target), required_class))
        meta_state = conjunction(states)
        if None in states:
            missing.append(image)
        eligible = ds.modality != 'image' or bool(row.get('readable', False) and row.get('annotation_present', False) and row.get('annotation_valid', False))
        image_objects = by_image[image]
        if ds.modality == 'image' and object_conditions:
            object_states = [(o, conjunction([condition(None if k=='object_size' and 'object_size_bins' not in g.uav_parameters else o.get('size_bin' if k == 'object_size' else k), v) for k, v in object_conditions.items()]+([condition(o['class'],required_class)] if required_class else []))) for o in image_objects]
            obj_state = True if any(s is True for _, s in object_states) else None if not eligible or any(s is None for _, s in object_states) else False
        else:
            object_states = [(o, True) for o in image_objects]
            obj_state = True if eligible else None
        final = conjunction([meta_state, obj_state])
        if final is True:
            matched.append(image)
            matching_objects.extend(o for o, s in object_states if s is True)
        elif final is None:
            unknown.append(image)
        else:
            nonmatching.append(image)
    state = 'pass' if minimum and len(matched) >= minimum else 'insufficient_evidence' if unknown else 'fail' if minimum else 'measured'
    return {'definition': {'conditions': conditions, 'required_class': required_class}, 'matching_image_count': len(matched),
            'matching_object_count': len(matching_objects), 'per_class_counts': dict(Counter(o['class'] for o in matching_objects)),
            'population': len(ds.frame), 'known_eligible_population': len(matched)+len(nonmatching), 'known_nonmatching_count': len(nonmatching),
            'unknown_count': len(unknown), 'missing_metadata_count': len(missing), 'required_minimum': minimum,
            'matching_image_fraction':len(matched)/len(ds.frame) if len(ds.frame) and not unknown else None,
            'coverage_identification_bounds':[len(matched)/len(ds.frame),(len(matched)+len(unknown))/len(ds.frame)] if len(ds.frame) else None,
            'status': state, 'matching_references': matched[:20], 'provenance': {'dataset_sha256': ds.sha256, 'metadata': 'user-declared; not appearance-inferred', 'classes_and_sizes': 'observed annotations; size from configured native-area bins'},
            'limitations': ['Unknown is not proven absence.', 'Named object_size conditions require explicitly confirmed object_size_bins; descriptive defaults are not task criteria.', 'Conjunctions are evaluated on the same annotated object; no marginal independence assumption.']}


def scenarios(ds, g, reference=None):
    if len(g.scenarios) > 32:
        raise ValueError('At most 32 confirmed scenarios are supported')
    definitions = [(s.name, s.conditions, s.required_class, s.minimum_samples) for s in g.scenarios]
    if g.target_environment:
        definitions.append(('Declared deployment environment', g.target_environment, None, None))
    output = []
    declared = g.uav_parameters.get('scenario_reference_fractions', {})
    for name, conditions, cls, minimum in definitions:
        item = {'name': name, **scenario_evidence(ds, g, conditions, cls, minimum)}
        comparison = {'status': 'insufficient_evidence', 'reason': 'No independent deployment reference or declared scenario fraction'}
        if reference is not None and reference.disk_path is None:
            comparison = {'status': 'measured', 'reference': scenario_evidence(reference, g, conditions, cls), 'reference_type': 'user-supplied dataset; relevance not authenticated'}
        elif name in declared:
            fraction = declared[name]
            if not finite(fraction) or not 0 <= fraction <= 1:
                raise ValueError('Scenario reference fractions must be finite in [0,1]')
            comparison = {'status': 'declared', 'reference_fraction': fraction, 'reference_type': 'user-declared deployment distribution'}
        item['reference_comparison'] = comparison
        if comparison['status']=='declared':
            comparison['difference_from_declared_fraction']=item['matching_image_fraction']-comparison['reference_fraction'] if item['matching_image_fraction'] is not None else None
        elif 'reference' in comparison:
            reference_fraction=comparison['reference']['matching_image_fraction']
            comparison['difference_from_reference_fraction']=item['matching_image_fraction']-reference_fraction if item['matching_image_fraction'] is not None and reference_fraction is not None else None
        output.append(item)
    from itertools import combinations,islice
    pairwise=[]
    limit=bounded(g.uav_parameters.get('scenario_pairwise_limit',0),0,16,'scenario_pairwise_limit')
    for left,right in islice(combinations(definitions,2),limit):
        name=left[0]+' AND '+right[0]
        if any(k in left[1] and left[1][k]!=v for k,v in right[1].items()) or left[2] and right[2] and left[2]!=right[2]:
            pairwise.append({'name':name,'status':'unsupported','reason':'Conflicting predicates require an explicit confirmed combined scenario; no intersection was assumed.'})
        else:
            pairwise.append({'name':name,**scenario_evidence(ds,g,{**left[1],**right[1]},left[2] or right[2])})
    return {'version': VERSION, 'status': 'measured' if output else 'insufficient_evidence', 'scenarios': output,
            'pairwise_summaries':pairwise,'pairwise_limit':limit,
            'pairwise_policy': 'Deterministic confirmed-scenario order; optional at most 16 pairs; conflicting predicates require explicit conjunctions.',
            'representativeness': 'unverified without a meaningful deployment reference; coverage is descriptive'}
