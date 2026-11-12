"""Deterministic dimension scores from frozen observations and explicit criteria.

Unknown evidence keeps its assigned weight. A primary score is returned only at
100% weighted evidence coverage. Genuine measured zeroes remain zero. Hard gates
are evaluated independently of all weighted means.
"""
import copy
import math
from metrics.quality_profiles import QualityProfile, fingerprint, load_profile

CONTEXT_DEPENDENT = {'missing_label_rate', 'class_distribution', 'verified_label_error', 'UAV-08', 'UAV-12', 'UAV-14', 'UAV-15', 'UAV-17', 'UAV-18', 'UAV-19', 'UAV-20'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(value, method):
    if not finite(value):
        raise ValueError('Expected finite numerical evidence, excluding booleans')
    if method in ('fraction', 'one_minus_fraction'):
        if not 0 <= value <= 1:
            raise ValueError('Fraction evidence must lie in [0, 1]')
        return 100 * (1 - value if method == 'one_minus_fraction' else value)
    if method == 'zero_count':
        if value < 0 or value != int(value):
            raise ValueError('Count evidence must be a nonnegative integer')
        return 100. if value == 0 else 0.
    raise ValueError('This normalization requires a structured measurement')


def geometric(scores, weights):
    if len(scores)!=len(weights) or any(not finite(s) or not 0<=s<=100 for s in scores) or any(not finite(w) or w<0 for w in weights):
        raise ValueError('Geometric aggregation requires matched finite scores in [0,100] and nonnegative weights')
    if not scores or sum(weights) <= 0:
        return None
    # Do not turn a real zero into epsilon, or take log(0).
    if any(s == 0 and w > 0 for s, w in zip(scores, weights)):
        return 0.
    total = sum(weights)
    return 100 * math.exp(sum(w / total * math.log(s / 100) for s, w in zip(scores, weights) if w > 0))


def collect_measurements(ds, threshold=None, goal=None):
    """Reuse ingestion pixel measurements, never decode/reprocess image files."""
    from numpy import histogram
    from pandas import to_numeric
    threshold = load_profile().clarity_threshold if threshold is None else threshold
    result = {'dataset_sha256': ds.sha256, 'scope': ds.profile.get('assessment_scope', 'full ingested dataset'),
              'distributions': {}, 'clarity': None, 'version': 'pixel-evidence-1.0'}
    if ds.modality != 'image' or ds.disk_path or 'readable' not in ds.frame:
        return result
    frame = ds.frame.loc[ds.frame.readable]
    for name in ('laplacian_variance', 'brightness', 'contrast', 'width', 'height', 'encoded_bytes_per_pixel',
                 'research_edge_density','research_high_frequency_residual_mad','research_underexposed_fraction','research_saturated_fraction','research_local_contrast_rms'):
        if name not in frame:
            continue
        values = to_numeric(frame[name], errors='coerce')
        known = values.dropna().map(float)
        known = known[known.map(math.isfinite)]
        if len(known):
            counts, edges = histogram(known, bins=min(16, max(1, len(known))))
            result['distributions'][name] = {'count': len(known), 'population': len(frame),
                'bins': [{'lower': float(edges[i]), 'upper': float(edges[i+1]), 'count': int(n)} for i, n in enumerate(counts)],
                'mean': float(known.mean()), 'minimum': float(known.min()), 'maximum': float(known.max()),
                'method': 'Histogram of actual readable-image ingestion measurements; final bin includes upper edge.'}
        if name == 'laplacian_variance' and len(known) == len(frame) and len(frame):
            result['clarity'] = {'threshold': threshold, 'meeting': int((known >= threshold).sum()),
                'denominator': len(known), 'population': len(ds.frame),
                'method': 'Four-neighbor Laplacian variance on ingestion thumbnail <=256 px; illustrative criterion.'}
    criterion = goal.uav_parameters.get('minimum_object_side_pixels') if goal else None
    if finite(criterion) and criterion > 0 and ds.objects:
        dimensions=frame.set_index('image')[['width','height']].to_dict('index')
        sizes=[min(o['bbox'][2]*dimensions[o['image']]['width'],o['bbox'][3]*dimensions[o['image']]['height'])
               for o in ds.objects if o['image'] in dimensions]
        if sizes:
            result['object_visibility']={'threshold':criterion,'meeting':sum(v>=criterion for v in sizes),'denominator':len(sizes),
                'criterion_provenance':'Explicit confirmed minimum_object_side_pixels; annotation-derived, not semantic detectability.'}
    return result


def _observed(metric):
    if not metric:
        return 'not_assessed', 'Measurement was not collected; run a task-specific assessment.'
    state = metric.get('assessment_status') or metric.get('state')
    if state == 'not_applicable':
        return 'not_applicable', metric.get('reason', '')
    if state in ('error', 'failed'):
        return 'failed', metric.get('reason', 'Measurement failed')
    if state in ('estimated', 'insufficient_evidence', 'unsupported') or metric.get('value') is None:
        return 'insufficient_evidence', metric.get('reason', 'Measured evidence unavailable')
    return 'measured', metric.get('reason', 'Observed evidence')


def _measure(spec, certificate, context, profile):
    metrics = {m['metric_id']: m for m in certificate.get('metrics', [])}
    goal = context['goal']
    metric = metrics.get(spec.metric_id)
    state, reason = _observed(metric)
    evidence = metric.get('evidence') if metric else None
    if spec.source == 'findings':
        findings = certificate.get('findings', [])
        if spec.metric_id == 'sample_sufficiency':
            findings = [f for f in findings if f['requirement']['metric_id'] in ('row_count', 'required_class_count', 'scenario_count', 'sample_sufficiency')
                and f['requirement'].get('operator') == 'gte' and finite(f['requirement'].get('expected_value')) and f['requirement']['expected_value'] > 1]
        else:
            findings = [f for f in findings if f['requirement']['metric_id'] == spec.metric_id]
        if not findings:
            return None, None, 'not_assessed', 'Define and confirm measurable requirements for this criterion.', None
        if any(f['state'] not in ('pass', 'fail', 'warning') for f in findings):
            return None, None, 'insufficient_evidence', 'At least one confirmed requirement lacks conclusive evidence.', findings
        raw = {'met': sum(f['state'] == 'pass' for f in findings), 'required': len(findings)}
        return 100 * raw['met'] / raw['required'], raw, 'measured', 'Fraction of confirmed requirements met; criteria are not performance estimates.', findings
    if spec.source == 'object_visibility':
        raw=certificate.get('quality_measurements',{}).get('object_visibility')
        if not raw:
            return None,None,'not_assessed','Confirm minimum_object_side_pixels and collect annotated object geometry.',None
        return 100*raw['meeting']/raw['denominator'],raw,'measured',spec.limitations,raw
    if spec.source == 'clarity':
        raw = certificate.get('quality_measurements', {}).get('clarity')
        if not raw or raw['threshold'] != profile.clarity_threshold:
            return None, raw, 'not_assessed', 'Collect complete pixel evidence under this normalization version.', None
        return 100 * raw['meeting'] / raw['denominator'], raw, 'measured', spec.limitations, raw
    if state != 'measured':
        return None, metric.get('value') if metric else None, state, reason, evidence
    raw = metric['value']
    if spec.source in ('timestamps','temporal_order'):
        if spec.source=='timestamps':
            total=raw['timestamp_records']+raw['missing_timestamps']+raw['invalid_timestamps']
            if not total:
                raise ValueError('No timestamp records')
            return 100*raw['timestamp_records']/total,raw,'measured','Parseable recorded timestamps / records; timing accuracy is not independently verified.',evidence
        denominator=raw['adjacent_timestamp_intervals']
        if not denominator:
            return None,raw,'insufficient_evidence','At least one adjacent timestamp pair is required.',evidence
        return 100*(1-raw['backwards_intervals']/denominator),raw,'measured','Non-backwards adjacent timestamps / observed adjacent pairs; repeated timestamps are separately reported.',evidence
    if spec.source=='sensor_completeness':
        sensors=list(raw['sensors'].values())
        if not sensors or any(s.get('status') for s in sensors):
            return None,raw,'insufficient_evidence','Supported complete sensor field mappings are required.',evidence
        population=(evidence or {}).get('sample_size')
        missing_fields=sum(len(s['missing_fields']) for s in sensors)
        present_fields=sum(len(s['missing_records']) for s in sensors)
        if not population or not missing_fields+present_fields:
            return None,raw,'insufficient_evidence','A full-record denominator and declared sensor fields are required.',evidence
        missing=missing_fields*population+sum(sum(s['missing_records'].values()) for s in sensors)
        total=(missing_fields+present_fields)*population
        return 100*(1-missing/total),{'missing_sensor_cells':missing,'declared_sensor_cells':total,'sensor_details':raw},'measured','Completeness of declared sensor fields; calibration validity and physical correctness remain unverified.',evidence
    if spec.source == 'categories':
        key = 'scene' if spec.id == 'scene_coverage' else 'weather'
        required = goal.get('target_environment', {}).get(key)
        if not isinstance(required, list) or not required:
            return None, raw, 'not_assessed', 'Confirm target_environment.' + key + ' categories to define coverage.', evidence
        if not isinstance(raw, dict) or not isinstance(raw.get('counts'), dict):
            return None, raw, 'insufficient_evidence', 'Recorded category counts required; visual classifier estimates are excluded.', evidence
        required = sorted(set(map(str, required)))
        covered = [k for k in required if raw['counts'].get(k, 0) > 0]
        return 100 * len(covered) / len(required), {'counts': raw['counts'], 'required': required, 'covered': covered}, 'measured', spec.limitations, evidence
    if spec.source == 'class_balance':
        if goal.get('uav_parameters', {}).get('class_balance_target') != 'uniform':
            return None, raw, 'not_assessed', 'Balance needs an explicit uniform target; deployment prevalence is not assumed uniform.', evidence
        if not isinstance(raw, dict) or any(not finite(n) or n < 0 for n in raw.values()):
            raise ValueError('Class counts must be finite and nonnegative')
        classes = set(raw) | set(goal.get('required_classes', [])) | set(certificate['dataset'].get('profile', {}).get('declared_classes', []))
        total = sum(raw.values())
        if len(classes) < 2 or total <= 0:
            return None, raw, 'insufficient_evidence', 'At least two declared classes and nonzero labels are required.', evidence
        score = -sum(n/total * math.log(n/total) for n in raw.values() if n) / math.log(len(classes))
        return 100 * score, raw, 'measured', 'Normalized Shannon entropy relative to explicitly requested uniformity.', evidence
    if spec.source == 'annotation_audit':
        audit = raw.get('independent_audit') if isinstance(raw, dict) else None
        if not audit:
            return None, raw, 'insufficient_evidence', 'Supply an independent label audit; annotation syntax does not establish semantic correctness.', evidence
        return normalize(audit['error_fraction'], 'one_minus_fraction'), audit, 'measured', spec.limitations, evidence
    return normalize(raw, spec.normalization), raw, 'measured', reason, evidence


def _weights(overrides, allowed):
    if set(overrides) - set(allowed):
        raise ValueError('Unknown weight IDs: ' + ', '.join(sorted(set(overrides) - set(allowed))))
    if any(not finite(v) or v < 0 or v > 1000 for v in overrides.values()):
        raise ValueError('Weights must be finite numbers in [0, 1000], excluding booleans')


def aggregate_quality(certificate, *, task=None, scenario='default', dimension_weights=None, metric_weights=None,
                      overrides_confirmed=False, profile=None):
    profile = QualityProfile.model_validate(profile) if isinstance(profile, dict) else profile or load_profile()
    context = copy.deepcopy(certificate['confirmed_context'])
    original_task = context['goal']['task_type']
    task = task or original_task
    if task not in profile.tasks:
        raise ValueError('Choose a supported task weighting profile')
    if scenario not in profile.scenarios or task not in profile.scenarios[scenario].tasks:
        raise ValueError('Scenario policy is incompatible with selected task')
    dw, mw = dimension_weights or {}, metric_weights or {}
    _weights(dw, profile.dimensions)
    _weights(mw, [c.id for c in profile.components])
    if (dw or mw) and not overrides_confirmed:
        raise ValueError('Manual weight changes require explicit user confirmation')
    certificate = copy.deepcopy(certificate)
    changed = task != original_task
    if changed:
        # Context-sensitive target counts and requirement findings cannot be relabeled
        # as evidence for a different scientific question.
        certificate['metrics'] = [m for m in certificate.get('metrics', []) if m['metric_id'] not in CONTEXT_DEPENDENT]
        certificate['findings'] = []
    context['goal']['task_type'] = task
    policy, scenario_policy = profile.tasks[task], profile.scenarios[scenario]
    modality = certificate['dataset']['modality']
    components = []
    for spec in profile.components:
        base = policy.metric_weights.get(spec.id, spec.weight)
        multiplier = scenario_policy.metric_multipliers.get(spec.id, 1.)
        effective = base * multiplier * mw.get(spec.id, 1.)
        row = {'id': spec.id, 'dimension': spec.dimension, 'label': spec.label, 'definition': spec.model_dump(),
               'base_weight': base, 'context_multiplier': multiplier, 'priority_multiplier': mw.get(spec.id, 1.),
               'weight': effective, 'effective_weight': 0., 'score': None, 'raw_value': None, 'status': 'not_assessed',
               'normalization': spec.normalization, 'threshold_provenance': spec.provenance,
               'aggregation_log_contribution': None, 'evidence': None}
        if modality not in spec.modalities or spec.tasks and task not in spec.tasks:
            row.update(status='not_applicable', reason='Excluded by documented task/modality applicability rule.')
        else:
            try:
                score, raw, state, reason, evidence = _measure(spec, certificate, context, profile)
                row.update(score=score, raw_value=raw, status=state, reason=reason, evidence=evidence)
            except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
                row.update(status='failed', reason='Invalid measurement: ' + str(exc))
        components.append(row)
    dimensions = []
    for id, name in profile.dimensions.items():
        members = [c for c in components if c['dimension'] == id]
        eligible = [c for c in members if c['status'] != 'not_applicable' and c['weight'] > 0]
        total = sum(c['weight'] for c in eligible)
        measured = [c for c in eligible if c['status'] == 'measured' and c['score'] is not None]
        observed = sum(c['weight'] for c in measured)
        coverage = observed / total if total else 0.
        for c in members:
            c['effective_weight'] = c['weight'] / total if c in eligible and total else 0.
            c['quality_status'] = ('warning' if c['score'] < profile.warning_below else 'pass') if c['score'] is not None else 'not_assessed'
            if c in measured and c['score'] > 0:
                c['aggregation_log_contribution'] = c['effective_weight'] * math.log(c['score'] / 100)
            c['recommendation'] = ('Review criterion failures and supporting observations.' if c in measured and c['score'] < profile.warning_below
                                   else c.get('reason') if c['status'] not in ('measured', 'not_applicable') else None)
        score = geometric([c['score'] for c in measured], [c['weight'] for c in measured]) if coverage == 1 else None
        base = policy.dimension_weights[id]
        factor = scenario_policy.dimension_multipliers.get(id, 1.)
        weight = base * factor * dw.get(id, 1.) if eligible else 0.
        dimensions.append({'id': id, 'name': name, 'score': score, 'evidence_coverage': coverage,
            'status': 'not_applicable' if not eligible else 'not_assessed' if not measured else 'insufficient_evidence' if coverage < 1 else 'warning' if score < profile.warning_below or any(c['quality_status']=='warning' for c in measured) else 'pass',
            'base_weight': base, 'context_multiplier': factor, 'priority_multiplier': dw.get(id, 1.), 'weight': weight,
            'effective_weight': 0., 'metrics': members,
            'deficiencies': [c['label'] + ': ' + c['reason'] for c in eligible if c['status'] != 'measured' or c['score'] < profile.warning_below],
            'recommendations': [c['label'] + ': ' + c['recommendation'] for c in eligible if c['recommendation']]})
    total = sum(d['weight'] for d in dimensions)
    if not total:
        raise ValueError('At least one applicable dimension must retain positive importance')
    for dimension in dimensions:
        dimension['effective_weight'] = dimension['weight'] / total
    gates = []
    for c in components:
        severity = c['definition'].get('gate')
        if severity and c['status'] != 'not_applicable':
            gates.append({'id': c['id'], 'severity': severity, 'kind': 'technical' if severity == 'block' else 'review',
                'status': 'pass' if c['score'] == 100 else 'fail' if c['score'] is not None else 'insufficient_evidence',
                'reason': c['label'] + ': ' + c['reason'], 'raw_value': c['raw_value'],
                'criterion': 'Required absence of observed defects; independent of assigned importance weights.'})
    for f in certificate.get('findings', []):
        if f['requirement']['severity'] in ('hard', 'preferred'):
            gates.append({'id': f['requirement']['requirement_id'], 'severity': 'block' if f['requirement']['severity'] == 'hard' else 'warning',
                'kind': 'confirmed_requirement', 'status': f['state'], 'reason': f['reason'], 'requirement': f['requirement'],
                'raw_value': f['result'].get('value')})
    if changed:
        gates.append({'id': 'TASK-EVIDENCE', 'severity': 'block', 'kind': 'evidence', 'status': 'insufficient_evidence',
                      'reason': 'Task changed; collect and confirm task-specific labels and requirements before making a readiness decision.'})
    if task in ('object_detection','tracking') and modality!='image' or task=='telemetry' and modality!='tabular':
        gates.append({'id':'TASK-MODALITY','severity':'block','kind':'technical','status':'fail',
                      'reason':'The ingested modality does not supply the required image or telemetry records for this task.'})
    blocked = any(g['severity'] == 'block' and g['status'] == 'fail' for g in gates)
    missing = any(d['status'] in ('not_assessed', 'insufficient_evidence') and d['weight'] > 0 for d in dimensions)
    missing |= any(g['severity'] == 'block' and g['status'] not in ('pass', 'fail') for g in gates)
    status = 'blocked' if blocked else 'insufficient_evidence' if missing else 'conditional' if any(g['status'] != 'pass' for g in gates) or any(d['status'] == 'warning' for d in dimensions) else 'ready'
    result = {'task': task, 'scenario': scenario, 'readiness_status': status, 'dimensions': dimensions, 'critical_requirements': gates,
        'critical_issues': [g for g in gates if g['status'] != 'pass'], 'individual_metrics': components,
        'raw_metrics': certificate.get('metrics', []), 'original_context': certificate['confirmed_context'],
        'requires_measurement': changed, 'measurement_requirements': [c['id'] for c in components if c['status'] in ('not_assessed', 'insufficient_evidence', 'failed')],
        'profile_version': profile.profile_version, 'normalization_version': profile.normalization_version,
        'aggregation_version': profile.aggregation_version, 'profile': profile.model_dump(), 'profile_sha256': fingerprint(profile.model_dump()),
        'weight_explanations': [policy.explanation, scenario_policy.explanation, profile.provenance],
        'confirmed_overrides': {'dimensions': dw, 'metrics': mw}, 'warning_below': profile.warning_below,
        'formula': '100 * exp(sum(w_i * ln(s_i/100))); zero preserved; score withheld unless evidence coverage is 1.',
        'limitations': ['Dimension importance is descriptive; dimensions are never collapsed into a primary overall percentage.',
            'Geometric aggregation penalizes weaknesses within a dimension; independent gates and separate dimension reporting prevent cross-dimension compensation.',
            'Full weighted coverage means configured criteria were measured; it does not establish scientific completeness or empirical validity.',
            'Engineering weights and warning threshold are uncalibrated. Missing evidence is not zero quality.'],
        'dataset_sha256': certificate['dataset']['sha256']}
    result['calculation_sha256'] = fingerprint(result)
    return result
