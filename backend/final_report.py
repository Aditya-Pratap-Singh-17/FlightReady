"""Canonical assessment snapshot. Empirical outcomes are linked separately.

Existing certificate decisions and mathematical policies remain unchanged. This
versioned interpretation deliberately withholds an overall numerical index.
"""
import copy
import hashlib
import json
import math

VERSION = 'final-readiness-1.0'


def dimension_bounds(dimension):
    """Bounds on the EXISTING geometric criterion index, not confidence limits."""
    members = [m for m in dimension['metrics'] if m['effective_weight'] > 0]
    if not members:
        return None
    known = [m for m in members if m['status'] == 'measured' and m['score'] is not None]
    if any(m['score'] == 0 for m in known):
        return [0., 0.]
    upper = 100 * math.exp(sum(m['effective_weight'] * math.log(m['score'] / 100) for m in known))
    return [upper if len(known) == len(members) else 0., upper]


def build_final_report(certificate, task_compatibility=None):
    cert = copy.deepcopy(certificate)
    profile = cert.get('quality_profile') or {}
    compat = task_compatibility or cert.get('task_compatibility') or {
        'status': 'unverified', 'summary': 'Compatibility preview was not recorded in this historical assessment.',
        'dataset_sha256': cert['dataset']['sha256'], 'checks': [], 'dataset_role': 'unspecified'}
    gates = profile.get('critical_requirements', [])
    hard = [g for g in gates if g['severity'] == 'block']
    # Historical certificates may have no adaptive profile. Never assume missing checks passed.
    if not gates:
        hard = [{'id': f['requirement']['requirement_id'], 'status': f['state'], 'reason': f['reason'],
                 'severity': 'block', 'requirement': f['requirement']} for f in cert['findings'] if f['requirement']['severity'] == 'hard']
    failures = [g for g in hard if g['status'] == 'fail']
    gaps = [g for g in hard if g['status'] not in ('pass', 'fail')]
    dimensions = [{k: d[k] for k in ('id', 'name', 'score', 'status', 'evidence_coverage', 'effective_weight', 'deficiencies', 'recommendations')}
                  | {'bounds': dimension_bounds(d)} for d in profile.get('dimensions', [])]
    coverage = sum(d['effective_weight'] * d['evidence_coverage'] for d in dimensions) if dimensions else None
    complete = bool(dimensions) and all(d['score'] is not None for d in dimensions if d['effective_weight'] > 0)
    if failures or compat['status'] == 'incompatible' or cert['overall_status'] == 'NOT_READY':
        decision = 'NOT_READY'
    elif gaps or not complete or compat['status'] != 'supported' or cert['overall_status'] != 'READY' or profile.get('readiness_status') != 'ready':
        decision = 'INDETERMINATE'
    else:
        decision = 'READY_FOR_DEFINED_USE'
    explanation = {'NOT_READY': 'At least one mandatory requirement is confirmed unmet. A quality score cannot override this blocker.',
                   'INDETERMINATE': 'The available evidence does not establish readiness for the defined use. Complete the missing evidence or resolve the recorded conditions.',
                   'READY_FOR_DEFINED_USE': 'The declared requirements and configured quality criteria are supported for this defined use. Model success and deployment safety are not established.'}[decision]
    actions = []
    for gate in failures + gaps:
        requirement = gate.get('requirement', {})
        metric = requirement.get('metric_id', '')
        independence = {'split_integrity': ('Exact content overlap between development and test partitions has not been established.', 'Declare split metadata and run the exact-content leakage audit on the complete partition manifest.'),
                        'group_integrity': ('Recording-group independence across partitions has not been established.', 'Supply flight/session identifiers and audit their assignment across partitions.')}
        action = ('Collect annotated training/evaluation data for the required class.' if metric == 'required_class_count' else
                  'Supply recorded scenario metadata or an independently validated audit; reassess the required intersection.' if metric == 'scenario_count' else
                  'Repair the confirmed defect and create a new dataset version.' if gate in failures else
                  'Collect the missing measurement and create a new assessment.')
        reason = gate['reason']
        if gate in gaps and gate['id'] in independence:
            reason,action = independence[gate['id']]
        actions.append({'priority': 'critical failure' if gate in failures else 'mandatory evidence gap',
                        'finding_id': gate['id'], 'reason': reason, 'action': action, 'effort': 'Requires data review; not estimated'})
    for d in sorted(dimensions, key=lambda d: d['effective_weight'], reverse=True):
        if d['effective_weight'] and d['deficiencies']:
            actions.append({'priority': 'quality evidence', 'finding_id': d['id'], 'reason': d['deficiencies'][0],
                            'action': d['recommendations'][0] if d['recommendations'] else 'Inspect this dimension and collect supporting evidence.',
                            'effort': 'Requires review of the measurement method'})
    role = compat.get('dataset_role')
    validation_blocked = compat['status'] != 'supported'
    report = {'report_version': VERSION, 'assessment_fingerprint': cert['reproducibility'].get('assessment_fingerprint'),
              'dataset': {k: cert['dataset'].get(k) for k in ('name', 'sha256', 'version', 'modality')},
              'intended_task': cert['confirmed_context']['goal'], 'compatibility': compat,
              'decision': decision, 'conclusion': explanation, 'technical_requirement_status': cert['overall_status'],
              'criterion_attainment_index': {'score': None, 'status': 'not_scored',
                  'reason': 'The frozen policy defines six geometric dimension indices, not a validated overall index. No universal readiness percentage is inferred.'},
              'evidence_coverage': coverage, 'coverage_definition': 'Sum of dimension importance times within-dimension measured criterion weight; excludes legitimate inapplicability. Not statistical confidence.',
              'evidence_strength': {'status': 'not_calibrated', 'reason': 'Inspect measurement provenance and independent audits. Coverage alone is not evidence strength.'},
              'critical_blockers': len(failures), 'hard_requirements': hard, 'confirmed_failures': failures, 'evidence_gaps': gaps,
              'dimensions': dimensions, 'top_problems': actions[:3], 'recommendations': actions,
              'validation': {'status': 'blocked' if validation_blocked else 'not_yet_validated',
                  'inference_allowed': not validation_blocked and cert['confirmed_context']['goal']['task_type'] == 'object_detection',
                  'training_allowed': not validation_blocked and role not in ('val', 'test') and cert['confirmed_context']['goal']['task_type'] == 'object_detection',
                  'reason': 'Resolve target/condition compatibility before testing this goal.' if validation_blocked else
                            'Validation-only/test-only data cannot train a model. Supply a separate training dataset.' if role in ('val', 'test') else
                            'Model and label compatibility, valid ground truth and independent partitions are checked again before execution.'},
              'limitations': [*compat.get('limitations', []), *cert['limitations'],
                  'Incomplete-dimension ranges are conservative bounds on the frozen geometric scoring index, not confidence intervals.',
                  'Linked model outcomes never rewrite this immutable analytical snapshot.'],
              'method': {'profile_sha256': profile.get('profile_sha256'), 'aggregation_version': profile.get('aggregation_version'),
                         'normalization_version': profile.get('normalization_version'), 'formula': profile.get('formula')},
              'appendix_reference': 'Full certificate metrics, registry, findings, quality_profile and reproducibility are retained in the same JSON export.'}
    report['dataset']['version'] = report['dataset'].get('version') or report['dataset']['sha256']
    report['report_sha256'] = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return report
