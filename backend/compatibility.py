"""Cheap compatibility preview using the same requirement measurements as assessment.

This checks annotated support, never whether an object/condition exists in reality.
No pixel-derived lighting guess is substituted for recorded scenario evidence.
"""
from backend.readiness_engine import build_requirements, evaluate
from metrics.core import run_metric

VERSION = 'task-compatibility-1.0'


def compatibility(ds, context):
    goal = context.goal
    checks = []
    selected = {'required_class_count', 'scenario_count', 'field_presence', 'usable_target_count',
                'unreadable_rate', 'bbox_invalid_rate', 'missing_label_rate'}
    for req in build_requirements(goal):
        if req.metric_id in selected:
            checks.append(evaluate(req, run_metric(req.metric_id, ds, goal, req.parameters)).model_dump(mode='json'))
    image_task = goal.task_type in ('object_detection', 'tracking')
    modality_failed = image_task and ds.modality != 'image' or 'image' in goal.input_modalities and ds.modality != 'image'
    failed = [c for c in checks if c['requirement']['severity'] == 'hard' and c['state'] == 'fail']
    unknown = [c for c in checks if c['requirement']['severity'] == 'hard' and c['state'] not in ('pass', 'fail')]
    unspecified = goal.task_type == 'unspecified' or image_task and not goal.required_classes
    status = 'incompatible' if failed or modality_failed else 'unverified' if unknown or unspecified else 'supported'
    declared = list(ds.profile.get('declared_classes') or ds.profile.get('class_counts', {}))
    role = ds.profile.get('source_split')
    if not role and ds.profile.get('annotation_format') == 'visdrone_det':
        wrapper = ds.profile.get('archive_root', '').rstrip('/').split('/')[-1].lower()
        role = next((value for suffix,value in (('-train','train'),('-val','val'),('-test-dev','test'),('-test-challenge','test')) if wrapper.endswith(suffix)), None)
    if not role and 'split' in ds.frame:
        roles = {str(v).lower() for v in ds.frame['split'].dropna()}
        aliases = {'validation': 'val', 'valid': 'val', 'training': 'train', 'testing': 'test'}
        roles = {aliases.get(r, r) for r in roles}
        role = next(iter(roles)) if len(roles) == 1 else 'mixed' if roles else None
    summary = {'incompatible': 'The supplied annotations or modality do not support the defined task.',
               'unverified': 'Task support is not yet established; required evidence is missing.',
               'supported': 'The recorded labels and checked conditions support this task; model performance remains untested.'}[status]
    if ds.profile.get('annotation_format') == 'visdrone_det' and any(c in ('drone', 'drones', 'uav') for c in goal.required_classes):
        summary = ('This dataset contains drone-captured imagery but does not provide drone-object annotations. '
                   'It cannot currently establish readiness for detecting drones' + (' at night.' if goal.target_environment.get('lighting') else '.'))
    return {'version': VERSION, 'status': status, 'summary': summary, 'dataset_sha256': ds.sha256,
            'task': goal.task_type, 'required_classes': goal.required_classes, 'annotated_classes': declared,
            'dataset_role': role or 'unspecified', 'training_partition_available': role not in ('val', 'test'),
            'checks': checks, 'confirmed_failures': len(failed) + int(bool(modality_failed)), 'evidence_gaps': len(unknown),
            'limitations': ['Annotated class support is distinct from objects appearing in imagery.',
                           'Missing operating-condition metadata establishes neither presence nor absence.',
                           'This preview does not establish sample adequacy, semantic label correctness or model compatibility.']}
