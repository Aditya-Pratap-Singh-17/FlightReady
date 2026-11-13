"""Requirement-based decisions. Metrics never vote or average into readiness."""
import hashlib
import importlib.metadata
import json
import platform
from typing import Any
from backend.context import Goal, Requirement, AssessmentFinding, Certificate, RemediationRecommendation, RequirementOperator, RequirementSeverity
from metrics.core import run_metric, DIAGNOSTICS, REGISTRY

ENGINE_VERSION = '1.6.0'

def build_requirements(g: Goal):
    requirements = list(g.requirements)
    def add(description: str, id: str, expected_value: Any=True,
            operator: RequirementOperator='equals', severity: RequirementSeverity='hard',
            *, metric_parameters: dict[str, Any] | None=None, **params: Any):
        if metric_parameters:
            params.update(metric_parameters)
        sequence=len(requirements)+1
        existing={r.requirement_id for r in requirements}
        while f'AUTO-{sequence:03}' in existing: sequence+=1
        requirements.append(Requirement(requirement_id=f'AUTO-{sequence:03}',description=description,
            metric_id=id,expected_value=expected_value,operator=operator,severity=severity,source='inferred',parameters=params))
    add('Dataset has observations','row_count',1,'gte')
    for c in g.required_inputs:
        add(f'Required input exists: {c}','field_presence',column=c)
    for c in g.trash_inputs:
        add(f'Excluded input is not used: {c}','excluded_input_usage',column=c)
    for modality in g.input_modalities:
        if modality in ('rgb','thermal','gps','imu','multisensor'):
            add(f'Declared sensor evidence: {modality}','sensor_modality',modality=modality)
        elif modality not in ('image','tabular','text'):
            add(f'Evidence for unsupported modality: {modality}','unsupported_modality')
    if g.task_type in ('classification','regression'):
        add(f'Prediction target exists: {g.prediction_target or "unspecified"}','field_presence',column=g.prediction_target)
        add('Supervised learning has usable target observations','usable_target_count',1,'gte')
    for c,limit in g.max_missing_fraction.items():
        add(f'Missingness of {c} is at most {limit}','missingness',limit,'lte',column=c)
    for c,expected in g.expected_types.items():
        add(f'{c} conforms to {expected}','type_validity',1.0,column=c,expected=expected)
    for c,limits in g.valid_ranges.items():
        add(f'{c} respects declared valid range','range_violations',0.0,column=c,range=limits)
    for c,values in g.allowed_values.items():
        add(f'{c} uses allowed values','allowed_violations',0.0,column=c,allowed=values)
    for c,unit in g.expected_units.items():
        add(f'{c} has declared unit {unit}','unit_consistency',column=c,unit=unit)
    if g.max_data_age_days is not None:
        add('Observations respect the declared maximum age','timeliness',1.0)
    # Manual environment edits are also binding even if no scenario was created
    # by the extractor. They are compiled without mutating the confirmed Goal.
    if g.target_environment:
        def represented(column,condition):
            observed=[s.conditions[column] for s in g.scenarios if column in s.conditions]
            if condition in observed: return True
            return isinstance(condition,list) and all([v] in observed or v in observed for v in condition)
        unrepresented={c:v for c,v in g.target_environment.items() if not represented(c,v)}
        if unrepresented:
            add('Declared deployment environment','scenario_count',1,'gte',conditions=unrepresented)
    for cls in g.required_classes:
        add(f'Dataset contains annotated {cls}','required_class_count',1,'gte',metric_parameters={'class':cls})
    for scenario in g.scenarios:
        add(scenario.name,'scenario_count',scenario.minimum_samples or 1,'gte',scenario.severity,
            conditions=scenario.conditions,required_class=scenario.required_class)
        # Critical-class intersections must be measured, not inferred from marginals.
        for cls in g.required_classes:
            if scenario.required_class is None:
                add(f'{scenario.name}, class {cls}','scenario_count',1,'gte',scenario.severity,
                    conditions=scenario.conditions,required_class=cls)
    for requested,field in [(g.require_source,'source'),(g.require_collection_date,'collection_date'),(g.require_dataset_version,'dataset_version')]:
        if requested: add(f'Provenance evidence: {field}','provenance',field=field)
    if g.task_type == 'object_detection':
        add('Detection image files are readable','unreadable_rate',0.0)
        add('Detection annotations are syntactically valid','bbox_invalid_rate',0.0)
        add('Detection images have annotation files, including explicit negatives','missing_label_rate',0.0)
    if g.task_type == 'tracking':
        columns=g.uav_parameters.get('columns',{})
        for field,default in [('timestamp',g.date_column or 'timestamp'),('sequence',g.group_column or 'sequence_id'),('track','track_id')]:
            add('Tracking evidence field: '+field,'field_presence',column=columns.get(field,default))
    return requirements

def evaluate(req,result):
    if result.state in ('insufficient_evidence','error','not_applicable') or result.value is None:
        state = result.state if result.state != 'not_applicable' or req.severity != 'hard' else 'insufficient_evidence'
        return AssessmentFinding(requirement=req,result=result,state=state,reason=result.reason)
    a,b = result.value,req.expected_value
    if req.metric_id.startswith('UAV-') and isinstance(a,dict) and req.operator!='contains' and not isinstance(b,dict):
        return AssessmentFinding(requirement=req,result=result,state='insufficient_evidence',
            reason='Select an exact output_path and confirm an appropriate criterion for this structured property; successful calculation is not a pass.')
    if result.assessment_status == 'estimated' and not req.parameters.get('accept_estimated_evidence',False):
        return AssessmentFinding(requirement=req,result=result,state='insufficient_evidence',reason='This confirmed requirement does not explicitly accept model-estimated evidence.')
    try:
        # Python's False == 0 is unsuitable for scientific evidence comparisons.
        if req.operator == 'equals':
            passed = a == b and (isinstance(a,bool) == isinstance(b,bool))
        elif req.operator == 'contains': passed = b in a
        elif req.operator == 'between':
            if isinstance(a,bool) or not isinstance(a,(int,float)): raise ValueError()
            passed = b[0] <= a <= b[1]
        else:
            if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(int,float)) or not isinstance(b,(int,float)): raise ValueError()
            passed = a <= b if req.operator == 'lte' else a >= b
        state = 'pass' if passed else 'warning' if req.severity == 'diagnostic' else 'fail'
        if req.metric_id=='scenario_count' and result.evidence.details.get('unknown_count') and not (req.operator=='gte' and passed):
            state='insufficient_evidence'
        reason = f'Observed {a!r}; required {req.operator} {b!r}. {result.reason}'
    except (TypeError,ValueError,IndexError):
        state,reason = 'error','Requirement operator and evidence types are incompatible'
    return AssessmentFinding(requirement=req,result=result,state=state,reason=reason)

def decide(findings,metrics,goal):
    hard = [f for f in findings if f.requirement.severity == 'hard']
    # A confirmed failure establishes NOT_READY even if another check is unknown.
    if any(f.state == 'fail' for f in hard): return 'NOT_READY'
    if any(f.state == 'error' for f in findings) or any(m.state == 'error' for m in metrics): return 'ASSESSMENT_ERROR'
    if goal.task_type == 'unspecified' or any(f.state != 'pass' for f in hard): return 'INSUFFICIENT_EVIDENCE'
    if goal.task_type == 'object_detection' and not goal.required_classes: return 'INSUFFICIENT_EVIDENCE'
    if any(f.state in ('fail','warning','insufficient_evidence') and f.requirement.severity == 'preferred' for f in findings): return 'CONDITIONALLY_READY'
    return 'READY'

def assess(ds,context,initial=None,reference=None,progress=lambda stage:None):
    from metrics.uav_metrics.vision import vision_session
    with vision_session(progress):
        return _assess(ds,context,initial,reference,progress)


def _assess(ds,context,initial=None,reference=None,progress=lambda stage:None):
    if not context.confirmed:
        raise ValueError('Requirements must be explicitly confirmed before assessment')
    if context.clarification_questions:
        raise ValueError('Resolve clarification questions before confirmation')
    goal = context.goal
    progress('Metrics selected')
    findings = []
    requirements=build_requirements(goal)
    if ds.modality=='image' and goal.task_type!='object_detection':
        requirements.append(Requirement(requirement_id='IMAGE-READABLE',description='Image inputs can be decoded',metric_id='unreadable_rate',expected_value=0.0,source='inferred'))
    if goal.task_type=='object_detection' and ds.profile.get('annotation_format') in ('coco','pascal_voc'):
        requirements.append(Requirement(requirement_id='MANIFEST',description='Annotation references are consistent',metric_id='annotation_manifest_errors',expected_value=0,source='inferred'))
    for req in requirements:
        findings.append(evaluate(req,run_metric(req.metric_id,ds,goal,req.parameters,reference)))
    progress('Profiling completed')
    from metrics.uav_metrics.registry import PROPERTY_IDS
    selected=list(DIAGNOSTICS)
    if ds.modality=='image' or goal.task_type in ('object_detection','tracking','telemetry') or goal.uav_parameters:
        selected+=PROPERTY_IDS
    metrics = [run_metric(id,ds,goal,reference=reference) for id in selected]
    from metrics.uav_metrics.learning import learning_views, recommendations as uav_recommendations
    views=learning_views(ds,goal,metrics) if any(mid in selected for mid in PROPERTY_IDS) else {}
    progress('Quality checks completed')
    # Explicit modality conflicts are requirements, not generic quality penalties.
    requested = set(goal.input_modalities)
    if 'image' in requested and ds.modality != 'image' or 'tabular' in requested and ds.modality != 'tabular' or 'text' in requested and ds.modality!='tabular':
        req = Requirement(requirement_id='MODALITY',description='Dataset supports requested modality',metric_id='modality')
        from backend.context import MetricResult,AssessmentEvidence
        m = MetricResult(metric_id='modality',dimension='integrity',state='fail',value=False,reason='Dataset format does not supply the requested modality',evidence=AssessmentEvidence(method='Compare requested and ingested modalities'))
        findings.append(AssessmentFinding(requirement=req,result=m,state='fail',reason=m.reason))
    status = decide(findings,metrics,goal)
    from metrics.experimental_scoring import calculate
    experimental_score = calculate(ds,goal,metrics,findings)
    progress('Readiness decision generated')
    passed = sum(f.state == 'pass' for f in findings)
    failed = sum(f.state == 'fail' for f in findings)
    unknown = sum(f.state == 'insufficient_evidence' for f in findings)
    benefits = [f'Supported: {f.requirement.description}' for f in findings if f.state == 'pass']
    distribution = next(m for m in metrics if m.metric_id == 'class_distribution')
    if isinstance(distribution.value,dict):
        extras = sorted(set(distribution.value)-set(goal.required_classes))
        if extras: benefits.append('Additional observed classes: ' + ', '.join(extras))
    limitations = [m.reason for m in metrics if m.state == 'insufficient_evidence']
    limitations += ['READY means the confirmed, measurable requirements passed; it does not establish model accuracy, sufficient training size, or deployment safety.',
        'Statistical outliers are not classified as corruption without declared valid ranges or platform specifications.',
        'Annotation completeness inside an image and true object-label correctness are unverified without independent audit.',
        'Dataset metadata and reference relevance are user-declared; their authenticity is not independently verified.']
    if ds.disk_path:
        limitations.append('Large-data checks query the full disk-backed table. The first 100 rows are a display preview only. Whole-dataset reference-distance comparisons are not implemented for this backend; unsupported checks remain unverified.')
    if goal.priorities:
        limitations.append('Qualitative priorities are recorded as preferences; dataset checks cannot establish high model accuracy or a false-negative target.')
    if goal.performance_objectives:
        limitations.append('Performance objectives are user-declared and unverified: accuracy, recall, generalization and robustness require independent target-model training/evaluation.')
    for field in ['target_population','prediction_horizon','relevant_time_period','geography']:
        if getattr(goal,field):
            limitations.append(f'{field}: recorded as context, not automatically verified; add measurable scenario requirements or relevant reference evidence.')
    recommendations = [RemediationRecommendation(description=('Repair or supply evidence for: ' + f.requirement.description),finding_ids=[f.requirement.requirement_id]) for f in findings if f.state in ('fail','insufficient_evidence','error')]
    recommendations+=uav_recommendations(metrics)
    if any(m.metric_id == 'duplicate_rate' and m.value and m.value > 0 for m in metrics):
        limitations.append('Exact repeated observations were detected; their legitimacy and effect on effective sample size require review.')
        recommendations.append(RemediationRecommendation(description='Review repeated records and construct evaluation splits by independent flight/session where appropriate.',conditional=True))
    def visual_signatures(value):
        if isinstance(value,dict):
            return [v for k,v in value.items() if k.endswith('configuration_sha256') and isinstance(v,str)]+[s for v in value.values() for s in visual_signatures(v)]
        if isinstance(value,list):return [s for v in value for s in visual_signatures(v)]
        return []
    config = {'near_duplicate_distance':ds.near_distance,'engine_version':ENGINE_VERSION,
              'experimental_score_profile_sha256':experimental_score.get('profile_sha256'),
              'uav_implementation_version':'research-evidence-2.0',
              'uav_metric_versions':{m.metric_id:m.reproducibility.get('implementation_version') for m in metrics if m.metric_id.startswith('UAV-')},
              'visual_models':{m.metric_id:sorted(set(visual_signatures(m.value))) for m in metrics if visual_signatures(m.value)},
              'storage_backend':ds.profile.get('storage_backend','pandas'),
              'processing':ds.profile.get('processing',{}),
              'source_container':ds.profile.get('source_archive') or ds.profile.get('compressed_source')}
    context_json = context.model_dump_json()
    versions = {}
    for name in ['pandas','numpy','Pillow','pydantic','scipy','scikit-learn','pyarrow','duckdb','ijson','defusedxml','openpyxl','PyYAML','onnxruntime','onnxruntime-gpu']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    fingerprint = hashlib.sha256((ds.sha256+context_json+json.dumps(config,sort_keys=True)+(reference.sha256 if reference else '')+json.dumps(ds.metadata,sort_keys=True)+json.dumps(versions,sort_keys=True)).encode()).hexdigest()
    certificate = Certificate(dataset={'name':ds.name,'sha256':ds.sha256,'version':ds.metadata.get('dataset_version'),'modality':ds.modality,'profile':ds.profile,
        'provenance':{'source_declared':bool(ds.metadata.get('source')),
            'source_description_sha256':hashlib.sha256(str(ds.metadata['source']).encode()).hexdigest() if ds.metadata.get('source') else None,
            'collection_date':ds.metadata.get('collection_date'),'authenticity':'user-declared; not independently verified'}},
        initial_context=initial,confirmed_context=context,overall_status=status,
        executive_summary=f'{status}: {passed} requirements passed, {failed} failed and {unknown} lack evidence. Readiness is relative to your confirmed goal.',
        findings=findings,metrics=metrics,additional_benefits=benefits,limitations=list(dict.fromkeys(limitations)),recommendations=recommendations,
        learning_views=views,
        experimental_score=experimental_score,
        reproducibility={'engine_version':ENGINE_VERSION,'assessment_fingerprint':fingerprint,'configuration':config,'python':platform.python_version(),
                         'dependencies':versions,'reference_sha256':reference.sha256 if reference else None,
                         'dataset_metadata':{k:v for k,v in ds.metadata.items() if k in ('dataset_version','collection_date','units','sensor_types','collection_kind','altitude_datum')},
                         'selected_metric_ids':list(dict.fromkeys([f.result.metric_id for f in findings]+selected)),
                         'registry':{id:REGISTRY[id].model_dump() for id in REGISTRY if id in selected or id in [f.result.metric_id for f in findings]}})
    from metrics.quality_aggregation import collect_measurements, aggregate_quality
    certificate.quality_measurements = collect_measurements(ds, goal=goal)
    from metrics.uav_metrics.presentation import evidence_sections
    certificate.quality_measurements['research_evidence'] = evidence_sections([m.model_dump(mode='json') for m in certificate.metrics])
    if goal.task_type != 'unspecified':
        certificate.quality_profile = aggregate_quality(certificate.model_dump(mode='json'), **goal.quality_policy.model_dump())
        certificate.reproducibility['configuration']['adaptive_quality_profile_sha256'] = certificate.quality_profile['profile_sha256']
        certificate.reproducibility['assessment_fingerprint'] = hashlib.sha256((fingerprint+certificate.quality_profile['calculation_sha256']).encode()).hexdigest()
        # Versioned v2 projection reuses the same measured components.  It does
        # not replace the historical experimental score or v1 adaptive profile.
        from metrics.readiness_v2 import V2Metric, build_scorecard
        projected = []
        for component in certificate.quality_profile['individual_metrics']:
            status = component['status']
            projected.append(V2Metric(
                metric_id=component['id'], dimension=component['dimension'], label=component['label'],
                weight=float(component['weight']) if component['weight'] > 0 else 1.0,
                status='measured' if status == 'measured' else 'not_applicable' if status == 'not_applicable' else 'error' if status == 'failed' else 'unavailable',
                score=component['score'] if status == 'measured' else None,
                raw_value=component['raw_value'], normalization=component['normalization'],
                evidence_type='observed_or_confirmed_requirement',
                limitation=component['definition']['limitations'],
                gate=component['definition'].get('gate'),
                gate_passed=(component['score'] == 100) if component['definition'].get('gate') and status == 'measured' else None,
            ))
        certificate.quality_profile_v2 = build_scorecard(projected)
    from backend.compatibility import compatibility
    from backend.final_report import build_final_report
    certificate.task_compatibility = compatibility(ds, context)
    certificate.final_report = build_final_report(certificate.model_dump(mode='json'), certificate.task_compatibility)
    return certificate
