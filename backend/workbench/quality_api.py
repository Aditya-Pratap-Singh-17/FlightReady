"""Frozen evidence reaggregation, version history, task previews and sensitivity."""
from fastapi import APIRouter
from pydantic import Field
from backend.context import Model
from backend.workbench.contracts import RecordResponse
from metrics.quality_profiles import load_profile
from metrics.quality_aggregation import aggregate_quality
from backend.workbench.api import ErrorRoute

router = APIRouter(prefix='/api/v1/quality', tags=['Adaptive quality'], route_class=ErrorRoute)


class AggregationRequest(Model):
    readiness_id: str
    task: str | None = None
    scenario: str = 'default'
    dimension_weights: dict[str, object] = Field(default_factory=dict, max_length=32)
    metric_weights: dict[str, object] = Field(default_factory=dict, max_length=64)
    overrides_confirmed: bool = False
    persist: bool = False
    use_current_profile: bool = False


class ComparisonRequest(Model):
    readiness_id: str
    contexts: list[AggregationRequest] = Field(min_length=2, max_length=8)


def store():
    from backend.workbench.api import services
    return services()[0]


def certificate(id):
    job = store().get(id, 'job')
    if job['payload']['task'] != 'readiness' or job['payload']['status'] != 'completed':
        raise ValueError('A completed readiness assessment is required')
    return job['payload']['result']


@router.get('/registry', response_model=dict)
def registry():
    from metrics.core import REGISTRY
    profile = load_profile()
    raw={}
    for id,definition in REGISTRY.items():
        assignments=[{'component_id':c.id,'dimension':c.dimension,'quality_direction':c.direction,'normalization':c.normalization,
                      'tasks':c.tasks,'measurement_requirements':c.requirements} for c in profile.components if c.metric_id==id]
        raw[id]={**definition.model_dump(),'score_assignments':assignments,
                 'quality_direction':'see component-specific normalization' if assignments else 'descriptive; no universal monotonic quality direction established'}
    return {'dimensions': profile.dimensions, 'components': [c.model_dump() for c in profile.components],
            'raw_metrics': raw,
            'notice': 'Raw diagnostics without a defensible quality direction remain inspectable and are not forced into numerical scores.'}


@router.get('/profiles', response_model=dict)
def profiles():
    return load_profile().model_dump()


@router.post('/aggregate', response_model=dict)
def aggregate(body: AggregationRequest):
    cert = certificate(body.readiness_id)
    frozen = cert.get('quality_profile')
    result = aggregate_quality(cert, task=body.task, scenario=body.scenario,
        dimension_weights=body.dimension_weights, metric_weights=body.metric_weights,
        overrides_confirmed=body.overrides_confirmed,
        profile=None if body.use_current_profile else frozen.get('profile') if frozen else None)
    result.update(readiness_id=body.readiness_id, preview_only=not body.persist)
    if body.persist:
        saved = store().create('aggregation', {'request': body.model_dump(), 'result': result,
                                              'source_assessment_fingerprint': cert.get('reproducibility', {}).get('assessment_fingerprint')})
        result['aggregation_id'] = saved['id']
    return result


@router.get('/history', response_model=list[RecordResponse])
def history(readiness_id: str | None = None):
    return [r for r in store().list('aggregation') if not readiness_id or r['payload']['request']['readiness_id'] == readiness_id]


@router.get('/{readiness_id}/dimensions', response_model=dict)
def dimensions(readiness_id: str):
    cert = certificate(readiness_id)
    return cert.get('quality_profile') or aggregate_quality(cert)


@router.post('/compare', response_model=dict)
def compare(body: ComparisonRequest):
    if any(c.readiness_id != body.readiness_id for c in body.contexts):
        raise ValueError('Task comparison must use the same frozen assessment')
    results = [aggregate(c.model_copy(update={'persist': False})) for c in body.contexts]
    return {'readiness_id': body.readiness_id, 'results': results,
            'notice': 'Changed tasks reuse task-independent evidence. Collect missing task-dependent measurements before interpreting a preview as an assessment.'}


@router.post('/sensitivity', response_model=dict)
def sensitivity(body: AggregationRequest):
    base = aggregate(body.model_copy(update={'persist': False}))
    results = []
    for dimension in base['dimensions']:
        if not dimension['weight']:
            continue
        for factor in (.5, 2.):
            weights = {**body.dimension_weights, dimension['id']: float(body.dimension_weights.get(dimension['id'], 1)) * factor}
            result = aggregate(body.model_copy(update={'persist': False, 'dimension_weights': weights, 'overrides_confirmed': True}))
            results.append({'dimension': dimension['id'], 'multiplier': factor,
                'importance': {d['id']: d['effective_weight'] for d in result['dimensions']},
                'scores': {d['id']: d['score'] for d in result['dimensions']}, 'status': result['readiness_status'],
                'importance_ranking': [d['id'] for d in sorted(result['dimensions'], key=lambda d: d['effective_weight'], reverse=True)]})
    from metrics.quality_aggregation import geometric
    deterioration=[]
    for dimension in base['dimensions']:
        members=[m for m in dimension['metrics'] if m['effective_weight']>0]
        if dimension['score'] is None:
            continue
        for metric in members:
            if metric['status']!='measured':
                continue
            for factor in (.5,0.):
                simulated=[m['score']*factor if m['id']==metric['id'] else m['score'] for m in members]
                deterioration.append({'dimension':dimension['id'],'component':metric['id'],'quality_multiplier':factor,
                    'observed_dimension_score':dimension['score'],
                    'hypothetical_dimension_score':geometric(simulated,[m['effective_weight'] for m in members]),
                    'hypothetical_critical_failure':bool(metric['definition'].get('gate') and metric['score']*factor<100),
                    'scope':'Mathematical sensitivity only. These values are not dataset measurements or experiment outcomes and are not persisted.'})
    return {'baseline': base, 'perturbations': results, 'metric_deterioration_simulations':deterioration,
            'interpretation': 'Dimension importance does not alter within-dimension scores or clear gates. Use metric priority overrides to study within-dimension sensitivity.'}
