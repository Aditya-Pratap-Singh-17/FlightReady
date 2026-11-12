"""Scientific invariants for adaptive profiles, coverage and non-compensatory gates."""
import copy
import math
import pytest
from backend.context import DatasetContext, Goal
from backend.ingestion import load_dataset
from backend.readiness_engine import assess
from metrics.quality_profiles import load_profile
from metrics.quality_aggregation import aggregate_quality, normalize, geometric


def cert():
    return {'dataset': {'sha256': 'fixture', 'modality': 'image', 'profile': {}},
        'confirmed_context': DatasetContext(original_text='Detect drones', confirmed=True,
            goal=Goal(task_type='object_detection')).model_dump(),
        'metrics': [{'metric_id': id, 'value': value, 'state': 'pass', 'evidence': {'sample_size': 10}, 'reason': 'Measured fixture'}
            for id, value in [('bbox_invalid_rate', 0.), ('missing_label_rate', 0.), ('unreadable_rate', 0.),
                ('duplicate_rate', 0.), ('split_overlap', 0), ('group_leakage', 0), ('near_split_overlap', 0)]],
        'findings': [], 'quality_measurements': {'clarity': {'threshold': 100, 'meeting': 9, 'denominator': 10}}}


def test_normalizations_and_true_zero():
    assert normalize(.25, 'one_minus_fraction') == 75
    assert normalize(0, 'zero_count') == 100
    assert normalize(2, 'zero_count') == 0
    assert geometric([0, 100], [1, 1000]) == 0
    assert geometric([25, 100], [1, 1]) == pytest.approx(50)
    for value in [True, None, float('nan'), float('inf'), -.1, 1.1]:
        with pytest.raises(ValueError): normalize(value, 'fraction')


def test_normalized_weights_include_unobserved_evidence():
    result = aggregate_quality(cert())
    assert sum(d['effective_weight'] for d in result['dimensions']) == pytest.approx(1)
    for d in result['dimensions']:
        if d['weight']:
            assert sum(m['effective_weight'] for m in d['metrics']) == pytest.approx(1)
    annotation = next(d for d in result['dimensions'] if d['id'] == 'annotation_quality')
    assert 0 < annotation['evidence_coverage'] < 1
    assert annotation['score'] is None
    audit = next(m for m in annotation['metrics'] if m['id'] == 'annotation_audit')
    assert audit['score'] is None and audit['effective_weight'] > 0


def test_critical_failure_survives_perfect_other_dimensions_and_zero_weight():
    source = cert()
    source['metrics'][0]['value'] = .75
    result = aggregate_quality(source, metric_weights={'annotation_validity': 0}, overrides_confirmed=True)
    assert result['readiness_status'] == 'blocked'
    assert any(g['id'] == 'annotation_validity' and g['status'] == 'fail' for g in result['critical_requirements'])
    integrity = next(d for d in result['dimensions'] if d['id'] == 'integrity')
    assert integrity['score'] == 100


def test_poor_audit_is_flagged_even_when_geometric_mean_exceeds_warning_threshold():
    source=cert()
    source['metrics'].append({'metric_id':'UAV-15','value':{'independent_audit':{'reviewed_objects':100,'incorrect_objects':75,'error_fraction':.75}},
                             'state':'warning','assessment_status':'measured','reason':'Declared audit','evidence':{'sample_size':100}})
    result=aggregate_quality(source)
    dimension=next(d for d in result['dimensions'] if d['id']=='annotation_quality')
    assert dimension['score']>70  # the geometric mean alone does not solve masking
    assert dimension['status']=='warning'
    assert next(m for m in dimension['metrics'] if m['id']=='annotation_audit')['quality_status']=='warning'


def test_task_applicability_scenario_policy_and_missing_new_measurements():
    source = cert()
    default = aggregate_quality(source)
    night = aggregate_quality(source, scenario='nighttime_detection')
    def importance(result, id): return next(d['effective_weight'] for d in result['dimensions'] if d['id'] == id)
    assert importance(night, 'signal_quality') > importance(default, 'signal_quality')
    classification = aggregate_quality(source, task='classification')
    bbox = next(m for m in classification['individual_metrics'] if m['id'] == 'annotation_validity')
    assert bbox['status'] == 'not_applicable'
    assert classification['requires_measurement'] and classification['readiness_status'] != 'ready'
    assert not any(m['metric_id'] == 'missing_label_rate' for m in classification['raw_metrics'])
    tabular = copy.deepcopy(source); tabular['dataset']['modality'] = 'tabular'
    telemetry = aggregate_quality(tabular, task='telemetry')
    assert next(m for m in telemetry['individual_metrics'] if m['id'] == 'clarity')['status'] == 'not_applicable'
    wrong_modality=aggregate_quality(source,task='telemetry')
    assert next(m for m in wrong_modality['individual_metrics'] if m['id']=='clarity')['status']=='not_applicable'
    assert wrong_modality['readiness_status']=='blocked'
    with pytest.raises(ValueError): aggregate_quality(source, task='classification', scenario='nighttime_detection')


def test_frozen_profiles_reproduce_results_and_do_not_mutate_sources():
    source = cert(); original = copy.deepcopy(source)
    result = aggregate_quality(source, metric_weights={'uniqueness': 2}, overrides_confirmed=True)
    assert source == original
    repeated = aggregate_quality(source, metric_weights={'uniqueness': 2}, overrides_confirmed=True, profile=result['profile'])
    assert result == repeated
    assert result['calculation_sha256'] == repeated['calculation_sha256']


@pytest.mark.parametrize('weights', [{'unknown': 1}, {'clarity': True}, {'clarity': math.nan}, {'clarity': -1}])
def test_invalid_overrides_fail_clearly(weights):
    with pytest.raises(ValueError): aggregate_quality(cert(), metric_weights=weights, overrides_confirmed=True)


def test_confirmation_and_nonzero_dimension_importance_required():
    with pytest.raises(ValueError, match='confirmation'): aggregate_quality(cert(), metric_weights={'clarity': 2})
    with pytest.raises(ValueError):
        aggregate_quality(cert(), dimension_weights={d: 0 for d in load_profile().dimensions}, overrides_confirmed=True)


def test_existing_assessment_keeps_all_raw_metrics_and_adds_quality_evidence():
    ds = load_dataset('fixture.csv', b'x,label\n1,drone\n2,bird\n')
    context = DatasetContext(original_text='Classify', confirmed=True, goal=Goal(task_type='classification', prediction_target='label'))
    certificate = assess(ds, context)
    assert certificate.quality_profile is not None
    assert len(certificate.metrics) >= 20
    assert certificate.quality_profile['profile_version'] == 'adaptive-1.0'
    assert certificate.overall_status in ('READY', 'CONDITIONALLY_READY')


def test_histograms_are_actual_image_measurements_not_dimension_scores():
    from types import SimpleNamespace
    import pandas as pd
    from metrics.quality_aggregation import collect_measurements
    ds = SimpleNamespace(sha256='fixture', profile={}, modality='image', disk_path=None,
                         frame=pd.DataFrame({'readable': [True]*4, 'laplacian_variance': [0, 20, 100, 200]}))
    evidence = collect_measurements(ds)
    assert evidence['clarity']['meeting'] == 2
    assert sum(b['count'] for b in evidence['distributions']['laplacian_variance']['bins']) == 4
