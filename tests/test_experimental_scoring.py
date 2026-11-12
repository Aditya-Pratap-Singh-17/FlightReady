"""Numerical invariants for the removable, explicitly uncalibrated score."""
import copy
import io
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from pydantic import ValidationError
from backend.context import Goal, DatasetContext, AssessmentEvidence, MetricResult, Requirement, AssessmentFinding
from metrics.experimental_scoring import Profile, calculate, aggregate, load_profile
from backend.ingestion import load_dataset
from backend.readiness_engine import assess
from backend.reports import html_report


def dataset():
    return SimpleNamespace(modality='image',disk_path=None,sha256='fixture',profile={},
        frame=pd.DataFrame({'readable':[True,True], 'laplacian_variance':[0.,200.]}))


def metric(id,value,state='pass',**kwargs):
    return MetricResult(metric_id=id,dimension='integrity',state=state,value=value,reason='fixture',
        evidence=AssessmentEvidence(method='fixture'),**kwargs)


def profile(*components):
    return Profile(version='test',components=[{'id':id,'label':id,'weight':weight,'source':'metric',
        'metric_id':id,'normalization':'fraction'} for id,weight in components])


def test_weighted_mean_contributions_and_scale_invariance():
    value=calculate(dataset(),Goal(),[metric('a',1.),metric('b',0.)],[],profile(('a',3),('b',1)))
    assert value['score_percent']==75
    assert value['status']=='complete' and value['evidence_coverage_percent']==100
    assert value['score_range_percent']==[75,75]
    assert sum(c['contribution_points'] for c in value['components'])==75
    assert aggregate(value,{'a':30,'b':10})['score_percent']==75
    changed=aggregate(value,{'a':1,'b':3})
    assert changed['score_percent']==25
    assert changed['profile_sha256']!=value['profile_sha256']
    assert value['components'][0]['weight']==3
    assert changed['components'][0]['definition']['weight']==1


def test_unknown_evidence_is_not_zero_and_bounds_expose_it():
    value=calculate(dataset(),Goal(),[metric('a',1.),metric('b',None,'insufficient_evidence')],[],profile(('a',3),('b',1)))
    assert value['score_percent']==100 and value['status']=='partial'
    assert value['evidence_coverage_percent']==75
    assert value['score_range_percent']==[75,100]
    assert value['components'][1]['score_percent'] is None
    missing=calculate(dataset(),Goal(),[],[],profile(('a',3),('b',1)))
    assert missing['score_percent'] is None and missing['status']=='unavailable'
    assert missing['score_range_percent']==[0,100]


def test_not_applicable_and_zero_weights_are_excluded():
    value=calculate(dataset(),Goal(),[metric('a',.5),metric('b',None,'not_applicable')],[],profile(('a',3),('b',1)))
    assert value['score_percent']==50 and value['evidence_coverage_percent']==100
    zero=aggregate(value,{'a':0,'b':0})
    assert zero['score_percent'] is None and zero['score_range_percent'] is None
    assert all(c['contribution_points'] is None for c in zero['components'])


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),True,'4',1000001])
def test_invalid_weights_rejected_without_changing_evidence(value):
    original=calculate(dataset(),Goal(),[metric('a',1.)],[],profile(('a',1)))
    with pytest.raises(ValueError): aggregate(original,{'a':value})
    with pytest.raises(ValueError): aggregate(original,{'unknown':1})
    assert original['score_percent']==100


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-.1,1.1])
def test_invalid_fraction_evidence_never_becomes_a_score(value):
    # Metric contracts reject nonfinite values; represent them with a test double
    # to exercise the scoring boundary independently as well.
    item=SimpleNamespace(metric_id='a',value=value,state='pass',assessment_status=None,reason='fixture')
    result=calculate(dataset(),Goal(),[item],[],profile(('a',1)))
    assert result['score_percent'] is None


def test_clarity_is_fraction_of_images_not_mean_sharpness():
    p=Profile(version='test',components=[{'id':'clarity','label':'clarity','weight':1,
        'source':'image_clarity','threshold':100}])
    value=calculate(dataset(),Goal(),[],[],p)
    assert value['score_percent']==50
    assert value['components'][0]['raw_value']['readable_images']==2
    ds=dataset();ds.disk_path='preview.duckdb'
    assert calculate(ds,Goal(),[],[],p)['score_percent'] is None


def test_custom_normalization_has_explicit_direction_and_anchors():
    p=Profile(version='test',components=[{'id':'distance','label':'distance','weight':1,
        'source':'metric','metric_id':'distance','normalization':'linear','good':0,'bad':10}])
    assert calculate(dataset(),Goal(),[metric('distance',2)],[],p)['score_percent']==80
    assert calculate(dataset(),Goal(),[metric('distance',20)],[],p)['score_percent']==0
    with pytest.raises(ValidationError):
        Profile(version='bad',components=[{**p.components[0].model_dump(),'good':10}])
    estimated=metric('distance',2,assessment_status='estimated')
    assert calculate(dataset(),Goal(),[estimated],[],p)['score_percent'] is None


def test_overlap_checks_exclude_non_applicable_but_keep_unknown_evidence():
    p=Profile(version='test',components=[{'id':'overlap','label':'overlap','weight':1,
        'source':'split_overlap_checks','metric_ids':['split_overlap','near_split_overlap']}])
    value=calculate(dataset(),Goal(),[metric('split_overlap',0),metric('near_split_overlap',None,'not_applicable')],[],p)
    assert value['score_percent']==100 and value['status']=='complete'
    missing=calculate(dataset(),Goal(),[metric('split_overlap',0),metric('near_split_overlap',None,'insufficient_evidence')],[],p)
    assert missing['score_percent'] is None


def test_class_coverage_missing_classes_and_unconfigured_scenarios():
    findings=[]
    for name,state in [('car','pass'),('drone','fail')]:
        findings.append(AssessmentFinding(requirement=Requirement(requirement_id=name,description=name,
            metric_id='required_class_count'),result=metric('required_class_count',0),state=state,reason='fixture'))
    p=Profile(version='test',components=[{'id':'classes','label':'classes','weight':3,'source':'required_class_coverage'},
        {'id':'scenarios','label':'scenarios','weight':1,'source':'scenario_coverage'}])
    result=calculate(dataset(),Goal(task_type='object_detection'),[],findings,p)
    assert result['score_percent']==50 and result['evidence_coverage_percent']==75
    assert result['hard_failure_ids']==['drone']


def test_disabled_bad_config_and_high_score_never_override_not_ready(monkeypatch,tmp_path):
    ds=load_dataset('fixture.csv',b'x,label\n1,car\n2,car\n')
    context=DatasetContext(original_text='classification fixture',confirmed=True,
        goal=Goal(task_type='classification',prediction_target='label',required_classes=['missing']))
    # Only score duplicates so the summary can be perfect despite a hard failure.
    config=tmp_path/'score.json'
    config.write_text(profile(('duplicate_rate',1)).model_dump_json())
    monkeypatch.setenv('EXPERIMENTAL_DATASET_SCORE_CONFIG',str(config))
    scored=assess(ds,context)
    assert scored.overall_status=='NOT_READY'
    assert scored.experimental_score['score_percent']==0  # fraction normalization of duplicate rate
    config.write_text(json.dumps({'version':'test','components':[{'id':'unique','label':'unique','weight':1,
        'source':'metric','metric_id':'duplicate_rate','normalization':'one_minus_fraction'}]}))
    scored=assess(ds,context)
    assert scored.experimental_score['score_percent']==100 and scored.overall_status=='NOT_READY'
    assert scored.experimental_score['hard_failure_ids']
    assert 'Experimental weighted quality score' in html_report(scored)
    monkeypatch.setenv('EXPERIMENTAL_DATASET_SCORE_ENABLED','false')
    disabled=assess(ds,context)
    assert disabled.experimental_score['status']=='disabled' and disabled.overall_status=='NOT_READY'
    monkeypatch.setenv('EXPERIMENTAL_DATASET_SCORE_ENABLED','true')
    config.write_text('{broken')
    broken=assess(ds,context)
    assert broken.experimental_score['status']=='configuration_error'
    assert broken.overall_status=='NOT_READY'


def test_historical_score_preview_uses_training_subset_and_preserves_report(tmp_path,monkeypatch):
    from backend.workbench import api
    from sql.store import Store
    from backend.workbench.datasets import import_file
    from backend.workbench.synthetic import detection_fixture
    from backend.workbench.splits import create_split
    from backend.workbench.contracts import SplitRequest,ScorePreviewRequest
    from backend.workbench.worker import run
    store=Store(tmp_path/'store')
    monkeypatch.setattr(api,'services',lambda:(store,None))
    path=detection_fixture(tmp_path/'fixture.zip')
    with path.open('rb') as incoming: ds=import_file(store,path.name,incoming)
    split=create_split(store,ds['id'],SplitRequest(group_column='flight_id'))
    context=DatasetContext(original_text='Detect drone',confirmed=True,
        goal=Goal(task_type='object_detection',required_classes=['drone']))
    job=store.create('job',{'task':'readiness','status':'running','config':{
        'dataset_id':ds['id'],'split_id':split['id'],'context':context.model_dump()}})
    run(store,job['id'])
    payload=store.get(job['id'])['payload']
    assert payload['status']=='completed'
    saved=copy.deepcopy(payload)
    result=api.score_preview(job['id'],ScorePreviewRequest(weights={'image_clarity':0}))
    assert result['preview_only'] is True
    assert store.get(job['id'])['payload']==saved
    with pytest.raises(ValueError): api.score_preview(job['id'],ScorePreviewRequest(weights={'readability':True}))
    payload['result'].pop('experimental_score')
    store.update(job['id'],result=payload['result'])
    historical=api.score_preview(job['id'],ScorePreviewRequest())
    clarity=next(c for c in historical['components'] if c['id']=='image_clarity')
    assert clarity['raw_value']['population_images']==6
    assert 'experimental_score' not in store.get(job['id'])['payload']['result']
    exported=api.report(job['id'],format='json')
    assert exported.status_code==200


def test_analytics_never_pools_partial_scores_with_different_evidence(tmp_path):
    from sql.store import Store
    from backend.workbench.analytics import research
    store=Store(tmp_path/'analytics')
    readiness_ids=[]
    for index in range(3):
        ds=store.create('dataset',{'name':str(index),'version':str(index),'sha256':str(index)})
        score={'status':'partial','score_percent':20+20*index,'evidence_coverage_percent':50,
            'profile_sha256':'same-profile','components':[{'id':'a','weight':1,'score_percent':20+20*index,'status':'measured'},
                {'id':'b','weight':1,'score_percent':None,'status':'insufficient_evidence'}]}
        ready=store.create('job',{'config':{'context':{'goal':'same'}},
            'result':{'overall_status':'READY','metrics':[],'experimental_score':score}})
        readiness_ids.append(ready['id'])
        model=store.create('model',{'dataset_id':ds['id'],'readiness_id':ready['id'],'training_images':10,
            'configuration':{'seed':42,'architecture':'fixture','model_type':'candidate'},'validation':{}})
        store.create('evaluation',{'model_id':model['id'],'dataset_id':ds['id'],'test_sha256':'same-test',
            'settings':{},'metrics':{'mAP50_95':.2+.2*index}})
    comparable=research(store,metric='experimental_weighted_score')
    assert comparable['correlations'][0]['independent_datasets']==3
    assert comparable['correlations'][0]['spearman']==pytest.approx(1)
    changed=store.get(readiness_ids[-1])['payload']['result']
    components=changed['experimental_score']['components']
    components[0]['score_percent']=None;components[0]['status']='insufficient_evidence'
    components[1]['score_percent']=60;components[1]['status']='measured'
    store.update(readiness_ids[-1],result=changed)
    split=research(store,metric='experimental_weighted_score')
    assert all(c['independent_datasets']<3 and c['spearman'] is None for c in split['correlations'])
