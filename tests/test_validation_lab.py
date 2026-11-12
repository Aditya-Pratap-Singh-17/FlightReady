import copy
import io
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from backend.context import DatasetContext, Goal
from sql.store import Store
from backend.workbench.datasets import import_file
from backend.workbench.splits import create_split
from backend.workbench.contracts import SplitRequest, TrainRequest
from backend.workbench.synthetic import detection_fixture
from backend.workbench.worker import run
from backend.workbench.experiments import ExperimentRequest, validate_experiment, _matched_split
from backend.workbench.analytics import predictive_comparison
from backend.workbench.detector import confusion_counts


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from backend.workbench import api, lab_api
    from backend.workbench.quality_api import router as quality_router
    store = Store(tmp_path/'store')
    class Queue:
        def enqueue(self,task,config):
            return store.create('job', {'task':task,'config':config,'status':'queued'})
    monkeypatch.setattr(api,'services',lambda:(store,Queue()))
    monkeypatch.setattr(lab_api,'services',lambda:(store,Queue()))
    app=FastAPI();app.include_router(api.router);app.include_router(quality_router);app.include_router(lab_api.router)
    configurations=[]
    for i in (1,2):
        path=detection_fixture(tmp_path/f'data{i}.zip',seed=i)
        with path.open('rb') as incoming: ds=import_file(store,path.name,incoming)
        split=create_split(store,ds['id'],SplitRequest(group_column='flight_id'))
        context=DatasetContext(original_text='Detect synthetic drones',confirmed=True,goal=Goal(task_type='object_detection',required_classes=['drone']))
        ready=store.create('job',{'task':'readiness','status':'running','config':{'dataset_id':ds['id'],'split_id':split['id'],'context':context.model_dump()}})
        run(store,ready['id'])
        configurations.append(TrainRequest(dataset_id=ds['id'],split_id=split['id'],readiness_id=ready['id']))
    path=detection_fixture(tmp_path/'external.zip',seed=93,external=True)
    with path.open('rb') as incoming: external=import_file(store,path.name,incoming)
    request=ExperimentRequest(name='Controlled fixture',candidate=configurations[0],reference=configurations[1],test_dataset_id=external['id'])
    with TestClient(app) as client:
        yield store,client,request
    store.engine.dispose()


def test_quality_api_records_auditable_versions_and_preserves_source(lab):
    store,client,request=lab
    original=store.get(request.candidate.readiness_id)
    body={'readiness_id':original['id'],'scenario':'nighttime_detection','dimension_weights':{'integrity':2},'overrides_confirmed':True,'persist':True}
    result=client.post('/api/v1/quality/aggregate',json=body)
    assert result.status_code==200,result.text
    data=result.json()
    assert data['aggregation_id'] and not data['preview_only']
    assert store.get(original['id'])==original
    assert store.get(data['aggregation_id'],'aggregation')['payload']['request']['dimension_weights']=={'integrity':2}
    assert len(client.get('/api/v1/quality/history',params={'readiness_id':original['id']}).json())==1
    for format in ('json','csv','html','pdf'):
        exported=client.get(f'/api/v1/reports/{data["aggregation_id"]}?format={format}')
        assert exported.status_code==200
        if format=='html':
            assert 'nighttime_detection' in exported.text and 'Saved Aggregation' in exported.text
    definitions=client.get('/api/v1/quality/registry').json()
    assert 'UAV-20' in definitions['raw_metrics'] and 'row_count' in definitions['raw_metrics']
    assert definitions['raw_metrics']['row_count']['quality_direction'].startswith('descriptive')
    error=client.post('/api/v1/quality/aggregate',json={**body,'dimension_weights':{'integrity':True}})
    assert error.status_code==422
    compare=client.post('/api/v1/quality/compare',json={'readiness_id':original['id'],'contexts':[
        {'readiness_id':original['id']},{'readiness_id':original['id'],'task':'classification'}]})
    assert compare.status_code==200
    assert compare.json()['results'][1]['requires_measurement']


def test_sensitivity_never_changes_original_or_reports_simulations_as_measurements(lab):
    store,client,request=lab
    original=store.get(request.candidate.readiness_id)
    data=client.post('/api/v1/quality/sensitivity',json={'readiness_id':original['id']}).json()
    assert data['perturbations']
    for scenario in data['metric_deterioration_simulations']:
        assert scenario['hypothetical_dimension_score']<=scenario['observed_dimension_score']+1e-9
        assert 'not dataset measurements' in scenario['scope']
    assert store.get(original['id'])==original


def test_experiment_protocol_validation_and_single_start(lab):
    store,client,request=lab
    protocol=validate_experiment(store,request)
    assert protocol['test_images']==4
    assert protocol['requested_training_sizes']==[6,6]
    created=client.post('/api/v1/experiments',json=request.model_dump())
    assert created.status_code==201,created.text
    id=created.json()['id']
    assert client.post(f'/api/v1/experiments/{id}/start').status_code==202
    assert client.post(f'/api/v1/experiments/{id}/start').status_code==422
    assert store.get(id,'experiment')['payload']['status']=='queued'


def test_concurrent_start_requests_create_only_one_parent_job(lab):
    from concurrent.futures import ThreadPoolExecutor
    from backend.workbench.lab_api import start_experiment
    store,client,request=lab
    created=client.post('/api/v1/experiments',json=request.model_dump()).json()
    def start():
        try:return start_experiment(created['id'])['id']
        except ValueError:return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:start(),range(2)))
    assert sum(r is not None for r in results)==1
    assert sum(j['payload']['task']=='experiment' for j in store.list('job'))==1


def test_final_test_overlap_rejected_before_protocol_creation(lab):
    _,client,request=lab
    request.test_dataset_id=request.candidate.dataset_id
    result=client.post('/api/v1/experiments',json=request.model_dump())
    assert result.status_code==422 and 'overlap' in result.text


def test_shared_validation_source_is_checked_against_training_and_final_test(lab):
    from backend.workbench.detector import validate_training,validation_source
    store,_,request=lab
    # Reuse only the reference dataset's held-out validation partition for both
    # models, preserving its training and final-test independence.
    for config in (request.candidate,request.reference):
        config.validation_dataset_id=request.reference.dataset_id
        config.validation_split_id=request.reference.split_id
    result=validate_experiment(store,request)
    assert result['requested_training_sizes']==[6,6]
    classes,_=validate_training(store,request.candidate.model_dump())
    source=validation_source(store,request.candidate.model_dump(),classes)
    assert len(source['names'])==2 and source['manifest']['images']['train']==[]
    request.candidate.validation_dataset_id=request.test_dataset_id
    request.candidate.validation_split_id=None
    with pytest.raises(ValueError,match='overlap'):
        validate_experiment(store,request)
    request.candidate.validation_dataset_id=request.candidate.dataset_id
    with pytest.raises(ValueError,match='overlap'):
        validate_training(store,request.candidate.model_dump())


def test_test_identity_includes_annotations_not_only_identical_image_bytes(lab,tmp_path):
    import zipfile
    from backend.workbench.detector import test_manifest
    store,_,request=lab
    original=store.get(request.test_dataset_id,'dataset')['payload']
    path=tmp_path/'changed-labels.zip'
    with zipfile.ZipFile(original['source_path']) as incoming,zipfile.ZipFile(path,'w') as output:
        for name in incoming.namelist():
            value=b'0 .6 .5 .25 .375\n' if name=='labels/test/0.txt' else incoming.read(name)
            output.writestr(name,value)
    with path.open('rb') as incoming:
        changed=import_file(store,path.name,incoming)
    before,_=test_manifest(store,{'dataset_id':request.test_dataset_id})
    after,_=test_manifest(store,{'dataset_id':changed['id']})
    assert before['image_checksum']==after['image_checksum']
    assert before['test_checksum']!=after['test_checksum']


def test_matched_size_manifests_are_reproducible_and_keep_validation(lab):
    store,_,request=lab
    before=store.get(request.candidate.split_id,'split')['payload']
    a=store.get(_matched_split(store,request.candidate,4,42),'split')['payload']
    b=store.get(_matched_split(store,request.candidate,4,42),'split')['payload']
    assert a==b and len(a['images']['train'])==4
    assert a['images']['val']==before['images']['val'] and a['images']['test']==before['images']['test']
    assert store.get(request.candidate.split_id,'split')['payload']==before
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**request.model_dump(),'modes':['combined'],'size_mode':'matched'})
    with pytest.raises(ValidationError,match='reference dataset'):
        ExperimentRequest.model_validate({**request.model_dump(),'modes':['candidate'],'reference':None,'size_mode':'matched'})


def test_combined_readiness_measures_both_training_sources_and_preserves_image_provenance(lab):
    from backend.workbench.combined import combined_readiness
    from backend.workbench.datasets import load,image_bytes
    store,client,request=lab
    config={**request.candidate.model_dump(),'model_type':'combined','reference_dataset_id':request.reference.dataset_id,
            'reference_split_id':request.reference.split_id,'class_mapping':{}}
    linked=combined_readiness(store,config,['drone'])
    record=store.get(linked['combined_readiness_id'],'job')
    certificate=record['payload']['result']
    assert certificate['dataset']['profile']['image_count']==12
    assert next(m for m in certificate['metrics'] if m['metric_id']=='row_count')['value']==12
    assert certificate['quality_measurements']['clarity']['denominator']==12
    ds=load(store,linked['combined_dataset_id'])
    assert len(ds.objects)==12 and len(ds.image_hashes)==12
    assert len(image_bytes(store,linked['combined_dataset_id'],next(iter(ds.image_hashes))))>0
    assert store.get(linked['combined_dataset_id'])['payload']['archived']
    assert client.get(f'/api/v1/reports/{record["id"]}?format=pdf').status_code==200


def test_incomplete_experiment_reports_stay_incomplete_and_escape_html(lab):
    _,client,request=lab
    request.name='<script>alert(1)</script>'
    created=client.post('/api/v1/experiments',json=request.model_dump()).json()
    response=client.get(f'/api/v1/reports/{created["id"]}?format=html')
    assert response.status_code==200
    assert 'Incomplete' in response.text and '<script>' not in response.text
    csv=client.get(f'/api/v1/reports/{created["id"]}?format=csv')
    assert 'Incomplete' in csv.text
    pdf=client.get(f'/api/v1/reports/{created["id"]}?format=pdf')
    assert pdf.status_code==200 and pdf.content.startswith(b'%PDF')
    from pypdf import PdfReader
    assert 'Incomplete' in ''.join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages)
    for format in ('json','csv','html','pdf'):
        response=client.get(f'/api/v1/reports/{request.candidate.readiness_id}?format={format}')
        assert response.status_code==200
    assert 'DIMENSION_' in client.get(f'/api/v1/reports/{request.candidate.readiness_id}?format=csv').text


def test_external_published_values_are_separate_from_local_evaluations(lab):
    store,client,_=lab
    before=len(store.list('evaluation'))
    body={'title':'Published fixture','source_url':'https://example.org/paper','citation':'Fixture, 2026',
          'architecture':'YOLO11n','training_dataset':'Declared source','evaluation_dataset':'Declared benchmark',
          'protocol':'Different benchmark; user supplied','metrics':{'mAP50_95':.4}}
    result=client.post('/api/v1/benchmarks/external',json=body)
    assert result.status_code==201
    assert result.json()['payload']['comparability']=='unverified'
    assert len(store.list('evaluation'))==before
    assert not client.get('/api/v1/research').json()['rows']


def test_spatial_confusion_counts_expose_class_errors_without_changing_primary_matching():
    truth=[{'class':'drone','bbox':[.5,.5,.2,.2]}]
    predictions=[{'class':'bird','confidence':.9,'bbox':[.5,.5,.2,.2]}]
    matrix=confusion_counts(truth,predictions,['drone','bird'],.5)
    assert matrix[1,0]==1 and matrix.sum()==1


def test_rank_stability_never_confuses_protocol_changes_or_ties_with_domain_effects():
    from backend.workbench.analytics import ranking_stability
    rows=[{'model_id':model,'test_sha256':test,'evaluation_settings':{'confidence':threshold},'metrics':{'mAP50_95':value}}
          for test,threshold,values in [('test-a',.25,[.9,.8]),('test-b',.5,[.2,.6])] for model,value in zip(['a','b'],values)]
    assert ranking_stability(rows,'mAP50_95')['status']=='insufficient_evidence'
    for row in rows:row['evaluation_settings']={'confidence':.25}
    assert ranking_stability(rows,'mAP50_95')['groups'][0]['ranking_changed']
    for row in rows:row['metrics']['mAP50_95']=.5
    assert not ranking_stability(list(reversed(rows)),'mAP50_95')['groups'][0]['ranking_changed']


def test_predictive_analysis_uses_held_out_datasets_and_honest_small_sample_state():
    small=predictive_comparison([{'quality_metrics':{'dimension.integrity':1},'mean':.1}], 'fixture','mAP50_95')
    assert small['status']=='insufficient_evidence' and not small['predictors']
    values=[{'dataset':str(i),'mean':i/10,'quality_metrics':{'dimension.integrity':i*10,'baseline.training_images':100+i}} for i in range(8)]
    result=predictive_comparison(values,'fixture','mAP50_95')
    assert result['status']=='exploratory'
    assert result['independent_datasets']==8
    assert all(p['mae']<1e-12 for p in result['predictors'])


def test_cancelling_an_experiment_child_cancels_its_owning_protocol(tmp_path):
    from backend.workbench.jobs import JobManager
    store=Store(tmp_path/'cancel')
    experiment=store.create('experiment',{'status':'running'})
    parent=store.create('job',{'task':'experiment','status':'running','config':{'experiment_id':experiment['id']}})
    child=store.create('job',{'task':'training','status':'running','parent_job_id':parent['id'],'config':{}})
    manager=JobManager(store)
    manager.cancel(child['id'])
    assert store.get(experiment['id'])['payload']['status']=='cancelled'
    assert store.get(parent['id'])['payload']['status']=='cancelled'
    assert store.get(child['id'])['payload']['status']=='cancelled'
    store.engine.dispose()


def test_scientific_conclusion_pairs_seeds_and_withholds_missing_quality_deltas(lab):
    """Artificial numbers here test interpretation arithmetic, not model performance."""
    from backend.workbench.conclusions import experiment_conclusion
    store,_,request=lab
    evaluations=[]
    for configuration,value in ((request.reference,.25),(request.candidate,.5)):
        cfg={**configuration.model_dump(),'initialization_sha256':'identical-test-fixture',
             'selected_device':'cpu','classes':['drone']}
        model=store.create('model',{'dataset_id':configuration.dataset_id,'readiness_id':configuration.readiness_id,
            'configuration':cfg,'training_images':6})
        evaluation=store.create('evaluation',{'model_id':model['id'],'dataset_id':request.test_dataset_id,'test_sha256':'unit-test-common-heldout',
            'settings':{'ground_truth_protocol_version':'unit-test'},'test_images':4,'metrics':{'mAP50_95':value}})
        evaluations.append(evaluation['id'])
    record=store.create('experiment',{'request':request.model_dump(),'status':'completed','evaluation_ids':evaluations})
    result=experiment_conclusion(store,record)
    pair=result['pairs'][0]
    assert pair['performance_changes']['mAP50_95']['absolute_change']==.25
    assert pair['performance_changes']['mAP50_95']['relative_change']==1
    assert pair['protocol_review']['matched_conditions']
    assert any(d['baseline_score'] is None and d['absolute_change'] is None for d in pair['quality_changes'])
    # A different evaluation protocol cannot support a paired difference.
    store.update(evaluations[-1],test_sha256='different-heldout')
    assert experiment_conclusion(store,record)['status']=='insufficient_comparison_evidence'
