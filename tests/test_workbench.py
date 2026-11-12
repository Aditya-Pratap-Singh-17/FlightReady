"""Persistence, parser integration, leakage, calculations, and real worker lifecycle."""
import io
import json
import time
import zipfile
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.context import DatasetContext, Goal
from sql.store import Store
from backend.workbench.datasets import import_file, load, register_directory, image_bytes, subset
from backend.workbench.splits import create_split, get_split, assert_independent, signatures
from backend.workbench.contracts import SplitRequest, VariantRequest
from backend.workbench.detector import detection_errors, class_mapping, validate_training, classes_for
from backend.workbench.jobs import JobManager
from backend.workbench.worker import run
from backend.workbench.synthetic import detection_fixture
from backend.workbench.analytics import research, comparison
from backend.workbench.variations import variation


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'workbench')
    yield value
    value.engine.dispose()


@pytest.fixture
def dataset(store,tmp_path):
    path=detection_fixture(tmp_path/'source.zip')
    with path.open('rb') as incoming:
        return import_file(store,'source.zip',incoming)


def confirmed():
    return DatasetContext(original_text='Detect drone objects in synthetic test images',confirmed=True,
        goal=Goal(task_type='object_detection',required_classes=['drone'],input_modalities=['image']))


def test_import_persistence_parser_preview(store,dataset):
    ds=load(store,dataset['id'])
    assert len(ds.frame)==10 and len(ds.objects)==10
    assert dataset['payload']['training']['eligible']
    assert len(image_bytes(store,dataset['id'],'images/train/0.png'))>0
    reopened=Store(store.root)
    assert reopened.get(dataset['id'])==dataset
    assert len(load(reopened,dataset['id']).frame)==10
    reopened.engine.dispose()
    with pytest.raises(ValueError,match='manifest'):
        image_bytes(store,dataset['id'],'../../secret')


def test_local_directory_no_copy_and_source_mutation(store,tmp_path):
    directory=tmp_path/'local';directory.mkdir()
    with zipfile.ZipFile(detection_fixture(tmp_path/'input.zip')) as archive:
        for name in archive.namelist():
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
    record=register_directory(store,directory)
    assert record['payload']['source_kind']=='directory'
    assert not list(Path(record['payload']['directory']).glob('*.png'))
    target=directory/'images/train/0.png';target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        image_bytes(store,record['id'],'images/train/0.png')


@pytest.mark.parametrize('name',['../outside.txt','/absolute.txt','C:/bad.txt'])
def test_archive_paths_rejected(store,name):
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w') as archive:
        archive.writestr(name,'bad')
    raw.seek(0)
    with pytest.raises(ValueError):
        import_file(store,'bad.zip',raw)
    assert not store.list('dataset')


def test_deterministic_split_and_train_only_metrics(store,dataset):
    config=SplitRequest(group_column='flight_id')
    a=create_split(store,dataset['id'],config)
    b=create_split(store,dataset['id'],config)
    assert a['payload']['checksum']==b['payload']['checksum']
    assert {k:len(v) for k,v in a['payload']['images'].items()}=={'train':6,'val':2,'test':2}
    ds=subset(load(store,dataset['id']),a['payload']['images']['train'])
    assert len(ds.objects)==6 and ds.profile['image_count']==6
    job=store.create('job',{'task':'readiness','status':'running','config':{'dataset_id':dataset['id'],'split_id':a['id'],'context':confirmed().model_dump()}})
    run(store,job['id'])
    payload=store.get(job['id'])['payload']
    assert payload['status']=='completed',payload.get('error')
    assert payload['result']['dataset']['profile']['image_count']==6
    assert len(payload['result']['metrics'])>20


def test_overlap_and_external_independence(store,dataset,tmp_path):
    split=create_split(store,dataset['id'],SplitRequest(group_column='flight_id'))['payload']
    assert_independent([split],split,split['images']['test'])
    with pytest.raises(ValueError,match='SHA256'):
        assert_independent([split],split,split['images']['train'])
    source=detection_fixture(tmp_path/'external.zip',seed=23,external=True)
    with source.open('rb') as incoming:
        test=import_file(store,'external.zip',incoming)
    external=load(store,test['id'])
    manifest={'hashes':external.image_hashes,'perceptual':signatures(store,test['id']),'groups':{},'group_column':None,'config':{'near_distance':4}}
    with pytest.raises(ValueError,match='acknowledge'):
        assert_independent([split],manifest,list(external.image_hashes))
    assert_independent([split],manifest,list(external.image_hashes),True)


def test_declared_duplicate_overlap_rejected(store,tmp_path):
    source=detection_fixture(tmp_path/'parent.zip')
    raw=io.BytesIO()
    with zipfile.ZipFile(source) as archive,zipfile.ZipFile(raw,'w') as output:
        for name in archive.namelist():
            output.writestr(name,archive.read('images/train/0.png') if name=='images/test/8.png' else archive.read(name))
    raw.seek(0);record=import_file(store,'overlap.zip',raw)
    with pytest.raises(ValueError,match='overlap'):
        create_split(store,record['id'],SplitRequest(group_column='flight_id'))


def test_class_mapping_and_detection_error_calculations():
    assert class_mapping(['drone'],['drone'],{})=={'drone':'drone'}
    with pytest.raises(ValueError,match='mapping'):
        class_mapping(['uav'],['drone'],{})
    assert class_mapping(['uav'],['drone'],{'uav':'drone'})=={'uav':'drone'}
    truth=[{'class':'drone','bbox':[.5,.5,.2,.2]}]
    prediction={'class':'drone','bbox':[.5,.5,.2,.2],'confidence':.9}
    metrics,boxes,missed=detection_errors(truth,[prediction,prediction],.5)
    assert metrics['true_positive']==1 and metrics['false_positive']==1 and metrics['false_negative']==0
    assert metrics['precision']==.5 and metrics['recall']==1 and metrics['f1']==pytest.approx(2/3)
    assert len(boxes)==2 and not missed
    metrics,_,_=detection_errors([],[],.5)
    assert metrics['f1'] is None and metrics['recall'] is None


@pytest.mark.parametrize('intervention',['blur','resolution','size','imbalance','label_omission','box_perturbation'])
def test_controlled_variations_preserve_test(store,dataset,intervention):
    split=create_split(store,dataset['id'],SplitRequest(group_column='flight_id'))
    request=VariantRequest(dataset_id=dataset['id'],split_id=split['id'],seed=7,intervention=intervention,severity=.5,classes=['drone'])
    changed=variation(store,request)
    assert changed['payload']['provenance']['artificial']
    manifest=changed['payload']['provenance']['manifest']
    import hashlib
    with zipfile.ZipFile(changed['payload']['source_path']) as archive:
        for name,expected in manifest['evaluation_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected
    assert store.get(dataset['id'])['payload']['sha256']==dataset['payload']['sha256']


def test_real_subprocess_queue_and_restart(store,dataset):
    manager=JobManager(store)
    job=manager.enqueue('readiness',{'dataset_id':dataset['id'],'context':confirmed().model_dump(),'split_id':None})
    deadline=time.monotonic()+90
    try:
        while time.monotonic()<deadline:
            result=store.get(job['id'])['payload']
            if result['status'] not in ('queued','running'):
                break
            time.sleep(.1)
        assert result['status']=='completed',result
    finally:
        manager.close()
    queued=store.create('job',{'task':'training','status':'queued','config':{}})
    assert manager.cancel(queued['id'])['payload']['status']=='cancelled'
    abandoned=store.create('job',{'task':'training','status':'running','config':{}})
    manager.recover();manager.close()
    assert store.get(abandoned['id'])['payload']['status']=='interrupted'


def test_api_context_errors_exports_and_archive(store,dataset,monkeypatch):
    import backend.workbench.api as module
    manager=JobManager(store)
    monkeypatch.setattr(module,'services',lambda:(store,manager))
    app=FastAPI();app.include_router(module.router)
    client=TestClient(app)
    assert client.get('/api/v1/datasets').status_code==200
    assert client.get('/api/v1/datasets/missing').status_code==404
    assert client.post('/api/v1/context/validate',json={}).status_code==422
    assert client.post('/api/v1/context/validate',json=confirmed().model_dump()).json()['confirmed'] is False
    assert client.post('/api/v1/context/interpret',json={'text':'Detect drone objects','tags':{'Task':['Detect objects']}}).json()['context']['provider']=='deterministic'
    bad=client.post('/api/v1/readiness',json={'dataset_id':dataset['id'],'context':{**confirmed().model_dump(),'confirmed':False}})
    assert bad.status_code==422
    assert client.get('/api/v1/research').json()['rows']==[]
    assert client.get('/api/v1/research/export?format=json').status_code==200
    assert client.patch('/api/v1/datasets/'+dataset['id'],json={'name':'renamed'}).json()['payload']['name']=='renamed'
    assert client.delete('/api/v1/datasets/'+dataset['id']).status_code==200
    assert client.get('/api/v1/datasets').json()==[]
    assert store.get(dataset['id'])['payload']['archived']


def test_analytics_seed_independence_and_comparable_groups(store):
    # Construct statistical fixture explicitly: this tests aggregation, not detector performance.
    for index,quality in enumerate([.2,.5,.8]):
        dataset=store.create('dataset',{'name':f'statistical fixture {index}','sha256':str(index),'version':str(index),'provenance':{}})
        for seed in [1,2]:
            readiness=store.create('job',{'config':{'context':{'goal':'same'}},'result':{'overall_status':'READY','metrics':[{'state':'pass','metric_id':'quality','value':quality}]}})
            model=store.create('model',{'dataset_id':dataset['id'],'readiness_id':readiness['id'],'training_images':20,
                'configuration':{'seed':seed,'architecture':'fixture','model_type':'candidate'},'validation':{'mAP50_95':quality+.1}})
            store.create('evaluation',{'model_id':model['id'],'dataset_id':dataset['id'],'test_sha256':'same-test',
                'settings':{},'metrics':{'mAP50_95':quality+seed*.01}})
    result=research(store,metric='quality')
    assert len(result['rows'])==6 and len(result['aggregates'])==3
    assert result['correlations'][0]['independent_datasets']==3
    assert result['correlations'][0]['pearson']==pytest.approx(1)
    assert all(g['independent_seeds']==2 for g in result['aggregates'])


def test_coco_import_declared_classes_and_split_metadata(store,tmp_path):
    parent=detection_fixture(tmp_path/'yolo.zip')
    raw=io.BytesIO();images=[];annotations=[];metadata=['image,split,flight_id']
    with zipfile.ZipFile(parent) as source,zipfile.ZipFile(raw,'w') as output:
        for index,name in enumerate(n for n in source.namelist() if n.endswith('.png')):
            output.writestr(name,source.read(name))
            images.append({'id':index,'file_name':name,'width':64,'height':64})
            annotations.append({'id':index,'image_id':index,'category_id':9,'bbox':[16,20,32,24],'iscrowd':0,'area':768})
            metadata.append(f'{name},{name.split("/")[1]},flight-{index}')
        output.writestr('annotations.json',json.dumps({'images':images,'annotations':annotations,'categories':[{'id':9,'name':'drone'},{'id':12,'name':'bird'}]}))
        output.writestr('metadata.csv','\n'.join(metadata))
    raw.seek(0);record=import_file(store,'coco.zip',raw)
    assert record['payload']['training']['eligible']
    assert classes_for(store,record['id'])==['drone','bird']
    manifest=create_split(store,record['id'],SplitRequest(group_column='flight_id'))
    assert len(manifest['payload']['images']['train'])==6


def test_worker_oom_error_is_actionable(store,monkeypatch):
    import backend.workbench.detector as detector
    def fail(*args):raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(detector,'train',fail)
    job=store.create('job',{'task':'training','status':'running','config':{}})
    run(store,job['id'])
    payload=store.get(job['id'])['payload']
    assert payload['status']=='failed' and 'Reduce batch size' in payload['error']


def test_unlabeled_single_image_stays_readiness_only(store,tmp_path):
    from PIL import Image
    raw=io.BytesIO();Image.new('RGB',(16,16),(30,40,50)).save(raw,format='PNG');raw.seek(0)
    record=import_file(store,'one.png',raw)
    assert not record['payload']['training']['eligible']
    assert image_bytes(store,record['id'],'one.png')


def test_training_requires_exact_training_report(store,dataset):
    from backend.workbench.contracts import TrainRequest
    split=create_split(store,dataset['id'],SplitRequest(group_column='flight_id'))
    readiness=store.create('job',{'task':'readiness','status':'completed','config':{'dataset_id':dataset['id'],'split_id':None}})
    config=TrainRequest(dataset_id=dataset['id'],split_id=split['id'],readiness_id=readiness['id']).model_dump()
    with pytest.raises(ValueError,match='exact dataset and training split'):
        validate_training(store,config)


def test_combined_cross_source_validation_leakage(store,dataset):
    from backend.workbench.splits import assert_development_disjoint
    import copy
    a=create_split(store,dataset['id'],SplitRequest(group_column='flight_id'))['payload']
    b=copy.deepcopy(a)
    b['images']['val']=[a['images']['train'][0]]
    with pytest.raises(ValueError,match='training with validation'):
        assert_development_disjoint(a,b)
