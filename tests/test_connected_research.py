"""Canonical verdicts, safe inference and backwards-compatible analytical evidence."""
import copy
import io
import json
import zipfile
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from backend.context import DatasetContext, Goal
from backend.ingestion import load_dataset
from backend.compatibility import compatibility
from backend.final_report import build_final_report, dimension_bounds
from backend.readiness_engine import assess
from backend.workbench.datasets import import_file, file_sha
from backend.workbench.splits import create_split
from backend.workbench.contracts import SplitRequest
from backend.workbench.synthetic import detection_fixture
from backend.workbench.worker import run
from sql.store import Store


def native_archive(root='VisDrone2019-DET-val/'):
    raw=io.BytesIO(); image=io.BytesIO(); Image.new('RGB',(64,64),(40,50,60)).save(image,format='PNG')
    with zipfile.ZipFile(raw,'w') as out:
        out.writestr(root+'images/1.png',image.getvalue())
        out.writestr(root+'annotations/1.txt','10,10,20,20,1,4,0,0\n')
    return raw.getvalue()


def context(classes=('drone',), night=False):
    return DatasetContext(original_text='Detect targets',confirmed=True,goal=Goal(task_type='object_detection',
        required_classes=list(classes), input_modalities=['image'],target_environment={'lighting':['nighttime']} if night else {}))


def test_visdrone_drone_target_mismatch_and_unknown_night_are_distinct():
    ds=load_dataset('native.zip',native_archive())
    preview=compatibility(ds,context(night=True))
    assert preview['status']=='incompatible' and preview['dataset_role']=='val'
    assert 'drone-object annotations' in preview['summary']
    class_check=next(c for c in preview['checks'] if c['requirement']['metric_id']=='required_class_count')
    scenario=next(c for c in preview['checks'] if c['requirement']['metric_id']=='scenario_count')
    assert class_check['state']=='fail' and class_check['result']['value']==0
    assert scenario['state']=='insufficient_evidence' and scenario['result']['value'] is None
    report=assess(ds,context(night=True)).final_report
    assert report['decision']=='NOT_READY'
    assert report['validation']['status']=='blocked' and not report['validation']['training_allowed']
    assert any(g['status']=='insufficient_evidence' for g in report['hard_requirements'])


def test_night_unknown_is_neither_failure_nor_pass_for_supported_car():
    ds=load_dataset('native.zip',native_archive())
    preview=compatibility(ds,context(('car',),True))
    assert preview['status']=='unverified'
    assert not preview['training_partition_available']
    report=assess(ds,context(('car',),True)).final_report
    assert report['decision']=='INDETERMINATE' and report['criterion_attainment_index']['score'] is None


def test_validation_source_can_be_evaluated_but_never_split_for_training(tmp_path):
    store=Store(tmp_path/'store')
    try:
        record=import_file(store,'native.zip',io.BytesIO(native_archive()))
        assert record['payload']['training']['eligible']  # canonical labels can execute
        with pytest.raises(ValueError,match='validation-only'):
            create_split(store,record['id'],SplitRequest(acknowledge_missing_groups=True))
        from backend.workbench.detector import test_manifest
        manifest,names=test_manifest(store,{'dataset_id':record['id']})
        assert len(names)==1 and manifest['test_checksum']
    finally: store.engine.dispose()


def test_geometric_missing_evidence_bounds_preserve_unknown_weight_and_zero():
    def dimension(a,b=None):
        return {'metrics':[{'effective_weight':.25,'status':'measured','score':a},
                           {'effective_weight':.75,'status':'measured' if b is not None else 'not_assessed','score':b}]}
    assert dimension_bounds(dimension(25))==pytest.approx([0,100*(.25**.25)])
    assert dimension_bounds(dimension(25,100))==pytest.approx([100*(.25**.25)]*2)
    assert dimension_bounds(dimension(0))==[0,0]
    assert dimension_bounds({'metrics':[]}) is None


@pytest.fixture
def research_api(tmp_path,monkeypatch):
    from backend.workbench import api,workflow_api,research_api
    store=Store(tmp_path/'store')
    class Queue:
        def enqueue(self,task,config): return store.create('job',{'task':task,'config':config,'status':'queued','stage':'Waiting','result':None,'error':None})
    for module in (api,workflow_api,research_api): monkeypatch.setattr(module,'services',lambda:(store,Queue()))
    app=FastAPI(); app.include_router(api.router); app.include_router(workflow_api.router); app.include_router(research_api.router)
    original=detection_fixture(tmp_path/'source.zip')
    car=tmp_path/'car.zip'
    with zipfile.ZipFile(original) as source,zipfile.ZipFile(car,'w') as out:
        for name in source.namelist(): out.writestr(name,b'car\n' if name=='classes.txt' else source.read(name))
    with car.open('rb') as source: dataset=import_file(store,'car.zip',source)
    ready=store.create('job',{'task':'readiness','status':'running','config':{'dataset_id':dataset['id'],'context':context(('car',)).model_dump()}})
    run(store,ready['id'])
    with TestClient(app) as client: yield store,client,dataset,store.get(ready['id'])
    store.engine.dispose()


def test_canonical_exports_share_verdict_version_and_source_fingerprint(research_api):
    from pypdf import PdfReader
    store,client,_,job=research_api
    original=copy.deepcopy(job)
    report=job['payload']['result']['final_report']
    assert report['decision']=='INDETERMINATE' and report['evidence_coverage']<1
    assert job['payload']['result']['overall_status']=='READY'  # preserved narrow technical decision
    for format in ('json','html','pdf','csv'):
        exported=client.get(f'/api/v1/reports/{job["id"]}?format={format}')
        assert exported.status_code==200
        if format=='json': assert exported.json()['payload']['result']['final_report']==report
        if format=='html':
            assert report['report_sha256'] in exported.text and report['decision'] in exported.text
            assert 'F. Technical appendix' in exported.text and 'Object-scale distributions' not in exported.text[:1000]
        if format=='pdf':
            text=''.join(p.extract_text() for p in PdfReader(io.BytesIO(exported.content)).pages)
            assert report['report_version'] in text and report['decision'] in text
            assert report['assessment_fingerprint'] in text.replace('\n','')
    assert store.get(job['id'])==original
    final=build_final_report(job['payload']['result'])
    assert final==report


def test_registry_download_requires_explicit_strict_consent(research_api):
    store,client,_,_=research_api
    registry=client.get('/api/v1/model-registry').json()
    assert len(registry[0]['revision'])==40 and 'drone' not in registry[0]['pretrained_classes']
    for body in ({'registry_id':'yolos-tiny-coco'}, {'registry_id':'yolos-tiny-coco','download_consent':'true','license_accepted':True},
                 {'registry_id':'evil/arbitrary','download_consent':True,'license_accepted':True}):
        assert client.post('/api/v1/model-registry/acquire',json=body).status_code==422
    assert not [j for j in store.list('job') if j['payload']['task']=='model_acquisition']


def tiny_model(store,tmp_path):
    torch=pytest.importorskip('torch'); pytest.importorskip('transformers'); pytest.importorskip('safetensors')
    from transformers import YolosConfig,YolosForObjectDetection,YolosImageProcessor
    torch.manual_seed(72); torch.set_num_threads(1)
    cfg=YolosConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,
        image_size=[64,64],patch_size=16,num_detection_tokens=4,use_mid_position_embeddings=False,
        id2label={0:'car'},label2id={'car':0})
    directory=tmp_path/'tiny-safe-model'; directory.mkdir()
    YolosForObjectDetection(cfg).save_pretrained(directory,safe_serialization=True)
    YolosImageProcessor(size={'shortest_edge':64,'longest_edge':128}).save_pretrained(directory)
    return store.create('acquired_model',{'registry_id':'test-fixture','name':'Tiny random software fixture',
        'architecture':'YolosForObjectDetection','adapter':'transformers_yolos','classes':['car'],
        'repository':'none: locally generated random software fixture','revision':'local-random-seed-72',
        'directory':str(directory),'file_sha256':{p.name:file_sha(p) for p in directory.iterdir()},
        'license':'test fixture only','limitations':'Random initialization; no empirical detection capability established.'})


def test_real_safe_inference_metrics_persist_and_link_without_mutating_assessment(research_api,tmp_path):
    pytest.importorskip('pycocotools')
    from backend.workbench.inference import benchmark
    store,client,dataset,job=research_api
    original=copy.deepcopy(job)
    model=tiny_model(store,tmp_path)
    request={'dataset_id':dataset['id'],'readiness_id':job['id'],'acquired_model_id':model['id'],
        'acknowledge_evaluation_scope':True,'device':'cpu','image_size':64}
    queued=client.post('/api/v1/inference-benchmarks',json=request)
    assert queued.status_code==202,queued.text
    run(store,queued.json()['id'])
    finished=store.get(queued.json()['id'])['payload']
    assert finished['status']=='completed',finished.get('error')
    bid=finished['result']['benchmark_id']; record=store.get(bid,'benchmark')['payload']
    assert record['test_images']==10
    assert 0<=record['metrics']['mAP50_95']<=1 and record['metrics']['latency_images']==10
    assert record['reproducibility']['dependencies']['pycocotools']
    predictions=client.get(f'/api/v1/inference-benchmarks/{bid}/predictions').json()
    assert len(predictions[0]['all_predictions'])==4  # actual tiny detector query outputs
    supplement=client.get(f'/api/v1/assessments/{job["id"]}/validation').json()
    assert supplement['status']=='completed' and supplement['inference_benchmarks'][0]['id']==bid
    assert store.get(job['id'])==original
    reopened=Store(store.root)
    assert reopened.get(bid,'benchmark')['payload']['metrics']==record['metrics']; reopened.engine.dispose()
    for format in ('json','html','pdf','csv'):
        response=client.get(f'/api/v1/reports/{bid}?format={format}')
        assert response.status_code==200,response.text[:400]
        if format=='csv': assert 'mAP50_95' in response.text and record['test_sha256'] in response.text


def test_coco_ap_and_operating_errors_against_known_geometry():
    pytest.importorskip('pycocotools')
    from backend.workbench.inference import coco_metrics
    from backend.workbench.detector import detection_errors
    truth=[{'class':'car','bbox':[.5,.5,.2,.2]}]
    perfect=[{'class':'car','bbox':[.5,.5,.2,.2],'confidence':.9}]
    sample={'width':100,'height':100,'ground_truth':truth,'all_predictions':perfect}
    assert coco_metrics([sample],['car'])['mAP50_95']==pytest.approx(1)
    sample['all_predictions']=[]
    assert coco_metrics([sample],['car'])['mAP50_95']==0
    counts,_,missed=detection_errors(truth,perfect+perfect,.5)
    assert counts['true_positive']==1 and counts['false_positive']==1 and counts['false_negative']==0 and not missed


def test_download_failure_is_actionable_and_never_creates_model(research_api,monkeypatch):
    pytest.importorskip('huggingface_hub')
    import huggingface_hub
    store,client,_,_=research_api
    def failure(**kwargs):
        assert kwargs['revision']=='95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de'
        assert kwargs['allow_patterns']==['config.json','preprocessor_config.json','model.safetensors']
        raise OSError('network unavailable')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',failure)
    result=client.post('/api/v1/model-registry/acquire',json={'registry_id':'yolos-tiny-coco','download_consent':True,'license_accepted':True})
    assert result.status_code==202
    run(store,result.json()['id'])
    assert 'Check network access' in store.get(result.json()['id'])['payload']['error']
    assert not store.list('acquired_model')


def test_safe_local_cache_activation_checks_schema_and_artifact_integrity(research_api,tmp_path,monkeypatch):
    pytest.importorskip('huggingface_hub')
    import huggingface_hub
    from backend.workbench import model_registry
    store,client,_,_=research_api
    generated=tiny_model(store,tmp_path)
    directory=generated['payload']['directory']
    spec={**model_registry.CATALOG['yolos-tiny-coco'],'pretrained_classes':['car']}
    monkeypatch.setitem(model_registry.CATALOG,'local-test-fixture',spec)
    def local_snapshot(**kwargs):
        assert kwargs['local_files_only'] is True and kwargs['token'] is False
        assert not any(name.endswith(('.py','.bin','.pt')) for name in kwargs['allow_patterns'])
        return directory
    monkeypatch.setattr(huggingface_hub,'snapshot_download',local_snapshot)
    result=client.post('/api/v1/model-registry/acquire',json={'registry_id':'local-test-fixture','local_only':True,'license_accepted':True})
    assert result.status_code==202,result.text
    run(store,result.json()['id'])
    job=store.get(result.json()['id'])['payload']
    assert job['status']=='completed',job.get('error')
    acquired=store.get(job['result']['acquired_model_id'],'acquired_model')['payload']
    assert acquired['local_only'] and not acquired['download_consent'] and acquired['classes']==['car']
    assert model_registry.verify_available(acquired)
    Path(directory,'config.json').write_text('{}')
    assert not model_registry.verify_available(acquired)


def test_official_train_source_requires_explicit_training_only_manifest_and_external_validation(tmp_path):
    from backend.workbench.detector import validate_training
    from backend.workbench.contracts import TrainRequest
    store=Store(tmp_path/'store')
    try:
        data=import_file(store,'train.zip',io.BytesIO(native_archive('VisDrone2019-DET-train/')))
        split=create_split(store,data['id'],SplitRequest(purpose='training_only',acknowledge_missing_groups=True))
        assert len(split['payload']['images']['train'])==1 and not split['payload']['images']['test'] and not split['payload']['images']['val']
        ready=store.create('job',{'task':'readiness','status':'running','config':{'dataset_id':data['id'],'split_id':split['id'],'context':context(('car',)).model_dump()}})
        run(store,ready['id'])
        cfg=TrainRequest(dataset_id=data['id'],split_id=split['id'],readiness_id=ready['id']).model_dump()
        with pytest.raises(ValueError,match='separate labeled validation dataset'):
            validate_training(store,cfg)
    finally: store.engine.dispose()


def test_import_local_transformers_folder_and_preview_without_assessment(research_api,tmp_path):
    store,client,dataset,_=research_api
    source=tiny_model(store,tmp_path)
    assert client.post('/api/v1/acquired-models/local',json={'path':source['payload']['directory']}).status_code==422
    imported=client.post('/api/v1/acquired-models/local',json={'path':source['payload']['directory'],'license_accepted':True})
    assert imported.status_code==201,imported.text
    acquired=imported.json()
    assert acquired['payload']['directory']!=source['payload']['directory']
    body={'dataset_id':dataset['id'],'acquired_model_id':acquired['id'],'mode':'preview','max_images':2,
          'device':'cpu','image_size':64,'validate_project_goal':False}
    queued=client.post('/api/v1/inference-benchmarks',json=body)
    assert queued.status_code==202,queued.text
    run(store,queued.json()['id']); job=store.get(queued.json()['id'])['payload']
    assert job['status']=='completed',job.get('error')
    result=store.get(job['result']['benchmark_id'],'benchmark')['payload']
    assert result['test_images']==2 and result['kind']=='inference_preview'
    assert not result['evaluation_scope']['scored']
    assert 'mAP50_95' not in result['metrics'] and 'recall' not in result['metrics']
    sample=client.get(f"/api/v1/inference-benchmarks/{job['result']['benchmark_id']}/predictions").json()[0]
    assert sample['scored'] is False


def test_hugging_face_repository_inspection_and_custom_source_acquisition(research_api,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    store,client,_,_=research_api
    model=tiny_model(store,tmp_path);directory=model['payload']['directory']
    calls=[]
    class Hub:
        def __init__(self,**kwargs): assert kwargs['token'] is False
        def model_info(self,repo,**kwargs):
            calls.append((repo,kwargs))
            return SimpleNamespace(sha='b'*40,card_data={'license':'apache-2.0'},siblings=[SimpleNamespace(rfilename=n,size=Path(directory,n).stat().st_size) for n in ('config.json','preprocessor_config.json','model.safetensors')])
    def metadata(repo,name,**kwargs):
        assert name=='config.json' and kwargs['revision']=='b'*40
        return str(Path(directory,name))
    monkeypatch.setattr(huggingface_hub,'HfApi',Hub);monkeypatch.setattr(huggingface_hub,'hf_hub_download',metadata)
    inspected=client.post('/api/v1/model-sources/inspect',json={'repository':'https://huggingface.co/researcher/custom-detector','revision':'experiment-v2'})
    assert inspected.status_code==200,inspected.text
    source=inspected.json();assert source['payload']['revision']=='b'*40 and source['payload']['pretrained_classes']==['car']
    assert calls==[('researcher/custom-detector',{'revision':'experiment-v2','files_metadata':True})]
    def snapshot(**kwargs):
        assert kwargs['repo_id']=='researcher/custom-detector' and kwargs['revision']=='b'*40
        assert not any(n.endswith(('.py','.bin','.pt')) for n in kwargs['allow_patterns'])
        return directory
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    queued=client.post('/api/v1/model-registry/acquire',json={'source_id':source['id'],'download_consent':True,'license_accepted':True})
    assert queued.status_code==202,queued.text
    run(store,queued.json()['id']);job=store.get(queued.json()['id'])['payload']
    assert job['status']=='completed',job.get('error')
    assert store.get(job['result']['acquired_model_id'])['payload']['repository']=='researcher/custom-detector'


def test_selected_annotation_scope_can_be_benchmarked_without_claiming_project_goal(research_api,tmp_path):
    store,client,dataset,_=research_api
    model=tiny_model(store,tmp_path)
    profile=copy.deepcopy(dataset['payload']['profile']);profile['class_counts']['van']=0
    # Old imported records carried a training-only eligibility flag. Reliable
    # canonical labels must remain benchmarkable without rewriting the record.
    store.update(dataset['id'],profile=profile,training={'eligible':False,'reasons':['Training currently supports YOLO and COCO detection annotations.']})
    queued=client.post('/api/v1/inference-benchmarks',json={'dataset_id':dataset['id'],'acquired_model_id':model['id'],
        'mode':'benchmark','evaluation_classes':['car'],'class_mapping':{'car':'car'},'validate_project_goal':False,
        'acknowledge_evaluation_scope':True,'device':'cpu','image_size':64})
    assert queued.status_code==202,queued.text
    run(store,queued.json()['id']);job=store.get(queued.json()['id'])['payload']
    assert job['status']=='completed',job.get('error')
    result=store.get(job['result']['benchmark_id'])['payload']
    assert result['evaluation_scope']=={'scored':True,'classes':['car'],'excluded_classes':['van'],'project_goal_validated':False}
    assert result['metrics']['mAP50_95'] is not None


@pytest.mark.parametrize('architecture',['detr','rt_detr'])
def test_builtin_transformer_detector_adapters_run_real_local_inference(research_api,tmp_path,architecture):
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    from transformers import (DetrConfig,DetrForObjectDetection,DetrImageProcessor,ResNetConfig,
        RTDetrConfig,RTDetrResNetConfig,RTDetrForObjectDetection,RTDetrImageProcessor)
    from backend.workbench.model_registry import import_local
    from backend.workbench.inference import YolosAdapter
    store,_,_,_=research_api;torch.manual_seed(79);torch.set_num_threads(1)
    if architecture=='detr':
        backbone=ResNetConfig(embedding_size=8,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],out_features=['stage4'])
        cfg=DetrConfig(backbone_config=backbone,use_timm_backbone=False,use_pretrained_backbone=False,d_model=32,
            encoder_layers=1,decoder_layers=1,encoder_attention_heads=4,decoder_attention_heads=4,
            encoder_ffn_dim=64,decoder_ffn_dim=64,num_queries=4,id2label={0:'car'},label2id={'car':0})
        model=DetrForObjectDetection(cfg);processor=DetrImageProcessor()
    else:
        backbone=RTDetrResNetConfig(embedding_size=8,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],out_features=['stage2','stage3','stage4'])
        cfg=RTDetrConfig(backbone_config=backbone,encoder_hidden_dim=32,encoder_in_channels=[16,32,64],
            encoder_layers=1,encoder_ffn_dim=64,encoder_attention_heads=4,d_model=32,num_queries=4,
            decoder_in_channels=[32,32,32],decoder_layers=1,decoder_attention_heads=4,decoder_ffn_dim=64,
            num_denoising=0,id2label={0:'car'},label2id={'car':0})
        model=RTDetrForObjectDetection(cfg);processor=RTDetrImageProcessor()
    directory=tmp_path/architecture;model.save_pretrained(directory,safe_serialization=True);processor.save_pretrained(directory)
    imported=import_local(store,str(directory),True)
    adapter=YolosAdapter(imported['payload'],'cpu',64)
    predicted=adapter.predict(Image.new('RGB',(64,64),(10,20,30)))
    assert len(predicted)==4 and all(p['class']=='car' and len(p['bbox'])==4 for p in predicted)
