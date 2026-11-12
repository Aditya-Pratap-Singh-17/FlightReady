"""Known synthetic evidence; no synthetic result validates real-world usefulness."""
import io
import json
import zipfile
import sys
import types
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.context import Goal,DatasetContext,Requirement,MetricResult
from backend.ingestion import load_dataset,Dataset,IngestionError
from metrics.core import REGISTRY,CALCULATIONS,run_metric
from backend.readiness_engine import assess,evaluate
from metrics.uav_metrics.assessment import summary,bins
from metrics.uav_metrics.learning import learning_views
from metrics.uav_metrics.registry import PROPERTY_IDS
from metrics.uav_metrics.vision import vision_session
from backend.reports import html_report

@pytest.fixture
def image_dataset():
    owned=[]
    def make(metadata='',labels=None,sizes=None,colors=None,extra=None):
        labels=labels if labels is not None else ['0 0.5 0.5 0.5 0.4\n','1 0.5 0.5 1 1\n','',None]
        sizes=sizes or [(20,10)]*len(labels);colors=colors or [(i*40,)*3 for i in range(len(labels))]
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:
            archive.writestr('classes.txt','car\npedestrian\n')
            for i,(label,size,color) in enumerate(zip(labels,sizes,colors)):
                pixels=io.BytesIO();Image.new('RGB',size,color=color).save(pixels,format='PNG')
                archive.writestr(f'images/{i}.png',pixels.getvalue())
                if label is not None:archive.writestr(f'labels/{i}.txt',label)
            if metadata:archive.writestr('metadata.csv',metadata)
        ds=load_dataset('synthetic.zip',output.getvalue(),extra);owned.append(ds);return ds
    yield make
    for ds in owned:ds.close()

def goal(**kwargs):
    return Goal(task_type='object_detection',required_classes=['car','pedestrian'],**kwargs)

def test_registry_has_all_properties_and_contracts(image_dataset):
    ds=image_dataset();g=goal()
    assert len(PROPERTY_IDS)==20 and len(set(PROPERTY_IDS))==20
    for mid in PROPERTY_IDS:
        d=REGISTRY[mid]
        assert d.name and d.description and d.calculation_method and d.limitations and d.required_dataset_fields
        m=run_metric(mid,ds,g)
        assert m.assessment_status and m.assessment_status!='error',(mid,m.reason)
        assert MetricResult.model_validate_json(m.model_dump_json())==m
        assert m.state!='pass'
        assert m.provenance['dataset_sha256']==ds.sha256

def test_luminance_resolution_and_clarity(image_dataset):
    ds=image_dataset(labels=['',''],sizes=[(20,10),(40,20)],colors=[(0,0,0),(100,100,100)])
    lighting=run_metric('UAV-01',ds,goal())
    assert lighting.value['luminance']['mean']==50
    assert lighting.value['contrast']['mean']==0
    clarity=run_metric('UAV-14',ds,goal())
    assert clarity.value['width']['minimum']==20 and clarity.value['width']['maximum']==40
    assert clarity.value['laplacian_variance']['mean']==0
    assert clarity.value['encoded_bytes_per_pixel']['unit']=='bytes/pixel'
    clutter=run_metric('UAV-12',ds,goal())
    assert clutter.value['luminance_entropy']['mean']==0
    assert clutter.value['semantic_clutter_status']=='unverified'

@pytest.mark.parametrize('mid,key', [('UAV-02','weather'),('UAV-05','scene'),('UAV-06','viewpoint')])
def test_metadata_categories_and_missing_denominator(image_dataset,mid,key):
    ds=image_dataset(metadata=f'image,{key}\nimages/0.png,a\nimages/1.png,b\nimages/2.png,a\n')
    m=run_metric(mid,ds,goal())
    assert m.value['counts']=={'a':2,'b':1}
    assert m.value['missing']==1 and m.evidence.sample_size==3
    assert m.value['fractions']['a']==pytest.approx(2/3)
    assert m.evidence.details['evidence_type']=='declared_metadata'

def test_timestamps_are_not_inferred_and_seasons_need_hemisphere(image_dataset):
    ds=image_dataset(metadata='image,timestamp\nimages/0.png,2026-01-01T03:00:00Z\nimages/1.png,2026-07-01T14:00:00Z\nimages/2.png,invalid\n')
    m=run_metric('UAV-03',ds,goal())
    assert m.value['utc_hour']['counts']=={'3':1,'14':1}
    assert m.value['missing']==1 and m.value['invalid']==1
    assert run_metric('UAV-04',ds,goal()).assessment_status=='insufficient_evidence'
    m=run_metric('UAV-04',ds,goal(uav_parameters={'hemisphere':'south'}))
    assert m.value['meteorological_seasons']['counts']=={'summer':1,'winter':1}

def test_altitude_units_invalid_and_explicit_bins(image_dataset):
    ds=image_dataset(metadata='image,altitude_meters\nimages/0.png,10\nimages/1.png,30\nimages/2.png,broken\n')
    assert run_metric('UAV-07',ds,goal()).assessment_status=='insufficient_evidence'
    ds.metadata['units']={'altitude_meters':'m'}
    m=run_metric('UAV-07',ds,goal(uav_parameters={'altitude_bins':{'low':[0,20],'high':[20,None]}}))
    assert m.value['median']==20 and m.value['missing']==1 and m.value['invalid']==1
    assert m.value['declared_bin_counts']=={'low':1,'high':1}

def test_geometry_known_boxes_and_projection_denominator(image_dataset):
    ds=image_dataset();g=goal(uav_parameters={'scale_bins':{'relative_occupancy':{'under_half':[0,.5],'over_half':[.5,None]}}})
    m=run_metric('UAV-08',ds,g);car=m.value['per_class']['car']
    assert car['width_pixels']['median']==10 and car['height_pixels']['median']==4
    assert car['area_pixels']['median']==40 and car['aspect_ratio']['median']==2.5
    assert car['relative_occupancy']['median']==.2
    assert car['relative_occupancy']['declared_bin_counts']['under_half']==1
    projected=run_metric('UAV-08',ds,g,{'output_path':['per_class','car','area_pixels','median']})
    assert projected.value==40 and projected.evidence.sample_size==1 and projected.unit=='pixel²'
    assert projected.evidence.details['source_denominator']==2

def test_occlusion_missing_is_not_none_and_border_is_proxy(image_dataset):
    ds=image_dataset();m=run_metric('UAV-10',ds,goal())
    assert m.assessment_status=='insufficient_evidence'
    assert m.value['per_class']['car']['missing']==1
    assert m.value['per_class']['car']['counts']=={}
    m=run_metric('UAV-11',ds,goal())
    assert m.value['border_contact_proxy']['car']['fraction']==0
    assert m.value['border_contact_proxy']['pedestrian']['fraction']==1
    assert m.value['true_truncation_status']=='insufficient_evidence'
    ds.objects[0]['occlusion']=2;ds.objects[0]['truncation']=1
    assert run_metric('UAV-10',ds,goal()).value['per_class']['car']['counts']=={'2':1}
    assert run_metric('UAV-11',ds,goal()).value['per_class']['car']['counts']=={'1':1}

def test_density_valid_negatives_missing_and_bounded_pairs(image_dataset):
    ds=image_dataset(labels=['0 .5 .5 .5 .4\n1 .5 .5 .5 .4\n','',None,'broken'])
    m=run_metric('UAV-13',ds,goal(uav_parameters={'max_spatial_pairs':1,'density_bins':{'empty':[0,1],'nonempty':[1,None]}}))
    assert m.evidence.sample_size==2
    assert m.value['objects_per_image']['mean']==1
    assert m.value['objects_per_megapixel']['mean']==5000
    assert m.value['pair_iou']['mean']==pytest.approx(1)
    assert m.value['pairs_examined']==1 and m.value['excluded_images']==2
    assert m.value['declared_density_bins']=={'empty':1,'nonempty':1}

def test_annotation_audit_does_not_claim_full_correctness(image_dataset):
    ds=image_dataset(labels=['0 .5 .5 .2 .2','bad','',None])
    m=run_metric('UAV-15',ds,goal())
    assert m.value['bbox_invalid_rate']['value']==pytest.approx(1/3)
    assert m.value['missing_label_rate']['value']==.25
    assert m.value['semantic_correctness']=='unverified'
    ds.metadata['annotation_audit']={'independent':True,'reviewer':'external panel','protocol':'double blind','reviewed_objects':20,'incorrect_objects':2}
    m=run_metric('UAV-15',ds,goal())
    assert m.value['independent_audit']['error_fraction']==.1
    assert m.value['semantic_correctness']=='unverified'

def test_reference_distribution_shift_without_accuracy_claim(image_dataset):
    ds=image_dataset(metadata='image,weather\nimages/0.png,clear\nimages/1.png,clear\nimages/2.png,clear\nimages/3.png,clear\n')
    assert run_metric('UAV-17',ds,goal()).assessment_status=='insufficient_evidence'
    g=goal(uav_parameters={'reference_distributions':{'weather':{'rainy':1}}})
    m=run_metric('UAV-17',ds,g)
    assert m.value['weather']['js_divergence_bits']==pytest.approx(1)
    assert m.value['weather']['denominator']==4
    assert m.value['weather']['unobserved_reference_categories']==['rainy']

def test_independence_reuses_existing_metrics(image_dataset):
    ds=image_dataset(metadata='image,split,flight\nimages/0.png,train,a\nimages/1.png,test,a\nimages/2.png,test,b\nimages/3.png,train,b\n',colors=[(5,5,5)]*4)
    g=goal(group_column='flight',split_column='split')
    m=run_metric('UAV-18',ds,g)
    assert m.value['duplicate_rate']['value']==.75
    assert m.value['split_overlap']['value']==1
    assert m.value['group_leakage']['value']==2
    assert m.value['collection_groups']['count']==2
    assert m.value['sample_adequacy'].startswith('unverified')

def test_sensor_schema_units_and_sync():
    ds=Dataset('telemetry.csv','hash','tabular',pd.DataFrame({'gps':[1,None],'t1':['2026-01-01T00:00:00Z','2026-01-01T00:00:01Z'],
        't2':['2026-01-01T00:00:00.050Z',None]}),{},metadata={'sensor_types':{'gps':['gps'],'imu':['missing']},'units':{'gps':'m'}})
    g=Goal(task_type='telemetry',expected_units={'gps':'degree'},uav_parameters={'synchronization':{'columns':['t1','t2'],'tolerance_seconds':.1}})
    m=run_metric('UAV-19',ds,g)
    assert m.value['sensors']['imu']['missing_fields']==['missing']
    assert m.value['sensors']['gps']['unit_mismatches']['gps']['actual']=='m'
    assert m.value['timestamp_delta_seconds']['mean']==.05
    assert m.value['pairs_within_declared_tolerance']==1 and m.value['missing_pairs']==1
    assert m.value['calibration_status']=='insufficient_evidence'

def test_temporal_checks_record_order_without_crossing_sequences():
    ds=Dataset('t.csv','hash','tabular',pd.DataFrame({'sequence_id':['a']*4+['b'],
        'timestamp':['2026-01-01T00:00:01Z','2026-01-01T00:00:01Z','2026-01-01T00:00:00Z',None,'2020-01-01T00:00:00Z'],
        'frame_index':[1,1,3,4,1],'track_id':['t']*5}),{})
    m=run_metric('UAV-20',ds,Goal(task_type='tracking'))
    assert m.value['repeated_timestamps']==1 and m.value['backwards_intervals']==1
    assert m.value['adjacent_timestamp_intervals']==2 and m.value['missing_timestamps']==1
    assert m.value['frame_gap_events']==1 and m.value['frame_order_violations']==1
    assert m.value['duplicate_track_frame_records']==1
    assert m.value['complete_track_frame_records']==5 and m.value['missing_track_ids']==0

def test_inapplicable_unknown_and_disk_preview(image_dataset,tmp_path):
    ds=image_dataset()
    assert run_metric('UAV-20',ds,goal()).assessment_status=='not_applicable'
    assert run_metric('UAV-19',ds,goal()).assessment_status=='insufficient_evidence'
    ds.disk_path=str(tmp_path/'fake.duckdb')
    assert run_metric('UAV-20',ds,goal()).assessment_status=='unsupported'
    ds.disk_path=None
    table=load_dataset('x.csv',b'x,label\n1,car\n')
    assert run_metric('UAV-08',table,Goal()).assessment_status=='not_applicable'

def test_success_is_not_a_requirement_pass_and_estimates_opt_in(image_dataset):
    ds=image_dataset();m=run_metric('UAV-08',ds,goal(),{'output_path':['per_class','car','relative_occupancy','median']})
    req=Requirement(requirement_id='size',description='User size criterion',metric_id='UAV-08',operator='gte',expected_value=.3)
    assert m.assessment_status=='measured' and m.state=='warning'
    assert evaluate(req,m).state=='fail'
    m.assessment_status='estimated';assert evaluate(req,m).state=='insufficient_evidence'
    req.parameters['accept_estimated_evidence']=True
    assert evaluate(req,m).state=='fail'

def test_learning_associations_and_no_invariance_claim(image_dataset):
    ds=image_dataset(labels=['0 .5 .5 .2 .2','0 .5 .5 .2 .2','1 .5 .5 .2 .2','1 .5 .5 .2 .2'],
        metadata='image,scene,lighting\nimages/0.png,road,daytime\nimages/1.png,road,daytime\nimages/2.png,field,nighttime\nimages/3.png,field,nighttime\n')
    ctx=DatasetContext(original_text='confirmed',confirmed=True,goal=goal())
    cert=assess(ds,ctx)
    assert cert.overall_status=='READY'
    v=cert.learning_views
    assert len(v)==6
    assoc=v['shortcut_vulnerability']['conditional_associations']['associations']['scene']['per_class']['car']
    assert assoc['normalized_mutual_information']==pytest.approx(1)
    assert assoc['denominator']==4
    assert v['invariance_support']['class_condition_image_counts']['lighting']['car']=={'daytime':2}
    assert 'unverified' in v['invariance_support']['label_preserving_transformations']
    assert 'Learning-potential diagnostics' in html_report(cert)

def test_absent_model_and_no_remote_enable(image_dataset,monkeypatch):
    monkeypatch.delenv('UAV_VISION_CONFIG',raising=False)
    ds=image_dataset(extra={'vision':{'enabled':True,'path':'https://example.com/model'}})
    assert run_metric('UAV-09',ds,goal()).assessment_status=='insufficient_evidence'
    assert run_metric('UAV-16',ds,goal()).assessment_status=='insufficient_evidence'

def test_local_encoder_classifier_batches_cache_and_discriminative(image_dataset,tmp_path,monkeypatch):
    """Fake model is only a unit-test double, never estimated-metric ground truth."""
    model=tmp_path/'model.onnx';model.write_bytes(b'test model fixture')
    calls=[]
    class Session:
        def __init__(self,path,providers):self.providers=providers
        def get_inputs(self):return [types.SimpleNamespace(name='input')]
        def get_providers(self):return self.providers
        def run(self,_,inputs):
            n=len(inputs['input']);calls.append(n)
            return [np.tile([1.,2.],(n,1))]
    monkeypatch.setitem(sys.modules,'onnxruntime',types.SimpleNamespace(InferenceSession=Session,get_available_providers=lambda:['CPUExecutionProvider']))
    spec={'path':str(model),'sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'name':'fixture','version':'unit-test','image_size':16}
    config={'enabled':True,'batch_size':1,'max_images':2,'max_objects':2,'cache_dir':str(tmp_path/'cache'),'device':'cuda','models':{'encoder':spec,'weather':{**spec,'labels':['clear','rainy']}}}
    path=tmp_path/'vision.json';path.write_text(json.dumps(config));monkeypatch.setenv('UAV_VISION_CONFIG',str(path))
    ds=image_dataset()
    with vision_session():
        m=run_metric('UAV-09',ds,goal());assert m.assessment_status=='estimated'
        assert m.value['per_class']['car']['mean_cosine_distance_to_centroid']==pytest.approx(0,abs=1e-6)
        assert m.value['providers']==['CPUExecutionProvider'] and m.value['count']==2
        confusion=run_metric('UAV-16',ds,goal())
        assert confusion.value['cross_class_nearest_similarity']['mean']==pytest.approx(1,abs=1e-6)
        assert len(calls)==2
    weather=run_metric('UAV-02',ds,goal())
    assert weather.assessment_status=='estimated' and weather.value['estimated_categories']['counts']=={'rainy':2}
    before=len(calls)
    assert run_metric('UAV-02',ds,goal()).assessment_status=='estimated' and len(calls)==before
    assert max(calls)==1

def test_helpers_known_quantiles_and_explicit_bins():
    assert summary([1,2,float('nan'),float('inf')],'m')['mean']==1.5
    assert bins([0,1,2],{'a':[0,1],'b':[1,None]})=={'a':1,'b':2}
    with pytest.raises(ValueError):bins([1],{'bad':[2,1]})
    from metrics.uav_metrics.vision import nearest_within
    scores,pairs=nearest_within(np.array([[1.,0.],[1.,0.],[0.,1.]]),[0,1,2])
    assert scores==[1.,1.,0.] and len(pairs)==3

def test_uav_context_and_registry_api(monkeypatch):
    from backend.context_extractor import interpret
    from main import app
    monkeypatch.setenv('CONTEXT_PROVIDER','deterministic')
    c=interpret('Detect cars in images in rainy weather with oblique viewpoint',{})
    assert c.goal.target_environment['weather']==['rainy']
    assert c.goal.target_environment['viewpoint']==['oblique'] and not c.confirmed
    response=TestClient(app).get('/api/metrics')
    assert response.status_code==200 and all(mid in response.json() for mid in PROPERTY_IDS)


def test_uav_api_confirmation_to_certificate_and_html(image_dataset):
    from main import app
    import time
    ds=image_dataset(labels=['0 .5 .5 .2 .2'])
    client=TestClient(app)
    response=client.post('/api/datasets/upload',files={'file':('api-uav.zip',Path(ds.source_path).read_bytes())})
    assert response.status_code==200
    dataset_id=response.json()['id']
    ctx=DatasetContext(original_text='Reviewed size requirement',confirmed=True,goal=Goal(task_type='object_detection',required_classes=['car'],requirements=[
        Requirement(requirement_id='manual-size',description='Reviewed size criterion',metric_id='UAV-08',operator='gte',expected_value=.5,
            parameters={'output_path':['per_class','car','relative_occupancy','median']})]))
    try:
        unconfirmed=ctx.model_dump();unconfirmed['confirmed']=False
        assert client.post('/api/assessments',json={'dataset_id':dataset_id,'context':unconfirmed}).status_code==422
        response=client.post('/api/assessments',json={'dataset_id':dataset_id,'context':ctx.model_dump()})
        assert response.status_code==202
        job=response.json()['id']
        for _ in range(200):
            progress=client.get(f'/api/assessments/{job}/progress').json()
            if progress['status'] in ('completed','assessment_error','error'):break
            time.sleep(.01)
        certificate=client.get(f'/api/assessments/{job}/certificate')
        assert certificate.status_code==200
        c=certificate.json()
        assert c['overall_status']=='NOT_READY' and c['findings'][0]['state']=='fail'
        assert all(mid in {m['metric_id'] for m in c['metrics']} for mid in PROPERTY_IDS)
        html=client.get(f'/api/assessments/{job}/export?format=html').text
        assert 'Measurement availability' in html and 'Sensor and temporal findings' in html
    finally:
        assert client.delete('/api/datasets/'+dataset_id).status_code==200


def test_unavailable_runtime_with_config_returns_insufficient(image_dataset,tmp_path,monkeypatch):
    model=tmp_path/'test.onnx';model.write_bytes(b'x')
    config=tmp_path/'c.json';config.write_text(json.dumps({'enabled':True,'models':{'encoder':{'path':str(model)}}}))
    monkeypatch.setenv('UAV_VISION_CONFIG',str(config));monkeypatch.setitem(sys.modules,'onnxruntime',None)
    assert run_metric('UAV-09',image_dataset(),goal()).assessment_status=='insufficient_evidence'


def test_invalid_model_and_projection_keep_uav_status_contract(image_dataset,tmp_path,monkeypatch):
    model=tmp_path/'test.onnx';model.write_bytes(b'x')
    config=tmp_path/'c.json';config.write_text(json.dumps({'enabled':True,'models':{'encoder':{'path':str(model),'name':'test','version':'1','sha256':'wrong'}}}))
    monkeypatch.setenv('UAV_VISION_CONFIG',str(config));monkeypatch.setitem(sys.modules,'onnxruntime',types.SimpleNamespace())
    ds=image_dataset()
    assert run_metric('UAV-09',ds,goal()).assessment_status=='error'
    m=run_metric('UAV-08',ds,goal(),{'output_path':['unknown']})
    assert m.assessment_status=='insufficient_evidence' and m.value is None


def test_uav_reference_numeric_units_and_weather_brightness_separation(image_dataset):
    a=load_dataset('a.csv',b'altitude,weather\n10,clear\n20,rainy\n',{'units':{'altitude':'m'}})
    b=load_dataset('b.csv',b'altitude,weather\n10,clear\n20,rainy\n',{'units':{'altitude':'ft'}})
    assert run_metric('reference_comparison',a,Goal(),{'columns':['altitude']},b).state=='insufficient_evidence'
    ds=image_dataset(colors=[(0,0,0)]*4)
    assert run_metric('UAV-02',ds,goal()).assessment_status=='insufficient_evidence'


def test_metadata_cannot_override_pixel_measurements(image_dataset):
    with pytest.raises(IngestionError,match='override'):
        image_dataset(metadata='image,laplacian_variance\nimages/0.png,100\n')


def test_family_requirement_needs_explicit_structured_criterion(image_dataset):
    m=run_metric('UAV-08',image_dataset(),goal())
    req=Requirement(requirement_id='unspecified',description='Needs a criterion',metric_id='UAV-08')
    assert evaluate(req,m).state=='insufficient_evidence'


def test_provider_cannot_invent_uav_thresholds():
    from backend.context_extractor import interpret
    class Provider:
        def extract(self,text,tags):
            return DatasetContext(original_text=text,goal=goal(uav_parameters={'scale_bins':{'relative_occupancy':{'small':[0,.02]}}}))
    c=interpret('Detect cars in images',{},provider=Provider())
    assert 'invalid output' in c.provider and c.goal.uav_parameters=={}


def test_provenance_does_not_export_private_source_location(image_dataset):
    ds=image_dataset(extra={'source':'C:/private-flight-collection','api_key':'private-secret'})
    ctx=DatasetContext(original_text='confirmed',confirmed=True,goal=goal(require_source=True))
    certificate=assess(ds,ctx)
    serialized=certificate.model_dump_json()
    assert 'private-flight-collection' not in serialized and 'private-secret' not in serialized
    assert certificate.dataset['provenance']['source_declared'] is True
