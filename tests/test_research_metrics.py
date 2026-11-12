"""Exact synthetic evidence, not empirical validation of UAV model performance."""
import copy
import io
import json
import zipfile
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter
from backend.context import Goal, DatasetContext, Requirement
from backend.ingestion import Dataset, load_dataset
from metrics.core import run_metric
from backend.readiness_engine import assess, evaluate
from metrics.uav_metrics.research import object_records, scale_analysis, scenario_evidence, condition, conjunction
from metrics.uav_metrics.pixels import pixel_features, local_measurements
from metrics.uav_metrics.audit import annotation_audit, negative_coverage, match_pairs
from metrics.uav_metrics.independence import collection_evidence
from backend.workbench.analytics import numeric_metrics


@pytest.fixture
def make_images():
    owned=[]
    def make(labels=None, pixels=None, metadata=None, extra=None):
        labels=labels if labels is not None else ['0 .5 .5 .2 .2\n','1 .5 .5 .5 .5\n','',None]
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr('classes.txt','drone\nbird\n')
            for i,label in enumerate(labels):
                a=pixels[i] if pixels is not None else np.tile(np.arange(100,dtype=np.uint8),(50,1))+i*20
                image=io.BytesIO();Image.fromarray(a).save(image,format='PNG')
                z.writestr(f'images/{i}.png',image.getvalue())
                if label is not None:z.writestr(f'labels/{i}.txt',label)
            if metadata is not None:z.writestr('metadata.csv',metadata)
        ds=load_dataset('research.zip',stream.getvalue(),extra)
        owned.append(ds)
        return ds
    yield make
    for ds in owned:ds.close()


def goal(**kwargs):
    return Goal(task_type='object_detection',required_classes=['drone'],**kwargs)


@pytest.mark.parametrize('mode,scaleup,expected', [('letterbox',True,(128,64)),('letterbox',False,(20,10)),('stretch',True,(128,128))])
def test_native_model_geometry(make_images,mode,scaleup,expected):
    ds=make_images(labels=['0 .5 .5 .2 .2'])
    g=goal(uav_parameters={'detector_preprocessing':{'input_size':[640,640],'mode':mode,'scaleup':scaleup}})
    s=scale_analysis(ds,g);stats=s['native_and_input_statistics']
    assert stats['width_pixels']['median']==20 and stats['height_pixels']['median']==10
    assert stats['area_pixels']['median']==200 and stats['minimum_side_pixels']['median']==10
    assert stats['model_width_pixels']['median']==pytest.approx(expected[0])
    assert stats['model_height_pixels']['median']==pytest.approx(expected[1])


def test_rectangular_letterbox_missing_geometry_and_denominators(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2\n0 .2 .2 .1 .1','1 .5 .5 .8 .8',''])
    s=scale_analysis(ds,goal())
    assert s['model_input']['status']=='insufficient_evidence'
    assert 'model_width_pixels' not in s['native_and_input_statistics']
    assert s['object_counts']['small']==2 and s['image_counts']['small']==1
    assert s['object_percentages']['small']==pytest.approx(200/3)
    assert s['small_object_coverage']['status']=='insufficient_evidence'
    g=goal(uav_parameters={'small_object_criteria':{'classes':['drone'],'native_area_range':[0,300],'minimum_objects':2}})
    assert scale_analysis(ds,g)['small_object_coverage']['status']=='pass'


@pytest.mark.parametrize('box', [[.5,.5,0,.1],[.5,.5,-1,.1],[0,0,.1,.1],[.5,.5,float('nan'),.1]])
def test_reloaded_invalid_boxes_not_measured(make_images,box):
    ds=make_images(labels=['0 .5 .5 .2 .2']);ds.objects[0]['bbox']=box
    assert object_records(ds,goal())==[]


@pytest.mark.parametrize('spec', [ {'mode':'letterbox','input_size':640}, {'mode':'square','input_size':640,'scaleup':True}, {'mode':'letterbox','input_size':float('nan'),'scaleup':True} ])
def test_preprocessing_requires_explicit_geometry(make_images,spec):
    with pytest.raises(ValueError):scale_analysis(make_images(),goal(uav_parameters={'detector_preprocessing':spec}))


def test_uniform_noise_blur_exposure_are_distinct():
    a=np.full((64,64),128,dtype=np.uint8)
    uniform=pixel_features(a)
    assert uniform['laplacian_variance']==0 and uniform['high_frequency_residual_mad']==0
    noise=np.clip(a.astype(float)+np.random.default_rng(4).normal(0,20,a.shape),0,255).astype(np.uint8)
    noisy=pixel_features(noise)
    blurred=pixel_features(np.asarray(Image.fromarray(noise).filter(ImageFilter.GaussianBlur(2))))
    assert noisy['high_frequency_residual_mad']>blurred['high_frequency_residual_mad']
    assert noisy['laplacian_variance']>uniform['laplacian_variance']
    assert pixel_features(np.zeros((8,8)))['underexposed_fraction']==1
    assert pixel_features(np.full((8,8),255))['saturated_fraction']==1
    assert pixel_features(np.zeros((1,1)))['laplacian_variance'] is None


def test_local_ring_excludes_target_border_and_sampling(make_images):
    a=np.zeros((100,100),dtype=np.uint8);a[40:60,40:60]=255
    ds=make_images(labels=['0 .5 .5 .2 .2'],pixels=[a])
    result=local_measurements(ds,goal())
    assert result['sampling']['objects_inspected']==1
    assert result['distributions']['luminance_difference']['mean']==1
    assert result['distributions']['normalized_local_contrast']['mean']==1
    assert result['distributions']['background_ring.entropy_bits']['mean']==0
    ds.objects[0]['bbox']=[.1,.1,.2,.2]
    assert local_measurements(ds,goal())['sampling']['objects_inspected']==1
    ds.objects[0]['bbox']=[.5,.5,1,1]
    whole=local_measurements(ds,goal())
    assert whole['distributions']['background_ring.entropy_bits']['status']=='insufficient_evidence'


def test_local_budgets_reproducible_shared_cache(make_images,monkeypatch):
    ds=make_images(labels=['0 .5 .5 .2 .2']*8)
    g=goal(uav_parameters={'local_analysis':{'max_images':2,'max_objects':1,'seed':11}})
    from metrics.uav_metrics.vision import vision_session
    with vision_session():
        first=local_measurements(ds,g)
        assert local_measurements(ds,g) is first
        assert run_metric('UAV-12',ds,g).value['local_complexity'] is not None
        assert run_metric('UAV-14',ds,g).value['object_local']['sampling']==first['sampling']
    assert first==local_measurements(ds,g)
    assert first['sampling']['objects_inspected']==1
    assert first['sampling']['object_coverage']==1/8
    assert first['sampling']['scope']=='bounded_sample'


def test_local_unavailable_cannot_project_fake_zero(make_images):
    ds=make_images(labels=['0 .5 .5 1 1'])
    m=run_metric('UAV-14',ds,goal(),{'output_path':['object_local','distributions','background_ring.entropy_bits','count']})
    assert m.assessment_status=='insufficient_evidence' and m.value is None


def test_local_ring_excludes_other_annotations(make_images):
    a=np.zeros((100,100),dtype=np.uint8);a[40:60,40:80]=255
    ds=make_images(labels=['0 .5 .5 .2 .2\n1 .7 .5 .2 .2'],pixels=[a])
    local=local_measurements(ds,goal())
    assert local['per_class']['drone']['background_ring.entropy_bits']['mean']==0
    assert local['per_class']['drone']['luminance_difference']['mean']==1


def test_missing_staged_pixels_never_draws_zero_distribution(make_images):
    from metrics.uav_metrics.presentation import evidence_sections
    ds=make_images(labels=['0 .5 .5 .2 .2']);ds.source_path=None
    metric=run_metric('UAV-14',ds,goal())
    local=metric.value['object_local']
    assert local['status']=='insufficient_evidence'
    sections=evidence_sections([metric.model_dump(mode='json')])
    assert all(not p['rows'] for p in next(s for s in sections['sections'] if s['id']=='object_local')['plots'])


@pytest.mark.parametrize('settings',[{'max_images':True},{'max_objects':5000},{'ring_expansion':float('nan')}])
def test_pixel_resource_configuration_rejects_invalid_values(make_images,settings):
    with pytest.raises(ValueError):local_measurements(make_images(),goal(uav_parameters={'local_analysis':settings}))


def audit_input(images,**extra):
    return {'independent':True,'evidence_type':'human_verified','reviewer':'Synthetic fixture reviewer','protocol':'Independent complete reference','source_id':'test-only','images':images,**extra}


def obj(cls,box):return {'class':cls,'bbox':box}


def test_audit_categories_one_to_one_and_denominators(make_images):
    ds=make_images(labels=['1 .2 .2 .1 .1\n1 .5 .5 .1 .1\n1 .5 .5 .1 .1\n0 .62 .8 .2 .2\n0 .1 .8 .05 .05',''])
    refs=[obj('drone',[.2,.2,.1,.1]),obj('bird',[.5,.5,.1,.1]),obj('drone',[.8,.2,.1,.1]),obj('drone',[.5,.8,.2,.2])]
    ds.metadata['annotation_audit']=audit_input([{'image':'images/0.png','reference_complete':True,'reference_objects':refs},{'image':'images/1.png','reference_complete':True,'reference_objects':[obj('drone',[.5,.5,.1,.1])]}])
    audit=annotation_audit(ds,goal());counts=audit['error_counts']
    assert counts['incorrect_classes']==1 and counts['localization_errors']==1
    assert counts['duplicate_annotations']==1 and counts['spurious_annotations']==1
    assert counts['missing_objects']==2 and counts['incorrect_negative_images']==1
    assert audit['reference_objects']==5 and audit['annotations_reviewed']==5
    assert audit['error_rates']['missing_objects']['fraction']==.4
    assert audit['image_coverage']==1
    assert audit['semantic_correctness']=='audited_subset_only'


@pytest.mark.parametrize('mode',['class_aware','class_agnostic'])
def test_class_audit_modes_and_unverified_model_reference(make_images,mode):
    ds=make_images(labels=['1 .5 .5 .2 .2'])
    ds.metadata['annotation_audit']=audit_input([{'image':'images/0.png','reference_complete':True,'reference_objects':[obj('drone',[.5,.5,.2,.2])]}],matching_mode=mode)
    assert annotation_audit(ds,goal())['error_counts']['incorrect_classes']==1
    ds.metadata['annotation_audit']['evidence_type']='model_prediction'
    assert annotation_audit(ds,goal())['semantic_correctness']=='unverified'


def test_audit_incomplete_reference_and_zero_denominators(make_images):
    ds=make_images(labels=[''])
    ds.metadata['annotation_audit']=audit_input([{'image':'images/0.png','reference_complete':False,'reference_objects':[]}])
    with pytest.raises(ValueError,match='reference_complete'):annotation_audit(ds,goal())
    ds.metadata['annotation_audit']['images'][0].update(reference_complete=True,negative_verified=True)
    a=annotation_audit(ds,goal())
    assert a['error_rates']['missing_objects']['fraction'] is None
    assert negative_coverage(ds,goal(),a)['verified_negative_images']==1


def test_ignored_reference_does_not_become_verified_correct_or_spurious(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2'])
    ds.metadata['annotation_audit']=audit_input([{'image':'images/0.png','reference_complete':True,'reference_objects':[{**obj('drone',[.5,.5,.2,.2]),'ignored':True}]}])
    audit=annotation_audit(ds,goal())
    assert audit['error_counts']['ignored_uncertain_objects']==1
    assert audit['annotations_reviewed']==0 and audit['error_counts']['spurious_annotations']==0
    assert audit['error_rates']['spurious_annotations']['fraction'] is None


def test_semantic_unverified_is_not_a_failed_observed_requirement(make_images):
    m=run_metric('UAV-15',make_images(),goal(),{'output_path':['semantic_correctness']})
    assert m.assessment_status=='insufficient_evidence' and m.value is None


def test_training_subset_scopes_audits_without_rewriting_parent(make_images):
    from backend.workbench.datasets import subset
    ds=make_images(labels=['0 .5 .5 .2 .2','0 .5 .5 .2 .2'])
    ds.metadata['annotation_audit']=audit_input([{'image':f'images/{i}.png','reference_complete':True,'reference_objects':[obj('drone',[.5,.5,.2,.2])]} for i in range(2)],dataset_sha256=ds.sha256)
    original=copy.deepcopy(ds.metadata)
    selected=subset(ds,{'images/0.png'})
    audit=annotation_audit(selected,goal())
    assert audit['images_audited']==1 and audit['image_coverage']==1
    assert audit['error_counts']['missing_objects']==0 and ds.metadata==original
    assert selected.metadata['annotation_audit']['dataset_sha256']==selected.sha256


def test_unscoped_legacy_audit_not_promoted_to_training_subset(make_images):
    from backend.workbench.datasets import subset
    ds=make_images();ds.metadata['annotation_audit']={'independent':True,'reviewer':'fixture','protocol':'fixture','reviewed_objects':100,'incorrect_objects':0}
    selected=subset(ds,{'images/0.png'})
    result=run_metric('UAV-15',selected,goal())
    assert 'independent_audit' not in result.value
    assert ds.metadata['annotation_audit']['reviewed_objects']==100


def test_negative_categories_never_fill_missing_labels(make_images):
    ds=make_images()
    v=negative_coverage(ds,goal(),{})
    assert v['explicit_negative_images']==1 and v['unknown_label_images']==1
    assert v['other_class_only_images']==1 and v['denominator']==3
    assert v['negative_proportion']==pytest.approx(1/3)
    assert v['verified_negative_images']==0
    assert v['per_class']['drone']['annotated_target_free_images']==2
    from backend.context import ScenarioRequirement
    g=goal(scenarios=[ScenarioRequirement(name='target presence',conditions={},required_class='drone')])
    assert negative_coverage(ds,g,{})['per_scenario'][0]['negative_image_count'] is None


def test_model_hard_negative_requires_identity_and_settings(make_images):
    ds=make_images();ds.metadata['hard_negative_candidates']=[{'image':'images/2.png','source_id':'x','reason':'test candidate','evidence_type':'model_prediction'}]
    with pytest.raises(ValueError,match='identity'):negative_coverage(ds,goal(),{})
    ds.metadata['hard_negative_candidates'][0].update(model_id='fixture',confidence_threshold=.3)
    v=negative_coverage(ds,goal(),{})
    assert v['candidate_count']==1 and v['hard_negative_candidates'][0]['status']=='unverified_candidate'


def test_scenario_same_object_intersection_and_unknown(make_images):
    ds=make_images(labels=['0 .5 .5 .8 .8\n1 .2 .2 .1 .1','0 .5 .5 .1 .1','0 .5 .5 .1 .1'],metadata='image,lighting\nimages/0.png,night\nimages/1.png,day\n')
    g=goal(uav_parameters={'object_size_bins':{'small':[0,100],'large':[100,None]}})
    v=scenario_evidence(ds,g,{'class':'drone','object_size':'small','lighting':'night'},minimum=1)
    assert v['matching_image_count']==0 and v['matching_object_count']==0
    assert v['known_nonmatching_count']==2 and v['unknown_count']==1
    assert v['status']=='insufficient_evidence'
    v=scenario_evidence(ds,g,{'class':'bird','object_size':'small','lighting':'night'},minimum=1)
    assert v['matching_image_count']==1 and v['matching_object_count']==1 and v['status']=='pass'
    unconfirmed=scenario_evidence(ds,goal(),{'class':'drone','object_size':'small'})
    assert unconfirmed['matching_image_count']==0 and unconfirmed['unknown_count']==3


def test_false_overrides_unknown_but_unknown_never_proves_absence(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2'],metadata='image,lighting\nimages/0.png,day\n')
    assert scenario_evidence(ds,goal(),{'lighting':'night','weather':'rain'})['unknown_count']==0
    assert scenario_evidence(ds,goal(),{'weather':'rain'})['unknown_count']==1
    assert conjunction([None,False]) is False


def test_observed_scenario_lower_bound_cannot_fail_unknown_minimum(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2']*2,metadata='image,weather\nimages/0.png,rain\n')
    m=run_metric('scenario_count',ds,goal(),{'conditions':{'weather':'rain'}})
    req=Requirement(requirement_id='coverage',description='test',metric_id='scenario_count',operator='gte',expected_value=2)
    assert m.value==1 and evaluate(req,m).state=='insufficient_evidence'
    req.expected_value=1
    assert evaluate(req,m).state=='pass'


def test_independence_cached_hash_bounded_and_cross_split(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2']*3,pixels=[np.zeros((50,100),dtype=np.uint8)]*3,
        metadata='image,split,flight_id\nimages/0.png,train,a\nimages/1.png,test,a\nimages/2.png,train,b\n')
    g=goal()
    out=collection_evidence(ds,g)
    assert out['unique_exact_hashes']==1 and out['exact_duplicate_images']==2
    assert out['cross_split_exact_duplicate_groups']==1
    assert out['collection_groups']['flight_id']['group_count']==2
    assert out['collection_groups']['flight_id']['cross_split_group_count']==1
    assert out['near_duplicate_candidates']['cross_split_candidate_pairs']>0
    limited=collection_evidence(ds,goal(uav_parameters={'near_duplicate_analysis':{'max_tree_visits':1}}))
    assert limited['near_duplicate_candidates']['tree_visits']<=1
    assert not limited['near_duplicate_candidates']['complete_retrieval']


def test_old_cached_dataset_near_evidence_stays_unknown(make_images):
    ds=make_images();ds.frame=ds.frame.drop(columns=['dhash64'])
    out=collection_evidence(ds,goal())
    assert out['near_duplicate_candidates']['status']=='insufficient_evidence'
    assert out['independence_status'].startswith('uncertain')


def test_global_candidate_pair_budget_is_bounded(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2']*10)
    out=collection_evidence(ds,goal(uav_parameters={'near_duplicate_analysis':{'max_candidate_pairs':2}}))['near_duplicate_candidates']
    assert out['retrieved_candidate_pairs']<=2 and not out['complete_retrieval']


def test_new_evidence_no_implicit_score_changes_or_historical_rewrite(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2'])
    ctx=DatasetContext(original_text='Synthetic confirmed goal',goal=goal(),confirmed=True)
    cert=assess(ds,ctx);snapshot=cert.model_dump(mode='json')
    assert cert.overall_status=='READY'
    assert len(cert.quality_profile['dimensions'])==6
    assert cert.quality_profile['profile_version']=='adaptive-1.0'
    assert len(cert.quality_measurements['research_evidence']['sections'])==8
    assert all(m.reproducibility['implementation_version']=='2.0' for m in cert.metrics if m.metric_id in ('UAV-08','UAV-12','UAV-14','UAV-15','UAV-17','UAV-18'))
    from metrics.quality_aggregation import aggregate_quality
    again=aggregate_quality(snapshot,profile=snapshot['quality_profile']['profile'])
    assert again['calculation_sha256']==snapshot['quality_profile']['calculation_sha256']
    assert cert.model_dump(mode='json')==snapshot
    assert all(d['score'] is None for d in again['dimensions'] if d['evidence_coverage']<1)


def test_analytics_does_not_correlate_unavailable_zero_counts():
    values=numeric_metrics({'metrics':[{'metric_id':'UAV-14','state':'warning','value':{'object_local':{'status':'insufficient_evidence','count':0},'good':{'status':'measured','count':0}}}]})
    assert 'UAV-14.object_local.count' not in values
    assert values['UAV-14.good.count']==0


def test_size_condition_does_not_override_required_class(make_images):
    ds=make_images(labels=['0 .5 .5 .2 .2'])
    evidence=scenario_evidence(ds,goal(),{'class':'bird'},required_class='drone')
    assert evidence['matching_image_count']==0 and evidence['known_nonmatching_count']==1


def test_audit_syntax_failure_does_not_become_semantic_omissions(make_images):
    ds=make_images(labels=['bad'])
    ds.metadata['annotation_audit']=audit_input([{'image':'images/0.png','reference_complete':True,'reference_objects':[obj('drone',[.5,.5,.2,.2])]}])
    audit=annotation_audit(ds,goal())
    assert audit['error_counts']['invalid_syntax_images']==1
    assert audit['error_counts']['missing_objects']==0
    assert audit['error_rates']['missing_objects']['fraction'] is None


def test_pairwise_cap_and_reference_missingness(make_images):
    from backend.context import ScenarioRequirement
    from metrics.uav_metrics.research import scenarios
    ds=make_images(labels=['0 .5 .5 .2 .2'],metadata='image,weather,lighting\nimages/0.png,rain,night\n')
    g=goal(scenarios=[ScenarioRequirement(name='rain',conditions={'weather':'rain'}),ScenarioRequirement(name='night',conditions={'lighting':'night'})],
        uav_parameters={'scenario_pairwise_limit':1,'scenario_reference_fractions':{'rain':.25}})
    out=scenarios(ds,g)
    assert out['scenarios'][0]['reference_comparison']['difference_from_declared_fraction']==.75
    assert len(out['pairwise_summaries'])==1 and out['pairwise_summaries'][0]['matching_image_count']==1


def test_native_area_recall_uses_truth_size_not_prediction_size():
    from backend.workbench.detector import native_size_outcomes,detection_errors
    truth=[obj('drone',[.5,.5,.1,.1]),obj('drone',[.5,.5,.8,.8])]
    predictions=[{**obj('drone',[.5,.5,.1,.1]),'confidence':.8}]
    _,annotated,missed=detection_errors(truth,predictions,.5)
    sizes=native_size_outcomes(truth,annotated,missed,100,100)
    assert sizes['small']['ground_truth_objects']==1 and sizes['small']['recall']==1
    assert sizes['medium']['ground_truth_objects']==1 and sizes['medium']['recall']==0
    assert sizes['large']['recall'] is None


def test_public_upload_readiness_and_exports(make_images,tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.workbench import api
    from sql.store import Store
    from backend.workbench.worker import run
    store=Store(tmp_path/'public')
    class Queue:
        def enqueue(self,task,config):
            return store.create('job',{'task':task,'config':config,'status':'queued','stage':'Waiting','error':None,'result':None})
    monkeypatch.setattr(api,'services',lambda:(store,Queue()))
    app=FastAPI();app.include_router(api.router)
    ds=make_images(labels=['0 .5 .5 .2 .2',''])
    with TestClient(app) as client:
        response=client.post('/api/v1/datasets/import',files={'file':('research.zip',Path(ds.source_path).read_bytes(),'application/zip')})
        assert response.status_code==201,response.text
        dataset=response.json()
        ctx=DatasetContext(original_text='Synthetic API evidence',goal=goal(),confirmed=True)
        response=client.post('/api/v1/readiness',json={'dataset_id':dataset['id'],'context':ctx.model_dump(mode='json')})
        assert response.status_code==202,response.text
        job=response.json();run(store,job['id'])
        result=client.get('/api/v1/jobs/'+job['id']).json()['payload']
        assert result['status']=='completed',result.get('error')
        cert=result['result'];assert cert['overall_status']=='READY'
        assert cert['quality_measurements']['research_evidence']['version']=='research-evidence-2.0'
        for format in ('json','csv','html','pdf'):
            exported=client.get(f'/api/v1/reports/{job["id"]}?format={format}')
            assert exported.status_code==200,exported.text[:300] if format!='pdf' else exported.status_code
            if format in ('json','csv'):assert 'scale_analysis' in exported.text and 'negative_coverage' in exported.text
            if format=='html':assert 'Object-scale distributions' in exported.text
            if format=='pdf':assert exported.content.startswith(b'%PDF')
    store.engine.dispose()


def test_persistence_report_formats_and_full_pipeline(make_images,tmp_path):
    from sql.store import Store
    from backend.workbench.datasets import import_file,load
    from backend.workbench.reporting import document,html_document,pdf_document
    from backend.reports import html_report
    ds=make_images(labels=['0 .5 .5 .2 .2',''])
    store=Store(tmp_path/'store')
    with Path(ds.source_path).open('rb') as incoming:record=import_file(store,'research.zip',incoming)
    reloaded=load(store,record['id'])
    assert list(reloaded.frame.dhash64)==list(ds.frame.dhash64)
    ctx=DatasetContext(original_text='Synthetic goal',goal=goal(),confirmed=True)
    cert=assess(reloaded,ctx)
    assert cert.overall_status=='READY'
    assert cert.quality_measurements['research_evidence']['sections'][1]['sample_size']==1
    job=store.create('job',{'task':'readiness','status':'completed','result':cert.model_dump(mode='json')})
    doc=document(store,job)
    assert 'Independent annotation audit categories' in html_report(cert)
    assert 'Target-negative image coverage' in html_document(doc)
    import pymupdf
    pdf=pymupdf.open(stream=pdf_document(doc),filetype='pdf')
    assert 'Enhanced research measurements' in ''.join(page.get_text() for page in pdf)
    assert 'Object-scale distributions' in ''.join(page.get_text() for page in pdf)
    store.engine.dispose()
