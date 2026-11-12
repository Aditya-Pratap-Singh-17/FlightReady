"""Native DET regressions use generated images and annotations, not user data."""
from pathlib import Path
import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from backend.context import Goal,DatasetContext
from backend.ingestion import IngestionError,load_dataset
from backend.large_ingestion import load_dataset_stream
from metrics.core import run_metric
from backend.readiness_engine import assess
from main import app

def archive(annotations='2,3,4,5,1,1,0,1\n',root='VisDrone2019-DET-train/',image_count=1,yolo=False):
    data=io.BytesIO()
    image=io.BytesIO();Image.new('RGB',(16,16),(25,120,70)).save(image,format='PNG')
    with zipfile.ZipFile(data,'w') as z:
        if yolo: z.writestr(root+'classes.txt','car\npedestrian\n')
        for i in range(image_count):
            z.writestr(root+f'images/{i}.png',image.getvalue())
            z.writestr(root+('labels/' if yolo else 'annotations/')+f'{i}.txt',annotations)
    return data.getvalue()

def goal(classes=None):
    return Goal(task_type='object_detection',prediction_target='boxes and classes',input_modalities=['image'],required_classes=classes or ['pedestrian'])

def test_native_visdrone_wrapped_and_pixel_normalization():
    ds=load_dataset('native.zip',archive())
    assert ds.profile['annotation_format']=='visdrone_det'
    assert ds.profile['image_count']==1
    assert ds.profile['class_counts']['pedestrian']==1
    obj=ds.objects[0]
    assert obj['bbox']==pytest.approx([.25,.34375,.25,.3125])
    assert obj['pixel_bbox']==[2,3,4,5]
    assert obj['occlusion']==1 and obj['truncation']==0
    assert assess(ds,DatasetContext(original_text='Synthetic DET test',goal=goal(),confirmed=True)).overall_status=='READY'

def test_ignored_regions_other_classes_and_flags_are_separate():
    text='0,0,4,4,0,0,0,0\n1,1,3,3,0,11,0,0\n2,2,4,4,0,4,0,0\n3,3,4,4,1,2,1,2\n'
    ds=load_dataset('visdrone.zip',archive(text))
    assert len(ds.objects)==1 and ds.objects[0]['class']=='people'
    assert ds.profile['class_counts']['pedestrian']==0
    assert ds.profile['annotation_summary']['ignored_annotations']==3
    assert ds.profile['annotation_summary']['score_zero_targets']==1
    assert run_metric('required_class_count',ds,goal(),{'class':'pedestrian'}).value==0

def test_zero_area_ignored_region_does_not_invalidate_target_boxes():
    ds=load_dataset('vis.zip',archive('0,0,4,0,0,0,0,0\n2,2,4,4,1,1,0,0\n'))
    assert bool(ds.frame.iloc[0]['annotation_valid'])
    assert ds.profile['annotation_summary']['invalid_ignored_boxes']==1
    assert len(ds.objects)==1
    assert run_metric('bbox_invalid_rate',ds,goal()).value==0

@pytest.mark.parametrize('line',['1,1,4,0,1,4,0,0','15,1,4,4,1,4,0,0','1,1,4,4,1,12,0,0','1,1,4,4,0.9,1,-1,-1','1,1,4,4,1,1,2,0','1,1,4,4,1,1,0,3','garbage'])
def test_invalid_target_annotations_are_findings_not_upload_crashes(line):
    ds=load_dataset('vis.zip',archive(line+'\n'))
    assert run_metric('bbox_invalid_rate',ds,goal()).value==1
    assert ds.issues[0]['kind']=='invalid_annotation'
    cert=assess(ds,DatasetContext(original_text='test',goal=goal(['car']),confirmed=True))
    assert cert.overall_status=='NOT_READY'

def test_valid_rows_retained_but_invalid_target_is_reported():
    ds=load_dataset('vis.zip',archive('2,3,4,5,1,1,0,0\n1,1,4,0,1,4,0,0\n'))
    assert len(ds.objects)==1
    assert ds.profile['annotation_summary']['invalid_target_records']==1
    assert not bool(ds.frame.iloc[0]['annotation_valid'])
    finding=run_metric('bbox_invalid_rate',ds,goal())
    assert finding.evidence.references[0].endswith(':2')
    assert 'zero/negative area' in finding.evidence.details['invalid_target_examples'][0]['reason']

def test_trailing_comma_and_unwrapped_root_supported():
    ds=load_dataset('vis.zip',archive('2,3,4,5,1,1,0,0,\n',root=''))
    assert ds.profile['archive_root']=='' and len(ds.objects)==1

def test_wrapped_yolo_preserves_original_support():
    ds=load_dataset('yolo.zip',archive('0 .5 .5 .25 .25\n',root='dataset/',yolo=True))
    assert ds.profile['annotation_format']=='yolo'
    assert ds.objects[0]['class']=='car'

def test_archive_staging_never_reads_whole_zip_into_memory(monkeypatch):
    original=Path.read_bytes
    def guarded(path):
        assert path.suffix.lower()!='.zip','ZIP staging must be passed as a path'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    ds=load_dataset_stream('vis.zip',io.BytesIO(archive()))
    assert ds.profile['processing']['archive_access']=='streamed_members'

def test_many_entries_are_supported_and_pairwise_detection_is_bounded(monkeypatch):
    monkeypatch.setenv('NEAR_DUPLICATE_DISTANCE','1')
    monkeypatch.setenv('MAX_NEAR_DUPLICATE_IMAGES','1')
    ds=load_dataset('vis.zip',archive(image_count=2))
    assert ds.near_distance is None
    assert 'skipped' in ds.profile['near_duplicate_analysis']
    assert run_metric('near_duplicates',ds,goal()).state=='insufficient_evidence'

def test_object_and_expansion_limits_still_enforced(monkeypatch):
    monkeypatch.setenv('MAX_ANNOTATED_OBJECTS','0')
    with pytest.raises(IngestionError,match='MAX_ANNOTATED_OBJECTS'):load_dataset('vis.zip',archive())
    monkeypatch.setenv('MAX_ANNOTATED_OBJECTS','1000000')
    monkeypatch.setenv('MAX_ZIP_MB','0')
    with pytest.raises(IngestionError,match='size'):load_dataset('vis.zip',archive())

def test_api_recognizes_native_archive_and_new_image_limit():
    client=TestClient(app)
    assert client.get('/api/config').json()['max_image_upload_bytes']==2*1024**3
    response=client.post('/api/datasets/upload',files={'file':('VisDrone2019-DET-train.zip',archive())})
    assert response.status_code==200,response.text
    value=response.json()
    try: assert value['profile']['annotation_format']=='visdrone_det'
    finally: client.delete('/api/datasets/'+value['id'])

def test_corrupt_archive_returns_meaningful_http_error():
    response=TestClient(app).post('/api/datasets/upload',files={'file':('broken.zip',b'bad')})
    assert response.status_code==422
    assert 'corrupt ZIP' in response.json()['detail']
