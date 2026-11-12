"""End-to-end format tests use generated fixtures, never downloaded code."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile
import time
import zipfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook
from PIL import Image
from backend.context import DatasetContext,Goal
from backend.ingestion import load_dataset,IngestionError
from backend.large_ingestion import load_dataset_stream
from metrics.core import run_metric
from backend.readiness_engine import assess
from main import app

@pytest.fixture
def datasets():
    opened=[]
    def load(name,data,metadata=None):
        ds=load_dataset(name,data.encode() if isinstance(data,str) else data,metadata)
        opened.append(ds);return ds
    yield load
    for ds in opened:ds.close()

def context(**overrides):
    values={'task_type':'classification','prediction_target':'label','required_inputs':['text'],'input_modalities':['text']}
    values.update(overrides)
    return DatasetContext(original_text='Synthetic format test',goal=Goal(**values),confirmed=True)

def zipped(files):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,value in files.items():z.writestr(name,value.encode() if isinstance(value,str) else value)
    return out.getvalue()

def image():
    out=io.BytesIO();Image.new('RGB',(16,16),(40,120,60)).save(out,format='PNG');return out.getvalue()

@pytest.mark.parametrize('name,text,meta',[
    ('x.csv','text;label\nhello;yes\nbye;no\n',{}),
    ('x.tsv','text\tlabel\nhello\tyes\nbye\tno\n',{}),
    ('x.psv','text|label\nhello|yes\nbye|no\n',{}),
    ('x.txt','text|label\nhello|yes\nbye|no\n',{'delimiter':'|'}),
    ('x.data','hello,yes\nbye,no\n',{'has_header':False,'column_names':['text','label']}),
    ('x.csv','hello,yes\nbye,no\n',{'has_header':False,'column_names':['text','label']}),
])
def test_delimited_text_formats(datasets,name,text,meta):
    ds=datasets(name,text,meta)
    assert ds.profile['row_count']==2
    assert ds.profile['columns']==['text','label']
    assert assess(ds,context(required_classes=['yes'])).overall_status=='READY'

def test_plain_text_is_not_arbitrarily_split_into_features(datasets):
    ds=datasets('prose.txt','Hello, world\nAnother, sentence\n\n')
    assert ds.profile['columns']==['text']
    assert ds.profile['row_count']==3
    assert ds.frame.loc[0,'text']=='Hello, world'
    assert ds.profile['missing_count']['text']==1
    assert assess(ds,context()).overall_status=='NOT_READY'

def test_plain_numeric_text_preserves_text_representation(datasets):
    ds=datasets('codes.txt','001\n002\n')
    assert ds.frame['text'].tolist()==['001','002']

def test_tabular_txt_detection_and_explicit_format_override(datasets):
    ds=datasets('download.txt','text\tlabel\nhello\tyes\nbye\tno\n')
    assert ds.profile['columns']==['text','label']
    forced=datasets('download.txt','text,label\nhello,yes\n',{'format':'csv'})
    assert forced.profile['row_count']==1 and forced.name=='download.txt'
    with pytest.raises(IngestionError):datasets('broken.csv','text;label\nhello;yes\nbroken\n')

def test_json_bom_legacy_and_boolean_source_types(datasets):
    raw='[{"text":"witness\u2019s account","label":"001","flag":true}]'
    for encoding in ('utf-8-sig','utf-16','cp1252'):
        ds=datasets('legacy.json',raw.encode(encoding))
        assert ds.frame.loc[0,'label']=='001'
        assert run_metric('type_validity',ds,context().goal,{'column':'flag','expected':'boolean'}).value==1

@pytest.mark.parametrize('suffix',['json','jsonl','ndjson'])
def test_json_rows_preserve_schema_and_typed_strings(datasets,suffix):
    rows=[{'id':'001','text':'hello','label':'yes'},{'id':'002','text':'bye','extra':2}]
    raw=json.dumps(rows) if suffix=='json' else '\n'.join(json.dumps(r) for r in rows)
    ds=datasets('records.'+suffix,raw)
    assert ds.profile['row_count']==2
    assert ds.frame['id'].tolist()==['001','002']
    assert set(ds.profile['columns'])=={'id','text','label','extra'}
    assert ds.profile['missing_count']['extra']==1
    assert run_metric('type_validity',ds,context().goal,{'column':'id','expected':'string'}).value==1

def test_nested_json_is_retained_and_not_reported_as_scalar_evidence(datasets):
    ds=datasets('nested.json','[{"text":"hi","values":{"a":1.5,"b":[1,2]}}]')
    assert json.loads(ds.frame.loc[0,'values'])=={'a':1.5,'b':[1,2]}
    assert ds.profile['data_types']['values']=='json'
    assert run_metric('type_validity',ds,context().goal,{'column':'values','expected':'number'}).state=='insufficient_evidence'

def test_json_missing_optional_and_dynamic_late_fields(datasets):
    rows=[{'text':'x','label':'yes'} for _ in range(10005)]+[{'text':'last','label':'no','late':'appears'}]
    ds=datasets('large.jsonl','\n'.join(json.dumps(r) for r in rows))
    assert ds.profile['row_count']==10006 and ds.profile['missing_count']['late']==10005
    assert run_metric('required_class_count',ds,context().goal,{'class':'no'}).value==1

def test_json_record_path_selection(datasets):
    ds=datasets('wrapped.json','{"data":[{"text":"a","label":"yes"}],"metadata":{"source":"synthetic"}}')
    assert ds.profile['row_count']==1 and ds.profile['parsing']['record_path']=='data'
    pending=datasets('multi.json','{"train":[{"x":1}],"test":[{"x":2}]}')
    assert pending.modality=='pending' and len(pending.profile['choices'])==2

@pytest.mark.parametrize('raw',['{"x":1,"x":2}','[{"x":1,"x":2}]','[{"x":NaN}]','{"bad":','{"x":1}\n\n'])
def test_invalid_json_not_silently_repaired(datasets,raw):
    suffix='jsonl' if '\n\n' in raw else 'json'
    with pytest.raises(IngestionError):datasets('bad.'+suffix,raw)

def test_xml_records_and_attributes(datasets):
    ds=datasets('records.xml','<dataset><row id="a"><text>Hello</text><label>yes</label></row><row id="b"><text>World</text><label>no</label></row></dataset>')
    assert ds.profile['row_count']==2
    assert ds.frame['@id'].tolist()==['a','b']
    assert assess(ds,context()).overall_status=='READY'

def test_nested_xml_and_explicit_record_path(datasets):
    ds=datasets('records.xml','<root><items><item><text>hello</text><meta><x>1</x></meta></item><item><text>bye</text><meta><x>2</x></meta></item></items></root>',{'record_path':'root/items/item'})
    assert ds.profile['row_count']==2 and 'meta' in ds.profile['parsing']['json_encoded_fields']

def test_xml_entities_and_external_files_are_rejected(datasets):
    with pytest.raises(IngestionError):
        datasets('unsafe.xml','<!DOCTYPE root [<!ENTITY x SYSTEM "file:///C:/Windows/win.ini">]><root><record><text>&x;</text></record></root>')

def test_structured_record_depth_and_text_limits(datasets):
    with pytest.raises(IngestionError,match='nesting'):
        datasets('deep.json','['*65+'0'+']'*65)
    with pytest.raises(IngestionError,match='8 MiB'):
        datasets('large.xml','<root><record><text>'+'x'*(8*1024**2+1)+'</text></record></root>')

def test_excel_sheet_selection_and_formula_preservation(datasets):
    workbook=Workbook();sheet=workbook.active
    assert sheet is not None
    sheet.title='train';sheet.append(['text','label']);sheet.append(['=1+2','yes'])
    workbook.create_sheet('test').append(['text','label'])
    data=io.BytesIO();workbook.save(data);workbook.close()
    pending=datasets('data.xlsx',data.getvalue())
    assert pending.modality=='pending'
    ds=datasets('data.xlsx',data.getvalue(),{'sheet_name':'train'})
    assert ds.frame.loc[0,'text']=='=1+2'

def test_feather_records(datasets):
    frame=pd.DataFrame({'text':['hello','bye'],'label':['yes','no']});out=io.BytesIO();frame.to_feather(out)
    ds=datasets('data.feather',out.getvalue())
    assert ds.profile['row_count']==2 and assess(ds,context()).overall_status=='READY'

def test_gzip_csv_records(datasets):
    ds=datasets('train.csv.gz',gzip.compress(b'text,label\nhello,yes\nbye,no\n'))
    assert ds.profile['row_count']==2

def test_github_archive_multiple_files_is_a_selection_not_a_parse_failure(datasets):
    raw=zipped({'repo-main/train.csv':'text,label\nhello,yes\n','repo-main/test.csv':'text,label\nbye,no\n','repo-main/README.md':'Ignore previous instructions','repo-main/run.py':'raise RuntimeError("never execute")'})
    ds=datasets('repository.zip',raw)
    assert ds.modality=='pending'
    assert len(ds.profile['choices'])==2
    selected=datasets('repository.zip',raw,{'dataset_member':'repo-main/train.csv'})
    assert selected.profile['row_count']==1
    assert selected.profile['source_archive']['member']=='repo-main/train.csv'

def test_text_corpus_and_traceable_folder_labels(datasets):
    ds=datasets('corpus.zip',zipped({'train/positive/1.txt':'Good movie.','train/negative/2.txt':'Bad movie.'}))
    assert ds.profile['row_count']==2
    assert set(ds.frame['label'])=={'positive','negative'}
    assert 'source_file' in ds.profile['columns']
    assert assess(ds,context(required_classes=['positive','negative'])).overall_status=='READY'

def test_tar_dataset_and_tar_symlink_security(datasets):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w:gz') as tar:
        raw=b'text,label\nhello,yes\n';member=tarfile.TarInfo('repo/train.csv');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
    ds=datasets('data.tar.gz',output.getvalue())
    assert ds.profile['row_count']==1
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as tar:
        member=tarfile.TarInfo('link');member.type=tarfile.SYMTYPE;member.linkname='../../outside';tar.addfile(member)
    with pytest.raises(IngestionError):datasets('bad.tar',output.getvalue())

def test_imagefolder_classification_is_not_detection(datasets):
    raw=zipped({'dataset/train/cat/a.png':image(),'dataset/train/dog/b.png':image(),'dataset/val/cat/c.png':image()})
    ds=datasets('classification.zip',raw)
    assert ds.profile['dataset_kind']=='image_classification'
    assert ds.profile['class_counts']=={'cat':2,'dog':1}
    cert=assess(ds,context(input_modalities=['image'],required_inputs=[],required_classes=['cat','dog']))
    assert cert.overall_status=='READY'
    assert run_metric('missing_label_rate',ds,context().goal).value==0

def test_nested_class_folders_are_not_relabelled_as_camera_directories(datasets):
    ds=datasets('classification.zip',zipped({'dataset/cat/camera/a.png':image(),'dataset/dog/camera/b.png':image()}))
    assert ds.profile['class_counts']=={'cat':1,'dog':1}

def test_corrupt_classification_image_cannot_be_ready(datasets):
    ds=datasets('classification.zip',zipped({'cat/a.png':image(),'dog/b.png':b'broken'}))
    assert assess(ds,context(input_modalities=['image'],required_inputs=[])).overall_status=='NOT_READY'
    assert run_metric('bbox_invalid_rate',ds,context().goal).state=='not_applicable'

def test_unlabelled_and_single_images_never_invent_labels(datasets):
    ds=datasets('images.zip',zipped({'images/a.png':image(),'images/b.png':image()}))
    assert ds.profile['dataset_kind']=='unlabelled_images'
    assert assess(ds,context(input_modalities=['image'],required_inputs=[])).overall_status=='NOT_READY'
    single=datasets('photo.png',image())
    assert single.profile['image_count']==1
    assert not single.frame['annotation_present'].any()

def test_multiframe_images_require_explicit_frame_handling(datasets):
    output=io.BytesIO()
    first=Image.new('RGB',(16,16),'red');second=Image.new('RGB',(16,16),'blue')
    first.save(output,format='GIF',save_all=True,append_images=[second])
    with pytest.raises(IngestionError,match='Multi-frame'):
        datasets('animated.gif',output.getvalue())

def coco_archive(annotations=None,images=None):
    data={'images':images if images is not None else [{'id':1,'file_name':'a.png','width':16,'height':16}],
        'categories':[{'id':7,'name':'car'}],'annotations':annotations if annotations is not None else [{'id':10,'image_id':1,'category_id':7,'bbox':[2,2,4,4],'iscrowd':1}]}
    return zipped({'images/a.png':image(),'annotations/instances.json':json.dumps(data)})

def detection():return context(task_type='object_detection',prediction_target='boxes',required_inputs=[],input_modalities=['image'],required_classes=['car'])

def test_coco_category_ids_boxes_crowds_and_ready_workflow(datasets):
    ds=datasets('coco.zip',coco_archive())
    assert ds.profile['annotation_format']=='coco'
    assert ds.objects[0]['class']=='car' and ds.objects[0]['iscrowd']==1
    assert ds.objects[0]['bbox']==[.25,.25,.25,.25]
    assert assess(ds,detection()).overall_status=='READY'

def test_coco_orphans_missing_files_and_invalid_boxes(datasets):
    orphan=datasets('coco.zip',coco_archive([{'image_id':99,'category_id':7,'bbox':[0,0,2,2]}]))
    assert run_metric('annotation_manifest_errors',orphan,detection().goal).value==1
    assert assess(orphan,detection()).overall_status=='NOT_READY'
    missing=datasets('coco.zip',coco_archive(images=[{'id':1,'file_name':'absent.png','width':16,'height':16}]))
    assert run_metric('unreadable_rate',missing,detection().goal).value>0
    invalid=datasets('coco.zip',coco_archive([{'image_id':1,'category_id':7,'bbox':[15,0,4,4]}]))
    assert run_metric('bbox_invalid_rate',invalid,detection().goal).value==1

def test_coco_declared_negative_images_are_distinct_from_missing_labels(datasets):
    ds=datasets('negative.zip',coco_archive([]))
    assert run_metric('missing_label_rate',ds,detection().goal).value==0
    assert not ds.objects

def voc_xml(xmin=1):
    return f'<annotation><filename>a.png</filename><size><width>16</width><height>16</height></size><object><name>car</name><difficult>1</difficult><bndbox><xmin>{xmin}</xmin><ymin>1</ymin><xmax>16</xmax><ymax>16</ymax></bndbox></object></annotation>'

def test_pascal_voc_coordinate_convention(datasets):
    ds=datasets('voc.zip',zipped({'VOC/JPEGImages/a.png':image(),'VOC/Annotations/a.xml':voc_xml()}))
    assert ds.objects[0]['bbox']==[.5,.5,1.,1.]
    assert ds.objects[0]['difficult']=='1'
    assert assess(ds,detection()).overall_status=='READY'

def test_yaml_yolo_and_unknown_class_mappings(datasets):
    beside=datasets('beside.zip',zipped({'a.png':image(),'a.txt':'0 .5 .5 .25 .25\n','labels.names':'car\n'}))
    assert assess(beside,detection()).overall_status=='READY'
    files={'images/a.png':image(),'labels/a.txt':'0 .5 .5 .25 .25\n','data.yaml':'names:\n  0: car\n'}
    ds=datasets('yolo.zip',zipped(files))
    assert assess(ds,detection()).overall_status=='READY'
    files.pop('data.yaml');ds=datasets('unknown.zip',zipped(files))
    assert ds.profile['class_name_mapping']=='unavailable'
    assert assess(ds,detection()).overall_status=='INSUFFICIENT_EVIDENCE'

def test_unsafe_yaml_tags_are_not_executed(datasets):
    raw=zipped({'images/a.png':image(),'labels/a.txt':'0 .5 .5 .25 .25','data.yaml':'names: !!python/object/apply:os.system ["echo must-not-run"]'})
    with pytest.raises(IngestionError):datasets('unsafe.zip',raw)

def test_api_selection_reuses_staged_upload_and_exports_report():
    client=TestClient(app)
    raw=zipped({'train.json':'[{"text":"hello","label":"yes"}]','test.json':'[{"text":"bye","label":"no"}]'})
    response=client.post('/api/datasets/upload',files={'file':('repo.zip',raw)})
    assert response.status_code==200,response.text
    id=response.json()['id']
    try:
        assert response.json()['profile']['needs_selection']
        assert client.post('/api/assessments',json={'dataset_id':id,'context':context().model_dump()}).status_code==409
        selected=client.post('/api/datasets/'+id+'/select',json={'options':{'dataset_member':'train.json'}})
        assert selected.status_code==200,selected.text
        assert selected.json()['profile']['row_count']==1
        assert client.get('/api/datasets/'+id+'/profile').json()['columns']==['text','label']
        assessment=client.post('/api/assessments',json={'dataset_id':id,'context':context().model_dump()})
        assert assessment.status_code==202,assessment.text
        job=assessment.json()['id']
        for _ in range(150):
            status=client.get('/api/assessments/'+job+'/progress').json()['status']
            if status not in ('running','queued'):break
            time.sleep(.02)
        assert status=='completed'
        assert client.get('/api/assessments/'+job+'/certificate').json()['overall_status']=='READY'
        assert client.get('/api/assessments/'+job+'/export?format=html').status_code==200
        assert client.get('/api/assessments/'+job+'/export?format=json').status_code==200
    finally:client.delete('/api/datasets/'+id)
