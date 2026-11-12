"""Exercise disk processing on small fixtures; the optional 2 GiB run is separate."""
from pathlib import Path
import hashlib
import io
import json
import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.context import Goal, DatasetContext, ScenarioRequirement, Requirement
from backend.ingestion import load_dataset, IngestionError
from backend.large_ingestion import load_dataset_stream, MAX_RECORD_BYTES
from metrics.core import run_metric
from backend.readiness_engine import assess
from main import app

@pytest.fixture
def large(monkeypatch):
    monkeypatch.setenv('LARGE_DATASET_THRESHOLD_MB','0')
    owned=[]
    def create(text,metadata=None,name='test.csv'):
        data=text.encode('utf-8') if isinstance(text,str) else text
        ds=load_dataset_stream(name,io.BytesIO(data),metadata)
        owned.append(ds)
        return ds
    yield create
    for ds in owned: ds.close()

def context(**kwargs):
    values={'task_type':'classification','prediction_target':'label','required_inputs':['x']}
    values.update(kwargs)
    return DatasetContext(original_text='Synthetic large-data test',goal=Goal(**values),confirmed=True)

def test_disk_backend_and_complete_counts(large):
    text='x,label\n'+''.join(f'{i},car\n' for i in range(1200))
    ds=large(text)
    assert ds.disk_path and Path(ds.disk_path).exists()
    assert len(ds.frame)==100
    assert ds.profile['row_count']==1200
    assert ds.sha256==hashlib.sha256(text.encode()).hexdigest()
    cert=assess(ds,context(required_classes=['car']))
    assert cert.overall_status=='READY'
    assert next(m for m in cert.metrics if m.metric_id=='class_distribution').value=={'car':1200}
    assert run_metric('row_count',ds,context().goal).value==1200
    json.dumps(cert.model_dump(),allow_nan=False)

def test_late_bad_row_cannot_hide_outside_preview(large):
    text='x,label\n'+''.join(f'{i},car\n' for i in range(1200))+'99999,pedestrian\n'
    ds=large(text)
    cert=assess(ds,context(valid_ranges={'x':(0,2000)},required_classes=['pedestrian']))
    assert cert.overall_status=='NOT_READY'
    count=run_metric('required_class_count',ds,context().goal,{'class':'pedestrian'})
    assert count.value==1
    assert count.evidence.details['scope']=='full_dataset'

def test_missingness_and_duplicates_across_batches(large):
    ds=large('x,label\n1,car\n,car\n1,car\n')
    assert run_metric('missingness',ds,context().goal,{'column':'x'}).value==pytest.approx(1/3)
    assert run_metric('duplicate_rate',ds,context().goal).value==pytest.approx(1/3)
    assert assess(ds,context(max_missing_fraction={'x':0})).overall_status=='NOT_READY'

def test_numeric_types_ranges_and_nonfinite(large):
    ds=large('x,label\n1,car\ninf,car\n2,pedestrian\n')
    assert run_metric('non_finite_rate',ds,context().goal).value==pytest.approx(1/3)
    assert assess(ds,context(expected_types={'x':'number'})).overall_status=='NOT_READY'
    ds2=large('x,label\n1,car\n2,pedestrian\n')
    assert assess(ds2,context(expected_types={'x':'integer'})).overall_status=='READY'

def test_boolean_types_and_allowed_values(large):
    ds=large('x,label\nTrue,car\nFalse,pedestrian\n')
    assert assess(ds,context(expected_types={'x':'boolean'},allowed_values={'x':[True,False]})).overall_status=='READY'
    mixed=large('x,label\n1,car\nbad,pedestrian\n')
    assert assess(mixed,context(allowed_values={'x':[1]})).overall_status=='NOT_READY'

def test_sql_values_and_headers_are_not_executed(large):
    name='x; DROP TABLE records; --'
    ds=large(f'{name},label\n1,car\n2,pedestrian\n')
    assert run_metric('required_class_count',ds,context().goal,{'class':"car' OR true --"}).value==0
    assert assess(ds,context(required_inputs=[name])).overall_status=='READY'

def test_coverage_unknown_and_exact_late_scenarios(large):
    ds=large('x,label,lighting\n1,car,daytime\n2,pedestrian,nighttime\n')
    scenario=ScenarioRequirement(name='Night pedestrian',conditions={'lighting':['nighttime']},required_class='pedestrian')
    assert assess(ds,context(scenarios=[scenario])).overall_status=='READY'
    unknown=ScenarioRequirement(name='Altitude',conditions={'altitude_meters':{'min':20,'max':80}})
    assert assess(ds,context(scenarios=[unknown])).overall_status=='INSUFFICIENT_EVIDENCE'

def test_group_and_content_overlap(large):
    ds=large('x,label,split,flight\n1,car,train,A\n1,car,test,A\n')
    goal=context(split_column='split',group_column='flight').goal
    assert run_metric('split_overlap',ds,goal).value==1
    assert run_metric('group_leakage',ds,goal).value==1

def test_kappa_and_verified_errors(large):
    ds=large('x,label,second,reference\n1,car,car,car\n2,pedestrian,pedestrian,car\n',{'independent_annotation_column':'second','trusted_reference_labels':True})
    goal=context(reference_columns={'label':'reference'}).goal
    assert run_metric('label_agreement',ds,goal).value==1
    assert run_metric('verified_label_error',ds,goal).value==.5
    assert 'not correctness' in run_metric('label_agreement',ds,goal).reason

def test_unsupported_large_metric_never_uses_preview(large):
    ds=large('x,label\n1,car\n2,pedestrian\n')
    r=Requirement(requirement_id='reference',description='Full reference comparison',metric_id='reference_comparison')
    assert assess(ds,context(requirements=[r])).overall_status=='INSUFFICIENT_EVIDENCE'

def test_encoding_and_malformed_csv(large):
    ds=large('x,label\n1,witness\u2019s account\n'.encode('cp1252'))
    assert ds.profile['parsing']['encoding']=='cp1252'
    with pytest.raises(IngestionError): large('x,label\n1,car,extra\n')
    with pytest.raises(IngestionError): large('x,label\n1,a\x00b\n')

def test_stream_reads_are_bounded(large):
    class Guarded(io.BytesIO):
        def read(self,size=-1):
            assert 0<size<=1024**2
            return super().read(size)
    ds=load_dataset_stream('test.csv',Guarded(b'x,label\n1,car\n'))
    try: assert ds.profile['row_count']==1
    finally: ds.close()

def test_stream_size_limit_and_cleanup(monkeypatch,tmp_path):
    import backend.large_ingestion as large_ingestion
    monkeypatch.setattr(large_ingestion,'ROOT',tmp_path)
    monkeypatch.setenv('MAX_UPLOAD_MB','0')
    with pytest.raises(IngestionError,match='size limit'):
        load_dataset_stream('x.csv',io.BytesIO(b'x,label\n1,car\n'))
    assert not list(tmp_path.iterdir())

def test_disk_deletion_removes_owned_directory(large):
    ds=large('x,label\n1,car\n')
    directory=Path(ds.disk_path).parent
    ds.close()
    assert not directory.exists()

def test_disk_parquet(large):
    frame=pd.DataFrame({'x':[1,2,None],'label':['car','pedestrian','car']})
    buf=io.BytesIO();frame.to_parquet(buf,index=False)
    ds=large(buf.getvalue(),name='data.parquet')
    assert ds.profile['row_count']==3
    assert ds.profile['missing_count']['x']==1
    assert run_metric('required_class_count',ds,context().goal,{'class':'car'}).value==2

def test_disk_api_and_limits(large):
    client=TestClient(app)
    assert client.get('/api/config').json()['max_upload_bytes']==2*1024**3
    response=client.post('/api/datasets/upload',files={'file':('x.csv',b'x,label\n1,car\n2,pedestrian\n')})
    assert response.status_code==200,response.text
    id=response.json()['id']
    try:
        assert response.json()['profile']['storage_backend']=='duckdb'
        assert client.get(f'/api/datasets/{id}/profile').json()['row_count']==2
    finally: assert client.delete('/api/datasets/'+id).status_code==200

def test_size_policy_does_not_raise_image_limits(large,monkeypatch):
    monkeypatch.setenv('MAX_IMAGE_UPLOAD_MB','0')
    with pytest.raises(IngestionError):load_dataset_stream('image.zip',io.BytesIO(b'not zip'))

def test_repeatable_disk_metrics(large):
    a=large('x,label\n1,car\n2,pedestrian\n')
    b=large('x,label\n1,car\n2,pedestrian\n')
    assert assess(a,context()).model_dump()==assess(b,context()).model_dump()
