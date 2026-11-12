"""Real API -> subprocess -> YOLO CPU/GPU smoke; explicitly synthetic data only."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','0','auto'],default='cpu')
    parser.add_argument('--output',default=None)
    parser.add_argument('--weights',default=None,help='Trusted local .pt file for an actual fine-tuning smoke')
    args=parser.parse_args()
    directory=Path(args.output or '.runtime/smoke-'+uuid.uuid4().hex[:8]).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    os.environ['WORKBENCH_ROOT']=str(directory/'storage')
    from fastapi.testclient import TestClient
    from main import app
    from backend.workbench.api import services
    from backend.workbench.synthetic import detection_fixture
    from backend.context import DatasetContext,Goal
    services.cache_clear()
    def post(client,path,body):
        response=client.post('/api/v1'+path,json=body)
        if response.status_code>=400:
            raise RuntimeError(response.text)
        return response.json()
    def wait(client,job):
        deadline=time.monotonic()+600
        last=None
        while time.monotonic()<deadline:
            response=client.get('/api/v1/jobs/'+job['id'])
            response.raise_for_status()
            record=response.json();payload=record['payload']
            if payload['stage']!=last:
                last=payload['stage'];print(payload['task']+': '+last,flush=True)
            if payload['status']=='completed':
                return record
            if payload['status'] not in ('queued','running'):
                print(client.get('/api/v1/jobs/'+job['id']+'/logs').json()['text'])
                raise RuntimeError(str(payload))
            time.sleep(.3)
        raise TimeoutError('Smoke job exceeded ten minutes')
    with TestClient(app) as client:
        datasets=[];splits=[];readiness=[];models=[]
        for index,group in enumerate(['candidate','reference']):
            source=detection_fixture(directory/(group+'.zip'),seed=index+1)
            with source.open('rb') as incoming:
                response=client.post('/api/v1/datasets/import',files={'file':(source.name,incoming,'application/zip')})
            response.raise_for_status();record=response.json();datasets.append(record)
            samples=client.get('/api/v1/datasets/'+record['id']+'/samples').json()
            assert len(samples['samples'])==10
            image=client.get(samples['samples'][0]['url']);assert image.status_code==200
            interpreted=post(client,'/context/interpret',{'text':'Detect drone objects in explicitly synthetic software-test images.','tags':{'Task':['Detect objects'],'Data type':['Images']}})
            context=DatasetContext.model_validate(interpreted['context'])
            context.goal=Goal(task_type='object_detection',required_classes=['drone'],input_modalities=['image'])
            context.confirmed=True;context.clarification_questions=[]
            split=post(client,'/datasets/'+record['id']+'/splits',{'group_column':'flight_id','seed':42})
            splits.append(split)
            ready=wait(client,post(client,'/readiness',{'dataset_id':record['id'],'split_id':split['id'],'context':context.model_dump()}))
            readiness.append(ready)
            assert ready['payload']['result']['dataset']['profile']['image_count']==6
            train=wait(client,post(client,'/training',{'dataset_id':record['id'],'split_id':split['id'],'readiness_id':ready['id'],
                'model_type':group,'epochs':1,'batch':2,'image_size':64,'seed':42,'device':args.device,
                'initialization':'pretrained' if args.weights else 'scratch','weights_path':str(Path(args.weights).resolve()) if args.weights else None}))
            models.append(train['payload']['result']['model_id'])
        source=detection_fixture(directory/'external-test.zip',seed=91,external=True)
        with source.open('rb') as incoming:
            response=client.post('/api/v1/datasets/import',files={'file':(source.name,incoming,'application/zip')})
        response.raise_for_status();test=response.json()
        evaluation=wait(client,post(client,'/evaluations',{'model_ids':models,'dataset_id':test['id'],'device':args.device}))
        ids=evaluation['payload']['result']['evaluation_ids']
        compared=post(client,'/comparisons',ids)
        assert compared['payload']['matched_conditions'],compared
        for id in ids:
            predictions=client.get('/api/v1/evaluations/'+id+'/predictions').json()
            assert predictions['total']==4
        research=client.get('/api/v1/research').json()
        assert len(research['rows'])==2
        assert all(c['pearson'] is None for c in research['correlations'])
        for format in ['json','csv']:
            response=client.get('/api/v1/research/export?format='+format)
            response.raise_for_status();(directory/('research.'+format)).write_bytes(response.content)
        result={'scope':'Explicitly synthetic software smoke, not evidence of UAV detector generalization','device':args.device,
                'model_ids':models,'evaluation_ids':ids,'comparison_id':compared['id'],'research_rows':len(research['rows']),
                'metrics':[r['metrics'] for r in research['rows']],'status':'passed'}
        (directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));print('Evidence: '+str(directory),flush=True)


if __name__=='__main__':
    main()
