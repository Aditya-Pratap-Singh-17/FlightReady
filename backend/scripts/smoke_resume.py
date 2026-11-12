"""Verify real epoch-checkpoint cancellation/resume on the isolated smoke backend."""
import argparse
import json
from pathlib import Path
import sys
import time
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend.workbench.contracts import TrainRequest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8049');args=parser.parse_args()
    with httpx.Client(base_url=args.url+'/api/v1',timeout=60) as client:
        def get(path):
            response=client.get(path);response.raise_for_status();return response.json()
        def post(path,body=None):
            response=client.post(path,json=body);response.raise_for_status();return response.json()
        model=next(m for m in get('/models') if m['payload']['configuration']['model_type']=='candidate')
        config={k:v for k,v in model['payload']['configuration'].items() if k in TrainRequest.model_fields}
        config.update(epochs=12,batch=2,image_size=64,device='cpu',experiment_group='Synthetic checkpoint-resume verification')
        source=post('/training',config)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            job=get('/jobs/'+source['id'])['payload']
            epoch=job.get('progress',{}).get('epoch',0)
            if job['status']=='running' and 1<=epoch<12:
                post('/jobs/'+source['id']+'/cancel');print('Cancelled after completed epoch '+str(epoch),flush=True);break
            if job['status'] not in ('running','queued'):raise RuntimeError('Training ended before cancellation: '+str(job.get('error')))
            time.sleep(.05)
        else:raise TimeoutError('No completed epoch checkpoint appeared')
        resumed=post('/jobs/'+source['id']+'/resume')
        while time.monotonic()<deadline:
            job=get('/jobs/'+resumed['id'])['payload']
            if job.get('progress',{}).get('epoch',0)>12:raise AssertionError('Progress exceeds actual epoch budget')
            if job['status']=='completed':
                output=Path('.runtime/resume-verification.json')
                output.write_text(json.dumps({'status':'passed','source_job_id':source['id'],'resumed_job_id':resumed['id'],'model_id':job['result']['model_id'],'epochs':12},indent=2))
                print(output.read_text(),flush=True);return
            if job['status'] not in ('running','queued'):
                print(get('/jobs/'+resumed['id']+'/logs')['text']);raise RuntimeError(job.get('error'))
            time.sleep(.2)
        raise TimeoutError('Resume did not complete')


if __name__=='__main__':main()
