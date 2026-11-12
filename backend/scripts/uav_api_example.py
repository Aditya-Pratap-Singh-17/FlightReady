"""Two-stage localhost API example that preserves explicit human confirmation.

prepare uploads and writes an unconfirmed context. Edit/review that file, set
confirmed=true, clear resolved questions, then run assess with the same state.
"""
import argparse
import json
from pathlib import Path
import time
import httpx

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','assess'])
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--directory',type=Path,default=Path('uav-api-example'))
    args=parser.parse_args();directory=args.directory
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=120) as client:
        if args.action=='prepare':
            if not args.dataset:parser.error('--dataset is required for prepare')
            directory.mkdir(parents=True,exist_ok=False)
            with args.dataset.open('rb') as source:
                upload=client.post('/api/datasets/upload',files={'file':(args.dataset.name,source)})
            upload.raise_for_status()
            if upload.json()['profile'].get('needs_selection'):raise ValueError('Choose a supported detection layout in the application first')
            response=client.post('/api/context/interpret',json={'text':'Detect cars in UAV images','tags':{'Task':['Detect objects'],'Data type':['Images']}})
            response.raise_for_status();interpretation=response.json()
            (directory/'state.json').write_text(json.dumps({'dataset_id':upload.json()['id'],'interpretation_id':interpretation['id']},indent=2))
            (directory/'context.json').write_text(json.dumps(interpretation['context'],indent=2))
            print('Review context.json, match the class names, resolve questions, and explicitly set confirmed=true before assess.')
        else:
            context=json.loads((directory/'context.json').read_text());state=json.loads((directory/'state.json').read_text())
            if context.get('confirmed') is not True or context.get('clarification_questions'):raise ValueError('Explicitly confirm the reviewed context and resolve its questions first')
            response=client.post('/api/assessments',json={**state,'context':context});response.raise_for_status();job=response.json()['id']
            deadline=time.monotonic()+3600
            while time.monotonic()<deadline:
                progress=client.get(f'/api/assessments/{job}/progress');progress.raise_for_status();status=progress.json()['status']
                if status in ('completed','assessment_error'):break
                if status in ('error','cancelled'):raise RuntimeError('Assessment did not produce a report: '+status)
                time.sleep(.5)
            else:raise TimeoutError('Assessment still running; retrieve it from the application using '+job)
            for format in ('json','html'):
                report=client.get(f'/api/assessments/{job}/export',params={'format':format});report.raise_for_status()
                with (directory/('report.'+format)).open('xb') as destination:destination.write(report.content)
            print('Saved JSON and HTML reports in '+str(directory))

if __name__=='__main__':main()
