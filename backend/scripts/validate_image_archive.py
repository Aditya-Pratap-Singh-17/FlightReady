"""Verify a real image ZIP through HTTP, assessment and exports without altering it.

The verification goal checks car/pedestrian annotation availability only; it is
not a claim that the dataset meets the user's intended deployment conditions.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def validate(source: Path):
    source=source.resolve(strict=True)
    work=ROOT/'.runtime';work.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='image-check-',dir=work) as directory:
        log_path=Path(directory)/'server.log'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=os.environ.copy()
        env.update(MAX_UPLOAD_MB='2048',MAX_IMAGE_UPLOAD_MB='2048',MAX_ZIP_MB='4096',MAX_ZIP_FILES='50000')
        with log_path.open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port),'--log-level','info'],cwd=ROOT,env=env,stdout=log,stderr=log,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            id=None
            start=time.monotonic()
            client=httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=httpx.Timeout(1800,connect=10))
            try:
                for _ in range(100):
                    try:
                        if client.get('/api/health').status_code==200:break
                    except httpx.TransportError:pass
                    if process.poll() is not None:raise RuntimeError(log_path.read_text())
                    time.sleep(.1)
                else:raise RuntimeError('Image verification server failed to start')
                boundary='image-verification-boundary'
                prefix=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{source.name}"\r\nContent-Type: application/zip\r\n\r\n').encode()
                suffix=f'\r\n--{boundary}--\r\n'.encode()
                def body():
                    yield prefix
                    with source.open('rb') as data:
                        while block:=data.read(1024**2):yield block
                    yield suffix
                print('Uploading real archive and decoding every image...',flush=True)
                response=client.post('/api/datasets/upload',content=body(),headers={'Content-Type':f'multipart/form-data; boundary={boundary}',
                    'Content-Length':str(len(prefix)+source.stat().st_size+len(suffix))})
                if response.status_code!=200:raise RuntimeError(response.text)
                upload=response.json();id=upload['id'];profile=upload['profile']
                print(f'Validated {profile["image_count"]} images and {profile["object_count"]} target boxes; running assessment...',flush=True)
                from backend.context import DatasetContext,Goal
                context=DatasetContext(original_text='Software verification only: check car and pedestrian annotations. No deployment environment asserted.',
                    goal=Goal(task_type='object_detection',prediction_target='bounding boxes and classes',input_modalities=['image'],required_classes=['car','pedestrian']),confirmed=True)
                response=client.post('/api/assessments',json={'dataset_id':id,'context':context.model_dump()})
                response.raise_for_status();job=response.json()['id']
                for _ in range(3600):
                    progress=client.get(f'/api/assessments/{job}/progress').json()
                    if progress['status'] not in ('queued','running'):break
                    time.sleep(.5)
                if progress['status'] not in ('completed','assessment_error'):raise RuntimeError(str(progress))
                response=client.get(f'/api/assessments/{job}/certificate');response.raise_for_status()
                certificate=response.json()
                assert not [m for m in certificate['metrics'] if m['state']=='error'],certificate['executive_summary']
                for format in ['json','html']:
                    exported=client.get(f'/api/assessments/{job}/export',params={'format':format})
                    exported.raise_for_status()
                summary={'input_name':source.name,'input_bytes':source.stat().st_size,'sha256':upload['sha256'],
                    'images':profile['image_count'],'annotation_format':profile['annotation_format'],'target_objects':profile['object_count'],
                    'class_counts':profile['class_counts'],'annotation_summary':profile['annotation_summary'],'issues':profile['issues'],
                    'unreadable_images':int(round(next(m for m in certificate['metrics'] if m['metric_id']=='unreadable_rate')['value']*profile['image_count'])),
                    'verification_goal_only':True,'status_for_verification_goal':certificate['overall_status'],
                    'failed_requirements':[f['reason'] for f in certificate['findings'] if f['state']=='fail'],
                    'json_and_html_exports':'passed','elapsed_seconds':round(time.monotonic()-start,2)}
                (work/'visdrone-validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
                print(json.dumps(summary,indent=2),flush=True)
            except Exception:
                print(log_path.read_text(encoding='utf-8'),flush=True)
                raise
            finally:
                if id:
                    try:client.delete('/api/datasets/'+id)
                    except httpx.TransportError:pass
                client.close();process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    validate(parser.parse_args().archive)
