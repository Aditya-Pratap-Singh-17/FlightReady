"""Optional real 2 GiB HTTP upload test using an explicitly synthetic CSV.

This writes a large temporary file, starts its own hidden local backend, streams
the file over HTTP, checks exact full-data metrics, and removes its own data.
Use only when there is sufficient disk space. This is a software capacity check,
not evidence of real UAV mission readiness.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import httpx
import psutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def validate(mebibytes=2048):
    target=mebibytes*1024**2
    if target<1024**2: raise ValueError('Use at least 1 MiB')
    work=ROOT/'.runtime'
    work.mkdir(exist_ok=True)
    if shutil.disk_usage(work).free<target*4:
        raise RuntimeError('This test needs at least four times the generated file size in free disk space')
    with TemporaryDirectory(prefix='capacity-check-',dir=work) as directory:
        path=Path(directory)/'SYNTHETIC-capacity.csv'
        header=b'x,label,payload\n'
        row=b'1,car,'+b's'*1017+b'\n'
        assert len(row)==1024
        whole,remainder=divmod(target-len(header),len(row))
        with path.open('wb') as output:
            output.write(header)
            block=row*1024
            blocks,tail=divmod(whole,1024)
            for _ in range(blocks):output.write(block)
            output.write(row*tail)
            final=b'2,pedestrian,'
            if remainder<len(final)+1:
                raise ValueError('Chosen byte size cannot hold the final synthetic record')
            output.write(final+b'z'*(remainder-len(final)-1)+b'\n')
        assert path.stat().st_size==target
        expected_rows=whole+1
        print(f'Generated {target:,} bytes; {expected_rows:,} rows. All data is synthetic.',flush=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=os.environ.copy()
        env['MAX_UPLOAD_MB']=str(max(mebibytes,2048))
        env['LARGE_DATASET_THRESHOLD_MB']='16'
        env['DATASET_MEMORY_MB']='256'
        log_path=Path(directory)/'server.log'
        with log_path.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=log,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            stop=threading.Event();peak=[0];dataset_id=None
            def observe():
                monitored=psutil.Process(process.pid)
                while not stop.wait(.2):
                    try:peak[0]=max(peak[0],monitored.memory_info().rss)
                    except psutil.Error:return
            monitor=threading.Thread(target=observe,daemon=True);monitor.start()
            client=httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=httpx.Timeout(1800,connect=10))
            started=time.monotonic()
            try:
                for _ in range(100):
                    try:
                        if client.get('/api/health').status_code==200:break
                    except httpx.TransportError:pass
                    if process.poll() is not None:raise RuntimeError(log_path.read_text())
                    time.sleep(.1)
                else:raise RuntimeError('Local test server did not start')
                boundary='capacity-test-boundary'
                prefix=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="SYNTHETIC-capacity.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()
                suffix=f'\r\n--{boundary}--\r\n'.encode()
                def body():
                    yield prefix
                    with path.open('rb') as source:
                        while block:=source.read(1024**2):yield block
                    yield suffix
                print('Streaming upload over HTTP, then ingesting and profiling...',flush=True)
                response=client.post('/api/datasets/upload',content=body(),headers={'Content-Type':f'multipart/form-data; boundary={boundary}',
                    'Content-Length':str(len(prefix)+target+len(suffix))})
                if response.status_code!=200:raise RuntimeError(response.text+'\n'+log_path.read_text())
                uploaded=response.json();dataset_id=uploaded['id']
                assert uploaded['profile']['row_count']==expected_rows
                assert uploaded['profile']['source_bytes']==target
                assert uploaded['profile']['storage_backend']=='duckdb'
                print('Full dataset ingested; running exact readiness checks...',flush=True)
                from backend.context import DatasetContext,Goal
                context=DatasetContext(original_text='SYNTHETIC capacity test only',goal=Goal(task_type='classification',prediction_target='label',required_inputs=['x'],required_classes=['pedestrian']),confirmed=True)
                response=client.post('/api/assessments',json={'dataset_id':dataset_id,'context':context.model_dump()})
                response.raise_for_status();job_id=response.json()['id']
                while True:
                    progress=client.get(f'/api/assessments/{job_id}/progress').json()
                    if progress['status'] not in ('queued','running'):break
                    time.sleep(.5)
                cert=client.get(f'/api/assessments/{job_id}/certificate')
                cert.raise_for_status();certificate=cert.json()
                assert certificate['overall_status']=='READY',certificate['executive_summary']
                counts=next(m for m in certificate['metrics'] if m['metric_id']=='class_distribution')['value']
                assert counts=={'car':whole,'pedestrian':1}
                assert certificate['dataset']['profile']['duplicate_rows']==whole-1
                result={'synthetic':True,'input_bytes':target,'data_rows':expected_rows,'full_class_counts':counts,
                    'readiness_for_synthetic_test_goal':certificate['overall_status'],
                    'elapsed_seconds':round(time.monotonic()-started,2),'peak_server_rss_bytes':peak[0],
                    'database_memory_setting_mb':256,'transport':'real HTTP multipart with bounded client reads'}
                (work/'large-upload-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(json.dumps(result,indent=2),flush=True)
            finally:
                if dataset_id:
                    try:client.delete('/api/datasets/'+dataset_id)
                    except httpx.TransportError:pass
                client.close();stop.set();monitor.join(timeout=2)
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib',type=int,default=2048)
    validate(parser.parse_args().mib)
