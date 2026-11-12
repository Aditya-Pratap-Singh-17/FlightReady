"""Local single-user API. A bounded executor runs real, cancellable assessments.

Large tables use local temporary databases; job records remain in memory. Nothing uploaded is
executed or sent to the language-model provider. Bind only to loopback in V1.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock, Event
from typing import Literal
import json
import logging
import os
import time
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from backend.context import Model, DatasetContext, Certificate
from backend.context_extractor import interpret
from backend.ingestion import IngestionError
from backend.large_ingestion import load_dataset_stream
from backend.readiness_engine import assess
from backend.reports import html_report

if os.getenv('PYTHON_DOTENV_DISABLED') != '1':
    load_dotenv(Path(__file__).resolve().parent / '.env')
log = logging.getLogger(__name__)
datasets, interpretations, jobs = {}, {}, {}
lock = RLock()
executor = ThreadPoolExecutor(max_workers=2,thread_name_prefix='readiness')
TTL = 3600

@asynccontextmanager
async def lifespan(app):
    from backend.workbench.api import services
    services()[1].recover()
    yield
    services()[1].close()
    executor.shutdown(wait=True,cancel_futures=True)
    for entry in datasets.values(): entry['dataset'].close()
    datasets.clear()
    interpretations.clear()
    jobs.clear()

app = FastAPI(title='Context-Aware UAV Dataset Readiness',version='1.4.0',lifespan=lifespan)

class BodyLimitMiddleware:
    """Count bytes as they stream; never accumulate the request in RAM."""
    def __init__(self,app): self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method'] not in ('POST','PUT','PATCH'):
            return await self.app(scope,receive,send)
        upload_paths=('/api/datasets/upload','/api/v1/datasets/import','/api/v1/datasets/directory-upload')
        overhead=16*1024**2 if scope['path']=='/api/v1/datasets/directory-upload' else 128*1024
        limit = int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2 + overhead if scope['path'] in upload_paths else 1024**2
        headers = dict(scope.get('headers',[]))
        try: length=int(headers.get(b'content-length',b'0'))
        except ValueError: length=limit+1
        if length>limit:
            return await JSONResponse({'detail':'Request exceeds upload size limit'},status_code=413)(scope,receive,send)
        received=0
        async def bounded_receive():
            nonlocal received
            message=await receive()
            received+=len(message.get('body',b''))
            if received>limit:
                raise HTTPException(413,'Request exceeds upload size limit')
            return message
        await self.app(scope,bounded_receive,send)

app.add_middleware(BodyLimitMiddleware)

class InterpretRequest(Model):
    text: str = Field(default='',max_length=12000)
    tags: dict[str,list[str]] = Field(default_factory=dict)
    enable_remote: bool = False

class InterpretationResponse(Model):
    id: str
    context: DatasetContext

class UploadResponse(Model):
    id: str
    name: str
    sha256: str
    modality: str
    profile: dict

class AssessmentRequest(Model):
    dataset_id: str
    context: DatasetContext
    interpretation_id: str | None = None
    reference_dataset_id: str | None = None

class DatasetSelection(Model):
    options: dict = Field(default_factory=dict)

class JobResponse(Model):
    id: str
    status: str
    stages: list[str]
    error: str | None = None

def cleanup():
    # Cleanup only completed jobs; running jobs keep their dataset references.
    now = time.monotonic()
    with lock:
        active={id for job in jobs.values() if job['status'] in ('queued','running') for id in job.get('dataset_ids',[])}
        for store in (datasets,interpretations,jobs):
            for key,entry in list(store.items()):
                if now-entry['created'] > TTL and not entry.get('processing') and entry.get('status') not in ('queued','running') and not (store is datasets and key in active):
                    if store is datasets: entry['dataset'].close()
                    del store[key]

def get_dataset(id):
    with lock:
        item = datasets.get(id)
    if not item: raise HTTPException(404,'Dataset not found or expired; upload it again')
    return item['dataset']

def get_job(id):
    with lock:
        item = jobs.get(id)
    if not item: raise HTTPException(404,'Assessment not found or expired')
    return item

@app.post('/api/context/interpret',response_model=InterpretationResponse)
def context_interpret(body:InterpretRequest):
    cleanup()
    if sum(len(v) for v in body.tags.values()) > 100: raise HTTPException(422,'Too many tags')
    try: context = interpret(body.text,body.tags,allow_remote=body.enable_remote)
    except ValueError as e: raise HTTPException(422,str(e)) from e
    id = uuid.uuid4().hex
    with lock: interpretations[id] = {'context':context.model_copy(deep=True),'created':time.monotonic()}
    return {'id':id,'context':context}

@app.post('/api/datasets/upload',response_model=UploadResponse)
def upload(file:UploadFile=File(...),metadata:str=Form('{}')):
    cleanup()
    with lock:
        if len(datasets) >= 10: raise HTTPException(429,'Ten-dataset session limit reached; delete an upload or restart')
    try:
        declared = json.loads(metadata)
        if not isinstance(declared,dict) or len(metadata)>100_000: raise ValueError()
        json.dumps(declared,allow_nan=False)
        # The multipart parser spools large files to disk. This loader reads
        # 1 MiB at a time, then ingests large tables in bounded Arrow batches.
        ds = load_dataset_stream(file.filename or '',file.file,declared)
    except (IngestionError,ValueError) as e:
        raise HTTPException(422,str(e) or 'Metadata must be a JSON object') from e
    finally:
        file.file.close()
    id = uuid.uuid4().hex
    with lock: datasets[id] = {'dataset':ds,'created':time.monotonic()}
    return {'id':id,'name':ds.name,'sha256':ds.sha256,'modality':ds.modality,'profile':ds.profile}

@app.get('/api/datasets/{id}/profile',response_model=dict)
def profile(id:str):
    return get_dataset(id).profile

@app.post('/api/datasets/{id}/select',response_model=UploadResponse)
def select_dataset(id:str,body:DatasetSelection):
    ds=get_dataset(id)
    if not ds.source_path or not ds.profile.get('needs_selection'):
        raise HTTPException(409,'This dataset has no pending layout selection')
    allowed={'format','dataset_member','annotation_format','annotation_file','image_root','class_file','class_names','record_path','sheet_name','has_header','delimiter','column_names','text_unit','voc_coordinate_base','collection_type'}
    if set(body.options)-allowed or len(json.dumps(body.options))>100_000:
        raise HTTPException(422,'Unsupported or oversized dataset selection options')
    with lock:
        if datasets[id].get('processing'):raise HTTPException(409,'This selection is already being processed')
        datasets[id]['processing']=True
    try:
        selected_metadata={**ds.metadata,**body.options}
        selected_metadata={key:value for key,value in selected_metadata.items() if value is not None}
        with Path(ds.source_path).open('rb') as source:
            loaded=load_dataset_stream(ds.name,source,selected_metadata)
        with lock:
            datasets[id]={'dataset':loaded,'created':time.monotonic()}
            ds.close()
        return {'id':id,'name':loaded.name,'sha256':loaded.sha256,'modality':loaded.modality,'profile':loaded.profile}
    except (IngestionError,ValueError) as exc:
        raise HTTPException(422,str(exc)) from exc
    finally:
        with lock:
            if id in datasets:datasets[id].pop('processing',None)

@app.delete('/api/datasets/{id}')
def delete_dataset(id:str):
    ds=get_dataset(id)
    with lock:
        if datasets[id].get('processing'):raise HTTPException(409,'Dataset selection is still being processed')
        if any(id in job.get('dataset_ids',[]) and job['status'] in ('queued','running') for job in jobs.values()):
            raise HTTPException(409,'Dataset is in use by an assessment; cancel or finish the job before deleting it')
        datasets.pop(id,None)
        ds.close()
    return {'deleted':True}

class Cancelled(Exception): pass

def perform(id,ds,context,initial,reference):
    job = get_job(id)
    def progress(stage):
        with lock:
            if job['cancel'].is_set(): raise Cancelled()
            job['stages'].append(stage)
    try:
        with lock: job['status'] = 'running'
        progress('Requirements confirmed')
        cert = assess(ds,context,initial,reference,progress)
        progress('Certificate ready')
        with lock:
            job['certificate'] = cert
            job['status'] = 'assessment_error' if cert.overall_status == 'ASSESSMENT_ERROR' else 'completed'
    except Cancelled:
        with lock: job['status'] = 'cancelled'
    except Exception:
        log.exception('Assessment failed')
        with lock:
            job['status'],job['error'] = 'error','Assessment failed; inspect the local server log and retry'

@app.post('/api/assessments',response_model=JobResponse,status_code=202)
def start(body:AssessmentRequest):
    cleanup()
    ds = get_dataset(body.dataset_id)
    if ds.profile.get('needs_selection'):raise HTTPException(409,'Choose the dataset file, sheet or record layout before assessment')
    reference = get_dataset(body.reference_dataset_id) if body.reference_dataset_id else None
    if not body.context.confirmed or body.context.clarification_questions:
        raise HTTPException(422,'Confirm requirements and resolve clarification questions first')
    if body.context.goal.task_type == 'unspecified': raise HTTPException(422,'Choose a task type before confirmation')
    initial = None
    if body.interpretation_id:
        with lock: record = interpretations.get(body.interpretation_id)
        if not record: raise HTTPException(404,'Initial interpretation not found; interpret the context again')
        initial = record['context'].model_copy(deep=True)
    with lock:
        if sum(j['status'] in ('queued','running') for j in jobs.values())>=4:
            raise HTTPException(429,'Assessment queue is full; retry after a job completes')
        id = uuid.uuid4().hex
        jobs[id] = {'id':id,'created':time.monotonic(),'status':'queued','stages':['Dataset uploaded','Dataset validated','Context interpreted' if initial else 'Manual context supplied'], 'error':None,'cancel':Event(),'certificate':None,
                    'dataset_ids':[body.dataset_id]+([body.reference_dataset_id] if body.reference_dataset_id else [])}
        executor.submit(perform,id,ds,body.context.model_copy(deep=True),initial,reference)
    return job_response(id)

def job_response(id):
    job = get_job(id)
    with lock: return {k:list(job[k]) if k=='stages' else job[k] for k in ('id','status','stages','error')}

@app.get('/api/assessments/{id}',response_model=JobResponse)
@app.get('/api/assessments/{id}/progress',response_model=JobResponse)
def progress(id:str): return job_response(id)

@app.post('/api/assessments/{id}/cancel')
def cancel(id:str):
    job = get_job(id)
    if job['status'] in ('queued','running'): job['cancel'].set()
    return {'cancellation_requested':True}

@app.get('/api/assessments/{id}/certificate',response_model=Certificate)
def certificate(id:str):
    job = get_job(id)
    if job['status'] not in ('completed','assessment_error'): raise HTTPException(409,'Certificate is not available until the assessment completes')
    return job['certificate']

@app.get('/api/assessments/{id}/export')
def export(id:str,format:Literal['json','html']=Query('json')):
    cert = certificate(id)
    headers = {'Content-Disposition':f'attachment; filename="readiness-{id}.{format}"'}
    return HTMLResponse(html_report(cert),headers=headers) if format=='html' else JSONResponse(cert.model_dump(mode='json'),headers=headers)

@app.get('/api/health')
def health(): return {'status':'ok','version':'1.4.0'}

@app.get('/api/config')
def config():
    return {'max_upload_bytes':int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2,
            'max_image_upload_bytes':int(os.getenv('MAX_IMAGE_UPLOAD_MB','2048'))*1024**2}

@app.get('/api/metrics')
def metric_registry():
    from metrics.core import REGISTRY
    return {mid:definition.model_dump(mode='json') for mid,definition in REGISTRY.items()}

# A production frontend build is served by the Python process for local use.
from backend.workbench.api import router as workbench_router
app.include_router(workbench_router)
from backend.workbench.quality_api import router as quality_router
app.include_router(quality_router)
from backend.workbench.lab_api import router as lab_router
app.include_router(lab_router)
from backend.workbench.workflow_api import router as workflow_router
app.include_router(workflow_router)
from backend.workbench.research_api import router as research_router
app.include_router(research_router)
dist = Path(__file__).resolve().parent/'frontend'/'dist'
if dist.exists(): app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
