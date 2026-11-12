"""Versioned API for durable local research workflows."""
from functools import lru_cache
from pathlib import Path, PurePosixPath
import csv
import io
import json
import os
import tempfile
import zipfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.routing import APIRoute
from fastapi.responses import Response, JSONResponse, HTMLResponse, FileResponse
from backend.context import Certificate,DatasetContext
from backend.context_extractor import interpret
from backend.reports import html_report
from backend.workbench.contracts import (ContextRequest, RegisterRequest, RenameRequest, SplitRequest, ReadinessRequest,
                                 TrainRequest, TestRequest, VariantRequest, RecordResponse, ScorePreviewRequest)
from sql.store import Store
from backend.workbench.datasets import import_file, register_directory, load, image_bytes, subset
from backend.workbench.splits import create_split, get_split, assert_independent
from backend.workbench.jobs import JobManager
from backend.workbench.detector import compute, validate_training, classes_for, class_mapping, test_manifest
from backend.workbench.analytics import research, comparison


class ErrorRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request):
            try:
                if request.url.path=='/api/v1/datasets/directory-upload':
                    # Preparse using the same documented file-count limit rather
                    # than Starlette's unrelated 1,000-file default. FastAPI reuses
                    # the cached form when validating the typed endpoint below.
                    await request.form(max_files=int(os.getenv('MAX_ZIP_FILES','50000')),max_fields=10,max_part_size=16*1024**2)
                return await original(request)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except (ValueError, OSError, zipfile.BadZipFile) as exc:
                raise HTTPException(422, str(exc)) from exc
        return handler


@lru_cache(maxsize=1)
def services():
    store = Store()
    return store, JobManager(store)


router = APIRouter(prefix='/api/v1', tags=['Research workbench'], route_class=ErrorRoute)


@lru_cache(maxsize=1)
def cached_compute():
    return compute()


@router.get('/compute', response_model=dict)
def compute_status():
    return cached_compute()


@router.get('/overview', response_model=dict)
def overview():
    store, _ = services()
    jobs = store.list('job')
    return {'datasets': len([d for d in store.list('dataset') if not d['payload'].get('archived')]),
            'evaluations': sum(j['payload']['task']=='readiness' and j['payload']['status']=='completed' for j in jobs),
            'training_experiments': sum(j['payload']['task']=='training' for j in jobs), 'comparisons': len(store.list('comparison')),
            'models': len(store.list('model')), 'compute': cached_compute(), 'recent_activity': jobs[:8]}


@router.post('/datasets/import', response_model=RecordResponse, status_code=201)
def upload(file: UploadFile=File(...), metadata: str=Form('{}')):
    declared = json.loads(metadata)
    if not isinstance(declared, dict) or len(metadata)>100000:
        raise ValueError('Metadata must be a JSON object under 100 KB')
    try:
        return import_file(services()[0], file.filename or '', file.file, declared)
    finally:
        file.file.close()


@router.post('/datasets/directory-upload', response_model=RecordResponse, status_code=201)
def directory_upload(files: list[UploadFile]=File(...), paths: str=Form(...)):
    members = json.loads(paths)
    if not isinstance(members,list) or len(members)!=len(files) or len(files)>int(os.getenv('MAX_ZIP_FILES','50000')):
        raise ValueError('Invalid directory file manifest')
    names = set()
    total = 0
    try:
        with tempfile.TemporaryFile() as staged:
            with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_STORED) as archive:
                for file, name in zip(files,members):
                    if not isinstance(name,str) or '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or name in names:
                        raise ValueError('Unsafe or duplicate relative upload path')
                    names.add(name)
                    with archive.open(name,'w') as output:
                        while block:=file.file.read(1024**2):
                            total+=len(block)
                            if total>int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2:
                                raise ValueError('Directory upload exceeds configured size limit')
                            output.write(block)
            staged.seek(0)
            return import_file(services()[0], 'directory.zip', staged)
    finally:
        for file in files:
            file.file.close()


@router.post('/datasets/register', response_model=RecordResponse, status_code=201)
def register(body: RegisterRequest):
    return register_directory(services()[0], body.path, body.name, body.metadata)


@router.get('/datasets', response_model=list[RecordResponse])
def datasets():
    return [d for d in services()[0].list('dataset') if not d['payload'].get('archived')]


@router.get('/datasets/{id}', response_model=RecordResponse)
def dataset(id: str):
    return services()[0].get(id,'dataset')


@router.patch('/datasets/{id}', response_model=RecordResponse)
def rename(id: str, body: RenameRequest):
    services()[0].get(id,'dataset')
    return services()[0].update(id,name=body.name)


@router.delete('/datasets/{id}', response_model=dict)
def delete(id: str):
    store,_=services()
    store.get(id,'dataset')
    for job in store.list('job'):
        if job['payload']['status'] in ('queued','running') and id in json.dumps(job['payload']['config']):
            raise HTTPException(409,'Dataset is in use by an active job')
    store.update(id,archived=True)
    return {'deleted': True, 'artifacts_retained_for_provenance': True}


@router.get('/datasets/{id}/samples', response_model=dict)
def samples(id: str, offset: int=Query(0,ge=0), limit: int=Query(12,ge=1,le=100)):
    ds=load(services()[0],id)
    if ds.modality!='image':
        return {'total':len(ds.frame),'samples':json.loads(ds.frame.iloc[offset:offset+limit].to_json(orient='records')),'modality':ds.modality}
    rows=json.loads(ds.frame.iloc[offset:offset+limit].to_json(orient='records'))
    for row in rows:
        row['annotations']=[o for o in ds.objects if o['image']==row['image']]
        row['url']=f'/api/v1/datasets/{id}/image?image='+__import__('urllib.parse',fromlist=['quote']).quote(row['image'])
    return {'total':len(ds.frame),'samples':rows,'modality':'image'}


@router.get('/datasets/{id}/image')
def image(id: str, image: str):
    from PIL import Image
    raw=image_bytes(services()[0],id,image)
    with Image.open(io.BytesIO(raw)) as original:
        thumbnail=original.convert('RGB')
        thumbnail.thumbnail((1280,1280))
        output=io.BytesIO()
        thumbnail.save(output,format='JPEG',quality=88)
    return Response(output.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=300'})


@router.post('/datasets/{id}/splits',response_model=RecordResponse,status_code=201)
def split(id:str,body:SplitRequest):
    return create_split(services()[0],id,body)


@router.get('/splits',response_model=list[RecordResponse])
def splits(dataset_id:str|None=None):
    return [r for r in services()[0].list('split') if not dataset_id or r['payload']['dataset_id']==dataset_id]


@router.post('/context/interpret',response_model=dict)
def context(body:ContextRequest):
    if sum(len(tags) for tags in body.tags.values())>100:
        raise ValueError('Too many tags')
    result=interpret(body.text,body.tags,allow_remote=body.enable_remote)
    saved=services()[0].create('context',result.model_dump(mode='json'))
    return {'id':saved['id'],'context':saved['payload']}


@router.post('/readiness',response_model=RecordResponse,status_code=202)
def readiness(body:ReadinessRequest):
    if not body.context.confirmed or body.context.clarification_questions or body.context.goal.task_type=='unspecified':
        raise ValueError('Review/confirm context, choose a task, and resolve ambiguous requirements first')
    store,manager=services()
    store.get(body.dataset_id,'dataset')
    if body.split_id:
        get_split(store,body.split_id,body.dataset_id)
    store.create('context',body.context.model_dump(mode='json'))
    return manager.enqueue('readiness',body.model_dump(mode='json'))


@router.post('/context/validate',response_model=DatasetContext)
def validate_context(body:DatasetContext):
    reviewed=body.model_copy(update={'confirmed':False})
    services()[0].create('context',reviewed.model_dump(mode='json'))
    return reviewed


@router.post('/training',response_model=RecordResponse,status_code=202)
def training(body:TrainRequest):
    store,manager=services()
    config=body.model_dump()
    validate_training(store,config)
    return manager.enqueue('training',config)


@router.post('/evaluations',response_model=RecordResponse,status_code=202)
def model_evaluation(body:TestRequest):
    store,manager=services()
    config=body.model_dump()
    source_classes=classes_for(store,body.dataset_id)
    manifest,names=test_manifest(store,config)
    for id in body.model_ids:
        model=store.get(id,'model')['payload']
        class_mapping(source_classes,model['configuration']['classes'],body.class_mapping)
        assert_independent(model['development_manifests'],manifest,names,body.acknowledge_missing_groups)
    return manager.enqueue('evaluation',config)


@router.get('/jobs',response_model=list[RecordResponse])
def jobs():
    return services()[0].list('job')


@router.get('/jobs/{id}',response_model=RecordResponse)
def job(id:str):
    return services()[0].get(id,'job')


@router.post('/jobs/{id}/cancel',response_model=RecordResponse)
def cancel(id:str):
    return services()[1].cancel(id)


@router.get('/jobs/{id}/logs',response_model=dict)
def logs(id:str):
    store,_=services()
    store.get(id,'job')
    path=store.directory('jobs',id)/'worker.log'
    if not path.exists():
        return {'text':''}
    with path.open('rb') as source:
        source.seek(max(0,path.stat().st_size-64000))
        return {'text':source.read().decode('utf-8',errors='replace')}


@router.get('/jobs/{id}/curves',response_model=list[dict])
def curves(id:str):
    store,_=services()
    job=store.get(id,'job')['payload']
    path=Path(job.get('run_directory',store.directory('jobs',id)/'run'))/'results.csv'
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as source:
        return [{k.strip():float(v) for k,v in row.items() if v and v.strip()} for row in csv.DictReader(source)]


@router.get('/models',response_model=list[RecordResponse])
def models():
    return services()[0].list('model')


@router.get('/models/{id}/artifact')
def artifact(id:str):
    payload=services()[0].get(id,'model')['payload']
    return FileResponse(payload['artifact'],filename=f'model-{id}.pt',media_type='application/octet-stream')


@router.get('/evaluations',response_model=list[RecordResponse])
def evaluations():
    return services()[0].list('evaluation')


@router.get('/evaluations/{id}/predictions',response_model=dict)
def predictions(id:str,offset:int=Query(0,ge=0),limit:int=Query(1,ge=1,le=20)):
    payload=services()[0].get(id,'evaluation')['payload']
    rows=json.loads(Path(payload['predictions_path']).read_text(encoding='utf-8'))
    return {'total':len(rows),'samples':rows[offset:offset+limit]}


@router.post('/comparisons',response_model=RecordResponse,status_code=201)
def compare(evaluation_ids:list[str]):
    if not 2<=len(evaluation_ids)<=10 or len(set(evaluation_ids))!=len(evaluation_ids):
        raise ValueError('Choose two to ten real evaluation results')
    store,_=services()
    return store.create('comparison',comparison(store,evaluation_ids))


@router.get('/research',response_model=dict)
def analytics(metric:str|None=None,performance:str='mAP50_95',test_sha256:str|None=None,configuration:str|None=None,dataset_id:str|None=None,scenario:str|None=None,experiment_id:str|None=None):
    if performance not in ('mAP50_95','mAP50','precision','recall','f1','recall_small_native','recall_medium_native','recall_large_native','false_positives_per_image'):
        raise ValueError('Unsupported performance metric')
    store,_=services()
    model_ids=set(store.get(experiment_id,'experiment')['payload'].get('model_ids',[])) if experiment_id else None
    return research(store,metric,performance,test_sha256,configuration,dataset_id,scenario,model_ids=model_ids)


@router.post('/variations',response_model=RecordResponse,status_code=201)
def variations(body:VariantRequest):
    from backend.workbench.variations import variation
    return variation(services()[0],body)


def csv_response(rows,filename):
    output=io.StringIO(newline='')
    if rows:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        # Prevent spreadsheet formula interpretation of user-controlled names/text.
        for row in rows:
            encoded={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()}
            writer.writerow({k:"'"+v if isinstance(v,str) and v[:1] in ('=','+','-','@','\t','\r') else v for k,v in encoded.items()})
    return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{filename}.csv"'})


@router.get('/reports/{id}')
def report(id:str,format:str=Query('json',pattern='^(json|csv|html|pdf)$')):
    record=services()[0].get(id)
    p=record['payload']
    canonical_html = format == 'html' and record['kind'] == 'job' and isinstance(p.get('result'), dict) and p['result'].get('final_report')
    if format == 'pdf' or canonical_html or format in ('html','csv') and record['kind'] in ('experiment','evaluation','comparison','analysis','aggregation','benchmark'):
        from backend.workbench.reporting import document, html_document, pdf_document
        doc=document(services()[0],record)
        if format=='pdf':
            return Response(pdf_document(doc),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="report-{id}.pdf"'})
        if format=='html':
            return HTMLResponse(html_document(doc),headers={'Content-Disposition':f'attachment; filename="report-{id}.html"'})
        if record['kind']=='aggregation':
            return csv_response([{'dimension':d['id'],'score':d['score'],'status':d['status'],
                'evidence_coverage':d['evidence_coverage'],'effective_weight':d['effective_weight'],'metrics':d['metrics']} for d in doc['quality_profile']['dimensions']],'aggregation-'+id)
        if record['kind']=='benchmark':
            return csv_response([{'benchmark_id':id,'dataset_sha256':p['dataset_sha256'],'test_sha256':p['test_sha256'],
                                  'metric':name,'value':value} for name,value in p['metrics'].items()],'inference-'+id)
        return csv_response(doc['rows'] or [{'status':doc['status'],'record_id':id}],'experiment-'+id)
    if record['kind']=='job' and p['task']=='readiness' and p['status']=='completed':
        if format=='html':
            return HTMLResponse(html_report(Certificate.model_validate(p['result'])),headers={'Content-Disposition':f'attachment; filename="readiness-{id}.html"'})
        if format=='csv':
            rows=list(p['result']['metrics'])
            quality=p['result'].get('quality_profile')
            if quality:
                keys=list(rows[0]) if rows else ['metric_id','dimension','state','value','reason','evidence']
                for dimension in quality['dimensions']:
                    row={key:None for key in keys}
                    row.update(metric_id='DIMENSION_'+dimension['id'],dimension=dimension['id'],state=dimension['status'],
                        value=dimension['score'],reason='Adaptive criterion score; inspect evidence coverage and frozen weights',evidence=dimension)
                    rows.append(row)
            score=p['result'].get('experimental_score')
            if score:
                # Use the same columns as the raw metric export, with an explicitly
                # separate experimental row. No existing raw measurement is replaced.
                row={key:None for key in rows[0]} if rows else {}
                row.update(metric_id='EXPERIMENTAL_WEIGHTED_SCORE',dimension='experimental',
                    state=score['status'],value=score['score_percent'],reason=score['notice'],evidence=score)
                rows.append(row)
            return csv_response(rows,'readiness-'+id)
    elif format!='json':
        raise ValueError('This record supports JSON export; use the consolidated research CSV for comparisons')
    return JSONResponse(record,headers={'Content-Disposition':f'attachment; filename="research-{id}.json"'})


@router.post('/reports/{id}/score',response_model=dict)
def score_preview(id:str,body:ScorePreviewRequest):
    """Read-only score preview. Historical assessment and training inputs stay immutable."""
    from metrics.experimental_scoring import aggregate, calculate, enabled
    store,_=services()
    record=store.get(id,'job')['payload']
    if record['task']!='readiness' or record['status']!='completed':
        raise ValueError('A completed readiness report is required')
    if not enabled():
        return calculate(None,None,[],[])
    certificate=Certificate.model_validate(record['result'])
    score=certificate.experimental_score
    if score is None:
        # Old reports can be previewed from their saved, correctly scoped evidence;
        # no metrics, assessments or training jobs are rerun.
        ds=load(store,record['config']['dataset_id'])
        if record['config'].get('split_id'):
            split=get_split(store,record['config']['split_id'],record['config']['dataset_id'])
            ds=subset(ds,split['images']['train'])
        score=calculate(ds,certificate.confirmed_context.goal,certificate.metrics,certificate.findings)
    return {**aggregate(score,body.weights),'preview_of_report_id':id,'preview_only':True}


@router.get('/research/export')
def research_export(format:str=Query('json',pattern='^(json|csv|html|pdf)$')):
    store,_=services()
    data=research(store)
    saved=store.create('analysis',data)
    if format in ('html','pdf'):
        from backend.workbench.reporting import document, html_document, pdf_document
        doc=document(store,saved)
        if format=='pdf':
            return Response(pdf_document(doc),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename="validation-research.pdf"'})
        return HTMLResponse(html_document(doc),headers={'Content-Disposition':'attachment; filename="validation-research.html"'})
    if format=='csv':
        return csv_response(data['rows'],'research')
    # Include exact manifests/configs and evidence, with large predictions in per-evaluation files.
    data['provenance']={kind:store.list(kind) for kind in ('dataset','split','context','model','evaluation','comparison','job')}
    return JSONResponse(data,headers={'Content-Disposition':'attachment; filename="research-evidence.json"'})


@router.get('/settings',response_model=dict)
def settings():
    return {'storage_root':str(services()[0].root),'execution':'One isolated worker process at a time; CPU and GPU jobs serialized',
            'remote_context_default':False,'supported_training_formats':['yolo','coco'], 'detector':'Ultralytics YOLO11n / YOLO11s',
            'weights':'Scratch initialization or trusted local .pt weights. No automatic downloads.',
            'license':'Ultralytics AGPL-3.0 or Enterprise; see https://www.ultralytics.com/license',
            'limits':{'max_upload_bytes':int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2,'max_files':int(os.getenv('MAX_ZIP_FILES','50000'))}}
