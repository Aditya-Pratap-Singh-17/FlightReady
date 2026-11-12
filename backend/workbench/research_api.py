"""Connected canonical reports, model acquisition and inference-only execution."""
from typing import Literal
import json
from fastapi import APIRouter, Query
from pydantic import Field, StrictBool, model_validator
from backend.context import Model
from backend.workbench.api import ErrorRoute, services
from backend.workbench.contracts import RecordResponse
from backend.workbench.model_registry import registry, entry
from backend.workbench.inference import validate_benchmark

router = APIRouter(prefix='/api/v1', tags=['Connected research'], route_class=ErrorRoute)


class AcquisitionRequest(Model):
    registry_id: str | None = None
    source_id: str | None = None
    download_consent: StrictBool = False
    license_accepted: StrictBool = False
    local_only: StrictBool = False

    @model_validator(mode='after')
    def one_source(self):
        if bool(self.registry_id)==bool(self.source_id): raise ValueError('Choose a registry model or an inspected Hugging Face source')
        return self


class SourceRequest(Model):
    repository: str = Field(min_length=3,max_length=240)
    revision: str = Field(default='main',min_length=1,max_length=160)


class LocalModelRequest(Model):
    path: str = Field(min_length=1,max_length=2048)
    license_accepted: StrictBool = False


@router.post('/model-sources/inspect', response_model=RecordResponse)
def inspect_model(body: SourceRequest):
    from backend.workbench.model_registry import inspect_repository
    return inspect_repository(services()[0],body.repository,body.revision)


@router.post('/acquired-models/local', response_model=RecordResponse,status_code=201)
def local_model(body: LocalModelRequest):
    from backend.workbench.model_registry import import_local
    return import_local(services()[0],body.path,body.license_accepted)


@router.get('/inference-runtime', response_model=dict)
def inference_runtime():
    import importlib.util
    required=('torch','transformers','safetensors','huggingface_hub','pycocotools')
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    return {'available':not missing,'missing':missing,'supported_architectures':['YOLOS','DETR','RT-DETR'],
            'install_command':'.\\.venv\\Scripts\\python.exe -m pip install -r requirements/requirements-inference.txt'}


class BenchmarkRequest(Model):
    dataset_id: str
    readiness_id: str | None = None
    acquired_model_id: str
    mode: Literal['benchmark','preview'] = 'benchmark'
    max_images: int | None = Field(default=None,ge=1,le=10000)
    evaluation_classes: list[str] = Field(default_factory=list,max_length=1000)
    validate_project_goal: StrictBool = True
    split_id: str | None = None
    class_mapping: dict[str,str] = Field(default_factory=dict, max_length=1000)
    acknowledge_evaluation_scope: StrictBool = False
    confidence: float = Field(default=.25, gt=0, lt=1)
    match_iou: float = Field(default=.5, gt=0, le=1)
    image_size: int = Field(default=512, ge=64, le=1024)
    device: Literal['auto','cpu','0'] = 'auto'


@router.get('/model-registry', response_model=list[dict])
def models():
    return registry(services()[0])


@router.get('/acquired-models', response_model=list[RecordResponse])
def acquired_models():
    return services()[0].list('acquired_model')


@router.post('/model-registry/acquire', response_model=RecordResponse, status_code=202)
def acquire_model(body: AcquisitionRequest):
    if body.source_id: services()[0].get(body.source_id,'model_source')
    else: entry(body.registry_id)
    if not body.local_only and not body.download_consent or not body.license_accepted:
        raise ValueError('Review the repository, exact revision, approximate size and license; explicitly consent before download')
    return services()[1].enqueue('model_acquisition', body.model_dump())


@router.post('/inference-benchmarks', response_model=RecordResponse, status_code=202)
def start_benchmark(body: BenchmarkRequest):
    store, manager = services()
    validate_benchmark(store, body.model_dump())
    return manager.enqueue('inference_benchmark', body.model_dump())


@router.get('/inference-benchmarks', response_model=list[RecordResponse])
def benchmarks(readiness_id: str | None = None, dataset_id: str | None = None):
    return [r for r in services()[0].list('benchmark') if (not readiness_id or r['payload'].get('readiness_id') == readiness_id) and (not dataset_id or r['payload']['dataset_id']==dataset_id)]


@router.get('/inference-benchmarks/{id}/predictions', response_model=list[dict])
def benchmark_predictions(id: str, offset: int = Query(0,ge=0), limit: int = Query(1,ge=1,le=20)):
    from pathlib import Path
    data = services()[0].get(id, 'benchmark')['payload']
    return json.loads(Path(data['predictions_path']).read_text(encoding='utf-8'))[offset:offset+limit]


@router.get('/assessments/{id}/validation', response_model=dict)
def validation_supplement(id: str):
    """Live linked outcomes, explicitly distinct from the immutable assessment."""
    store, _ = services()
    source = store.get(id, 'job')['payload']
    if source.get('task') != 'readiness' or source.get('status') != 'completed':
        raise ValueError('A completed readiness assessment is required')
    result = source['result']; final = result.get('final_report')
    measured = benchmarks(id)
    models = {r['id'] for r in store.list('model') if r['payload'].get('readiness_id') == id}
    evaluations = [r for r in store.list('evaluation') if r['payload']['model_id'] in models]
    experiments = [r for r in store.list('experiment') if any((r['payload'].get('request',{}).get(key) or {}).get('readiness_id') == id for key in ('candidate','reference'))]
    active = [r for r in store.list('job') if r['payload'].get('status') in ('queued','running') and
        (r['payload'].get('config',{}).get('readiness_id') == id or r['payload'].get('config',{}).get('experiment_id') in {e['id'] for e in experiments})]
    goal_results=[r for r in measured if r['payload'].get('kind')!='inference_preview' and r['payload'].get('evaluation_scope',{}).get('project_goal_validated',True)]
    status = 'in_progress' if active else 'completed' if goal_results or evaluations else 'preview_or_annotation_benchmark_available' if measured else (final or {}).get('validation',{}).get('status','not_yet_validated')
    comparisons = []
    from backend.workbench.conclusions import experiment_conclusion
    for exp in experiments:
        comparisons.append({'experiment_id':exp['id'], 'scientific_interpretation':experiment_conclusion(store,exp)})
    return {'assessment_id':id, 'assessment_fingerprint':result['reproducibility'].get('assessment_fingerprint'),
        'report_sha256':(final or {}).get('report_sha256'), 'status':status, 'inference_benchmarks':measured,
        'evaluations':evaluations, 'experiments':experiments, 'comparisons':comparisons, 'active_jobs':active,
        'interpretation': 'These are linked empirical outcomes, not a revision to the original readiness assessment. Inference-only benchmarks do not test whether improving data quality improves training performance.'}
