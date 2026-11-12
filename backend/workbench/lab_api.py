"""Lab orchestration, documented external benchmarks, and trusted checkpoint import."""
from pathlib import Path
from typing import Literal
import shutil
from threading import RLock
from fastapi import APIRouter
from pydantic import Field, HttpUrl
from backend.context import Model
from backend.workbench.api import ErrorRoute, services
from backend.workbench.contracts import RecordResponse, TrainRequest
from backend.workbench.experiments import ExperimentRequest, validate_experiment
from backend.workbench.detector import validate_training, setup_detector, validation_source, training_only_manifest
from backend.workbench.datasets import file_sha

router = APIRouter(prefix='/api/v1', tags=['Model Validation Lab'], route_class=ErrorRoute)
lab_action_lock=RLock()


@router.get('/experiments', response_model=list[RecordResponse])
def experiments():
    return services()[0].list('experiment')


@router.post('/experiments', response_model=RecordResponse, status_code=201)
def create_experiment(body: ExperimentRequest):
    store, _ = services()
    protocol = validate_experiment(store, body)
    return store.create('experiment', {'request': body.model_dump(mode='json'), 'protocol': protocol,
        'version': 'validation-protocol-1.0', 'status': 'configured', 'child_job_ids': [], 'model_ids': [], 'evaluation_ids': []})


@router.get('/experiments/{id}', response_model=RecordResponse)
def experiment(id: str):
    return services()[0].get(id, 'experiment')


@router.post('/experiments/{id}/start', response_model=RecordResponse, status_code=202)
def start_experiment(id: str):
    store, manager = services()
    with lab_action_lock:
        record = store.get(id, 'experiment')['payload']
        if record['status'] != 'configured':
            raise ValueError('This immutable experiment has already started; create another protocol to repeat it')
        validate_experiment(store, ExperimentRequest.model_validate(record['request']))
        store.update(id, status='queued')
        try:
            job = manager.enqueue('experiment', {'experiment_id': id})
        except Exception:
            store.update(id, status='configured')
            raise
        store.update(id, job_id=job['id'])
    return job


class ExternalBenchmarkRequest(Model):
    title: str = Field(min_length=1, max_length=240)
    source_url: HttpUrl
    citation: str = Field(min_length=1, max_length=2000)
    architecture: str = Field(min_length=1, max_length=160)
    training_dataset: str = Field(min_length=1, max_length=500)
    evaluation_dataset: str = Field(min_length=1, max_length=500)
    protocol: str = Field(min_length=1, max_length=4000)
    metrics: dict[str, float] = Field(min_length=1, max_length=40)
    license_notes: str = Field(default='', max_length=2000)


@router.post('/benchmarks/external', response_model=RecordResponse, status_code=201)
def external_benchmark(body: ExternalBenchmarkRequest):
    return services()[0].create('external_benchmark', {**body.model_dump(mode='json'),
        'category': 'external_published', 'comparability': 'unverified',
        'warning': 'Published values are supplied by the user, not locally measured. Dataset, labels, training budget and protocol may differ; excluded from controlled rankings and quality correlations.'})


@router.get('/benchmarks/external', response_model=list[RecordResponse])
def external_benchmarks():
    return services()[0].list('external_benchmark')


class ImportModelRequest(Model):
    path: str
    training: TrainRequest
    provenance: str = Field(min_length=1, max_length=4000)
    trusted_checkpoint_confirmed: bool = False


@router.post('/models/import', response_model=RecordResponse, status_code=201)
def import_model(body: ImportModelRequest):
    if not body.trusted_checkpoint_confirmed:
        raise ValueError('Confirm this is a trusted checkpoint before loading a .pt file')
    store, _ = services()
    classes, split = validate_training(store, body.training.model_dump())
    source = Path(body.path).resolve()
    if source.suffix.lower() != '.pt' or not source.is_file():
        raise ValueError('Choose an existing trusted local .pt checkpoint')
    model = setup_detector()(str(source), task='detect')
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if names != classes:
        raise ValueError('Imported model class names/order do not match the declared training schema')
    yaml = model.model.yaml
    if 'yolo11' not in str(yaml.get('yaml_file', '')) or yaml.get('scale', 'n') != body.training.architecture[-1]:
        raise ValueError('Imported checkpoint architecture does not match the declaration')
    import uuid
    directory = store.directory('models', uuid.uuid4().hex)
    artifact = directory / 'imported.pt'
    shutil.copyfile(source, artifact)
    configuration = {**body.training.model_dump(), 'classes': classes, 'selected_device': 'unverified',
                     'initialization_sha256': None, 'imported': True}
    validation=validation_source(store,body.training.model_dump(),classes)
    manifests=[training_only_manifest(split),validation['manifest']] if validation else [split]
    return store.create('model', {'job_id': None, 'dataset_id': body.training.dataset_id, 'readiness_id': body.training.readiness_id,
        'configuration': configuration, 'artifact': str(artifact), 'artifact_sha256': file_sha(artifact),
        'validation': {}, 'development_manifests': manifests, 'training_images': len(split['images']['train']),
        'imported': True, 'provenance': body.provenance,
        'comparability_warning': 'Training configuration and development provenance are user declared, not independently verified.'})


@router.post('/jobs/{id}/resume', response_model=RecordResponse, status_code=202)
def resume(id: str):
    store, manager = services()
    # Never hold Store.lock while entering JobManager: its worker uses the
    # opposite lock order. Serialize route actions with a separate lock.
    with lab_action_lock:
        job = store.get(id, 'job')['payload']
        if job['task'] != 'training' or job['status'] not in ('cancelled', 'failed', 'interrupted'):
            raise ValueError('Only interrupted, failed or cancelled training jobs can resume')
        checkpoint = Path(job.get('run_directory', store.directory('jobs', id) / 'run')) / 'weights' / 'last.pt'
        if not checkpoint.is_file():
            raise ValueError('No resumable checkpoint exists; complete at least one epoch before cancellation')
        if any(j['payload']['status'] in ('queued', 'running') and j['payload']['config'].get('resume_from_job_id') == id for j in store.list('job')):
            raise ValueError('A resume for this checkpoint is already active')
        return manager.enqueue('training', {**job['config'], 'resume_from_job_id': id})
