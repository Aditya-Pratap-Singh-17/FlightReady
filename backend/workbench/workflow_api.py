"""Guided projects retain goals and orchestrate existing assessment/lab services."""
import os
from threading import RLock
from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from backend.context import Model, DatasetContext
from backend.context_extractor import interpret
from backend.workbench.api import ErrorRoute, services, readiness
from backend.workbench.contracts import RecordResponse, ReadinessRequest, SplitRequest, VariantRequest, TrainRequest
from backend.workbench.splits import create_split, get_split
from backend.workbench.variations import variation
from backend.workbench.experiments import ExperimentRequest
from backend.workbench.lab_api import create_experiment

router = APIRouter(prefix='/api/v1', tags=['Guided workflow'], route_class=ErrorRoute)
action_lock = RLock()


class ProjectRequest(Model):
    dataset_id: str


class MessageRequest(Model):
    message: str = Field(min_length=1, max_length=6000)
    enable_remote: bool = True


class ConfirmRequest(Model):
    context: DatasetContext


class CompatibilityRequest(Model):
    dataset_id: str
    context: DatasetContext


@router.post('/compatibility', response_model=dict)
def preview_compatibility(body: CompatibilityRequest):
    from backend.compatibility import compatibility
    from backend.workbench.datasets import load
    return compatibility(load(services()[0], body.dataset_id), body.context)


class AssessRequest(Model):
    scope: Literal['training', 'entire'] = 'training'
    split_id: str | None = None
    split: SplitRequest = Field(default_factory=SplitRequest)


class PrepareRequest(Model):
    intervention: Literal['blur', 'resolution', 'label_omission', 'box_perturbation', 'wrong_class', 'noise', 'exposure'] = 'blur'
    severity: float = Field(default=.25, gt=0, le=1)
    acknowledge_missing_groups: bool = False


class ProtocolRequest(Model):
    epochs: int = Field(default=10, ge=1, le=10000)
    image_size: int = Field(default=512, ge=64, le=2048, multiple_of=32)
    batch: int = Field(default=4, ge=1, le=256)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=20)
    device: Literal['auto', 'cpu', '0'] = 'auto'


def project(store, id):
    return store.get(id, 'workflow')['payload']


def require_idle(store, data):
    for key in ('readiness_id',):
        if data.get(key) and store.get(data[key], 'job')['payload']['status'] in ('queued', 'running'):
            raise ValueError('Wait for the active analysis to finish or cancel it before changing this project')
    if data.get('validation_id'):
        val = store.get(data['validation_id'], 'workflow_validation')['payload']
        ready = store.get(val['readiness_id'], 'job')['payload']
        if ready['status'] in ('queued', 'running'):
            raise ValueError('Wait for comparison preparation to finish before changing this project')
        if val.get('experiment_id') and store.get(val['experiment_id'], 'experiment')['payload']['status'] in ('queued', 'running'):
            raise ValueError('Wait for model validation to finish before changing this project')


@router.get('/context/provider', response_model=dict)
def provider_status():
    # Presence flags only: no credential values, paths, URLs or headers returned.
    provider = os.getenv('CONTEXT_PROVIDER', '').strip().lower()
    gemini = (provider == 'gemini' or not provider and bool(os.getenv('GEMINI_API_KEY'))) and bool(os.getenv('GEMINI_API_KEY', '').strip())
    other = (provider == 'openai-compatible' or not provider and not gemini) and bool(os.getenv('CONTEXT_MODEL'))
    return {'configured': bool(gemini or other), 'name': 'Gemini' if gemini else 'Configured language model' if other else 'Local interpreter'}


@router.post('/workflows', response_model=RecordResponse, status_code=201)
def create_project(body: ProjectRequest):
    store, _ = services()
    store.get(body.dataset_id, 'dataset')
    return store.create('workflow', {'dataset_id': body.dataset_id, 'messages': [], 'context': None,
        'context_id': None, 'confirmed_context_id': None, 'readiness_id': None, 'split_id': None, 'validation_id': None})


@router.get('/workflows', response_model=list[RecordResponse])
def projects():
    return services()[0].list('workflow')


@router.get('/workflows/{id}', response_model=RecordResponse)
def get_project(id: str):
    return services()[0].get(id, 'workflow')


@router.post('/workflows/{id}/goal', response_model=RecordResponse)
def project_goal(id: str, body: MessageRequest):
    store, _ = services()
    with action_lock:
        data = project(store, id)
        require_idle(store, data)
        messages = [*data['messages'], {'role': 'user', 'text': body.message.strip()}]
        text = '\n\n'.join(m['text'] for m in messages if m['role'] == 'user')
        if not body.message.strip() or len(text) > 12000:
            raise ValueError('Keep the complete goal conversation between 1 and 12,000 characters; start a new project for a new goal')
        result = interpret(text, {}, allow_remote=body.enable_remote)
        saved = store.create('context', result.model_dump(mode='json'))
        return store.update(id, messages=messages, context=saved['payload'], context_id=saved['id'],
            confirmed_context_id=None, readiness_id=None, split_id=None, validation_id=None)


@router.post('/workflows/{id}/confirm', response_model=RecordResponse)
def confirm_project(id: str, body: ConfirmRequest):
    store, _ = services()
    with action_lock:
        data = project(store, id)
        require_idle(store, data)
        context = body.context
        if not data.get('context') or context.original_text != data['context']['original_text']:
            raise ValueError('Review the latest interpreted goal before confirming')
        if not context.confirmed or context.clarification_questions or context.goal.task_type == 'unspecified':
            raise ValueError('Choose a task and resolve the clarification questions before confirming')
        saved = store.create('context', context.model_dump(mode='json'))
        return store.update(id, context=saved['payload'], context_id=saved['id'], confirmed_context_id=saved['id'],
            readiness_id=None, split_id=None, validation_id=None)


@router.post('/workflows/{id}/assess', response_model=RecordResponse, status_code=202)
def assess_project(id: str, body: AssessRequest):
    store, _ = services()
    with action_lock:
        data = project(store, id)
        # Repeated clicks return the original active job; they never enqueue twice.
        if data.get('readiness_id'):
            job = store.get(data['readiness_id'], 'job')
            if job['payload']['status'] in ('queued', 'running', 'completed'):
                return job
        if not data.get('confirmed_context_id'):
            raise ValueError('Confirm and save the project goal before analysis')
        context = DatasetContext.model_validate(store.get(data['confirmed_context_id'], 'context')['payload'])
        from backend.compatibility import compatibility
        from backend.workbench.datasets import load
        preview = compatibility(load(store, data['dataset_id']), context)
        if body.scope == 'training' and preview['status'] == 'incompatible':
            raise ValueError(preview['summary'] + ' Use an entire-dataset assessment to document the mismatch; training is blocked.')
        split_id = None
        if body.scope == 'training':
            split_id = body.split_id or data.get('split_id')
            if split_id:
                get_split(store, split_id, data['dataset_id'])
            else:
                split_id = create_split(store, data['dataset_id'], body.split)['id']
                store.update(id, split_id=split_id)
        job = readiness(ReadinessRequest(dataset_id=data['dataset_id'], context=context, split_id=split_id))
        store.update(id, readiness_id=job['id'], split_id=split_id, validation_id=None)
        return job


@router.post('/workflows/{id}/validation/prepare', response_model=RecordResponse, status_code=202)
def prepare_validation(id: str, body: PrepareRequest):
    store, manager = services()
    with action_lock:
        data = project(store, id)
        if data.get('validation_id'):
            return store.get(data['validation_id'], 'workflow_validation')
        if not data.get('readiness_id') or not data.get('split_id'):
            raise ValueError('Analyze the training partition before model validation')
        assessment = store.get(data['readiness_id'], 'job')['payload']
        if assessment['status'] != 'completed':
            raise ValueError('Wait for a completed readiness report')
        context = DatasetContext.model_validate(assessment['result']['confirmed_context'])
        if context.goal.task_type != 'object_detection':
            raise ValueError('The current training adapter supports object detection; this task has readiness analysis only')
        variant = variation(store, VariantRequest(dataset_id=data['dataset_id'], split_id=data['split_id'],
            intervention=body.intervention, severity=body.severity))
        source = get_split(store, data['split_id'], data['dataset_id'])
        split = create_split(store, variant['id'], SplitRequest(**{**source['config'], 'acknowledge_missing_groups': body.acknowledge_missing_groups}))
        ready = manager.enqueue('readiness', {'dataset_id': variant['id'], 'split_id': split['id'], 'context': context.model_dump(mode='json')})
        saved = store.create('workflow_validation', {'workflow_id': id, 'dataset_id': variant['id'], 'split_id': split['id'],
            'readiness_id': ready['id'], 'request': body.model_dump(), 'experiment_id': None})
        store.update(id, validation_id=saved['id'])
        return saved


@router.get('/workflow-validations/{id}', response_model=RecordResponse)
def get_validation(id: str):
    return services()[0].get(id, 'workflow_validation')


@router.post('/workflows/{id}/validation/reset', response_model=RecordResponse)
def reset_validation(id: str):
    store, _ = services()
    with action_lock:
        data = project(store, id)
        require_idle(store, data)
        return store.update(id, validation_id=None)


@router.post('/workflows/{id}/validation/protocol', response_model=RecordResponse, status_code=201)
def validation_protocol(id: str, body: ProtocolRequest):
    store, _ = services()
    with action_lock:
        data = project(store, id)
        if not data.get('validation_id'):
            raise ValueError('Prepare the comparison dataset first')
        val = store.get(data['validation_id'], 'workflow_validation')['payload']
        if val.get('experiment_id'):
            return store.get(val['experiment_id'], 'experiment')
        ready = store.get(val['readiness_id'], 'job')['payload']
        if ready['status'] != 'completed':
            raise ValueError('Wait for comparison readiness analysis to complete')
        ack = val['request']['acknowledge_missing_groups']
        common = dict(epochs=body.epochs, image_size=body.image_size, batch=body.batch, device=body.device,
            validation_dataset_id=data['dataset_id'], validation_split_id=data['split_id'], acknowledge_validation_groups=ack)
        candidate = TrainRequest(**common, dataset_id=data['dataset_id'], split_id=data['split_id'], readiness_id=data['readiness_id'])
        reference = TrainRequest(**common, dataset_id=val['dataset_id'], split_id=val['split_id'], readiness_id=val['readiness_id'])
        context = store.get(data['confirmed_context_id'], 'context')['payload']
        request = ExperimentRequest(name=f"Readiness validation {id[:8]}",
            hypothesis=f"For the goal '{context['goal']['task_description']}', changing {val['request']['intervention']} at severity {val['request']['severity']} changes measured quality and held-out performance.",
            candidate=candidate, reference=reference, modes=['candidate', 'reference'], seeds=body.seeds,
            test_dataset_id=data['dataset_id'], test_split_id=data['split_id'], acknowledge_missing_groups=ack)
        record = create_experiment(request)
        store.update(data['validation_id'], experiment_id=record['id'])
        return record
