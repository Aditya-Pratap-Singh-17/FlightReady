"""Workflow transitions use durable goals and the real assessment/experiment contracts."""
import copy
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sql.store import Store
from backend.workbench.synthetic import detection_fixture
from backend.workbench.datasets import import_file, image_bytes
from backend.workbench.worker import run


@pytest.fixture
def guided(tmp_path, monkeypatch):
    from backend.workbench import api, workflow_api, lab_api
    store = Store(tmp_path/'store')
    class Queue:
        def enqueue(self, task, config):
            return store.create('job', {'task': task, 'config': config, 'status': 'queued', 'stage': 'Waiting', 'error': None, 'result': None})
    for module in (api, workflow_api, lab_api):
        monkeypatch.setattr(module, 'services', lambda: (store, Queue()))
    app = FastAPI()
    app.include_router(api.router)
    app.include_router(lab_api.router)
    app.include_router(workflow_api.router)
    path = detection_fixture(tmp_path/'guided.zip', seed=318)
    with path.open('rb') as incoming:
        dataset = import_file(store, path.name, incoming)
    with TestClient(app) as client:
        workflow = client.post('/api/v1/workflows', json={'dataset_id': dataset['id']}).json()
        yield store, client, workflow, dataset
    store.engine.dispose()


def confirm(client, workflow):
    id = workflow['id']
    response = client.post(f'/api/v1/workflows/{id}/goal', json={'message': 'Detect drones in images', 'enable_remote': False})
    assert response.status_code == 200, response.text
    context = response.json()['payload']['context']
    context['confirmed'] = True
    context['clarification_questions'] = []
    response = client.post(f'/api/v1/workflows/{id}/confirm', json={'context': context})
    assert response.status_code == 200, response.text
    return response.json()


def assess(client, workflow):
    confirm(client, workflow)
    response = client.post(f"/api/v1/workflows/{workflow['id']}/assess", json={'scope': 'training'})
    assert response.status_code == 202, response.text
    return response.json()


def test_goal_is_required_saved_immutably_and_reused_by_pipeline(guided):
    store, client, workflow, dataset = guided
    endpoint = f"/api/v1/workflows/{workflow['id']}"
    assert client.post(endpoint+'/assess', json={}).status_code == 422
    assert client.post(endpoint+'/validation/prepare', json={}).status_code == 422
    saved = confirm(client, workflow)
    goal_id = saved['payload']['confirmed_context_id']
    frozen = store.get(goal_id, 'context')
    assert frozen['payload']['confirmed']
    assert frozen['payload']['goal']['required_classes'] == ['drone']
    response = client.post(endpoint+'/assess', json={})
    assert response.status_code == 202, response.text
    job = response.json()
    assert job['payload']['config']['context'] == frozen['payload']
    assert job['payload']['config']['dataset_id'] == dataset['id']
    assert store.get(goal_id, 'context') == frozen
    restored = Store(store.root)
    assert restored.get(workflow['id'], 'workflow')['payload']['readiness_id'] == job['id']
    restored.engine.dispose()


def test_repeated_assessment_is_idempotent_and_changes_blocked_during_processing(guided):
    store, client, workflow, _ = guided
    job = assess(client, workflow)
    endpoint = f"/api/v1/workflows/{workflow['id']}"
    assert client.post(endpoint+'/assess', json={}).json()['id'] == job['id']
    assert len(store.list('job')) == 1 and len(store.list('split')) == 1
    assert client.post(endpoint+'/goal', json={'message': 'Change the goal', 'enable_remote': False}).status_code == 422
    store.update(job['id'], status='failed')
    retry = client.post(endpoint+'/assess', json={})
    assert retry.status_code == 202
    assert retry.json()['id'] != job['id']
    assert len(store.list('split')) == 1


def test_conversation_keeps_user_text_and_remote_opt_in_without_dataset_content(guided, monkeypatch):
    from backend.workbench import workflow_api
    from backend.context_extractor import interpret as local
    _, client, workflow, _ = guided
    calls = []
    def extract(text, tags, *, allow_remote):
        calls.append((text, tags, allow_remote))
        return local(text, tags, allow_remote=False)
    monkeypatch.setattr(workflow_api, 'interpret', extract)
    endpoint = f"/api/v1/workflows/{workflow['id']}/goal"
    client.post(endpoint, json={'message': 'Detect drones in images', 'enable_remote': True})
    response = client.post(endpoint, json={'message': 'Use nighttime imagery', 'enable_remote': False})
    assert response.status_code == 200
    assert calls == [('Detect drones in images', {}, True), ('Detect drones in images\n\nUse nighttime imagery', {}, False)]
    assert len(response.json()['payload']['messages']) == 2
    assert response.json()['payload']['context']['goal']['target_environment']['lighting'] == ['nighttime']


def test_goal_revision_resets_current_results_but_keeps_previous_records(guided):
    store, client, workflow, _ = guided
    job = assess(client, workflow)
    old_id = store.get(workflow['id'])['payload']['confirmed_context_id']
    store.update(job['id'], status='completed')
    response = client.post(f"/api/v1/workflows/{workflow['id']}/goal", json={'message': 'Prefer urban images', 'enable_remote': False})
    assert response.status_code == 200
    data = response.json()['payload']
    assert data['confirmed_context_id'] is None and data['readiness_id'] is None
    assert store.get(old_id, 'context')['payload']['confirmed']
    assert store.get(job['id'], 'job')['payload']['status'] == 'completed'
    stale = copy.deepcopy(store.get(old_id)['payload'])
    assert client.post(f"/api/v1/workflows/{workflow['id']}/confirm", json={'context': stale}).status_code == 422


def test_controlled_validation_measures_real_data_and_freezes_shared_protocol(guided):
    store, client, workflow, dataset = guided
    job = assess(client, workflow)
    run(store, job['id'])
    assert store.get(job['id'])['payload']['status'] == 'completed'
    endpoint = f"/api/v1/workflows/{workflow['id']}"
    response = client.post(endpoint+'/validation/prepare', json={'intervention': 'blur', 'severity': .25})
    assert response.status_code == 202, response.text
    validation = response.json()
    val = validation['payload']
    assert val['dataset_id'] != dataset['id']
    assert client.post(endpoint+'/validation/prepare', json={'intervention': 'resolution'}).json()['id'] == validation['id']
    assert client.post(endpoint+'/validation/protocol', json={'epochs': 1}).status_code == 422
    assert client.post(endpoint+'/validation/reset').status_code == 422
    run(store, val['readiness_id'])
    protocol = client.post(endpoint+'/validation/protocol', json={'epochs': 1, 'image_size': 64, 'batch': 2})
    assert protocol.status_code == 201, protocol.text
    frozen = protocol.json()['payload']
    assert frozen['protocol']['test_images'] == 2
    assert frozen['protocol']['requested_training_sizes'] == [6, 6]
    a, b = frozen['request']['candidate'], frozen['request']['reference']
    assert a['validation_dataset_id'] == b['validation_dataset_id'] == dataset['id']
    assert a['validation_split_id'] == b['validation_split_id']
    assert frozen['request']['test_split_id'] == a['split_id']
    # Actual archived test bytes survive the intervention.
    varied = store.get(val['dataset_id'])['payload']['provenance']['source_mapping']
    for modified, original in varied.items():
        if '/test/' in modified:
            assert image_bytes(store, val['dataset_id'], modified) == image_bytes(store, dataset['id'], original)
    assert client.post(endpoint+'/validation/protocol', json={'epochs': 100}).json()['id'] == protocol.json()['id']
    assert client.post(endpoint+'/validation/reset').status_code == 200
    assert store.get(validation['id'], 'workflow_validation')['payload']['experiment_id'] == protocol.json()['id']


def test_provider_status_exposes_only_presence_and_name(guided, monkeypatch):
    _, client, _, _ = guided
    monkeypatch.setenv('CONTEXT_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-workflow-test-secret')
    response = client.get('/api/v1/context/provider')
    assert response.json() == {'configured': True, 'name': 'Gemini'}
    assert 'fake-workflow-test-secret' not in response.text
    monkeypatch.delenv('GEMINI_API_KEY')
    assert client.get('/api/v1/context/provider').json()['configured'] is False


def test_entire_dataset_assessment_does_not_create_training_evidence(guided):
    store, client, workflow, _ = guided
    confirm(client, workflow)
    endpoint = f"/api/v1/workflows/{workflow['id']}"
    job = client.post(endpoint+'/assess', json={'scope': 'entire'}).json()
    assert job['payload']['config']['split_id'] is None
    assert not store.list('split')
    run(store, job['id'])
    assert client.post(endpoint+'/validation/prepare', json={}).status_code == 422
