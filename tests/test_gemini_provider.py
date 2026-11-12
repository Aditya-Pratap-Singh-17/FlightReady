"""Exercise the real HTTP payload/validation path without spending API quota."""
import json
import httpx
import pytest
import backend.context_extractor as extractor

TEXT = 'Classify text using label column label'
TAGS = {'Task': ['Classify text'], 'Data type': ['Text']}

@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv('CONTEXT_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
    monkeypatch.setenv('CONTEXT_MODEL', 'gemini-2.5-flash-lite')

def mock_api(monkeypatch, handler):
    # Keep a reference before replacing the module's Client to avoid recursion.
    client = httpx.Client
    monkeypatch.setattr(extractor.httpx, 'Client', lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))

def reply(content, finish='STOP'):
    return {'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': json.dumps(content)}]}}]}

def valid_context():
    return {'original_text': 'model changed text', 'selected_tags': {}, 'confirmed': True,
            'goal': {'task_type': 'classification', 'prediction_target': 'label', 'input_modalities': ['text']}}

def test_gemini_request_and_untrusted_response(gemini, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        assert str(request.url) == 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent'
        assert request.headers['x-goog-api-key'] == 'test-only-key'
        assert 'test-only-key' not in str(request.url)
        payload = json.loads(request.content)
        assert json.loads(payload['contents'][0]['parts'][0]['text']) == {'text': TEXT, 'tags': TAGS}
        assert payload['generationConfig']['responseMimeType'] == 'application/json'
        assert 'tools' not in payload
        return httpx.Response(200, json=reply(valid_context()))
    mock_api(monkeypatch, handler)
    result = extractor.interpret(TEXT, TAGS)
    assert len(calls) == 1
    assert result.provider == 'gemini:gemini-2.5-flash-lite'
    assert result.original_text == TEXT and result.selected_tags == TAGS and not result.confirmed

@pytest.mark.parametrize('status,body', [
    (429, {'error': {'message': 'quota exhausted'}}),
    (403, {'error': {'message': 'invalid key'}}),
    (200, {'promptFeedback': {'blockReason': 'SAFETY'}}),
    (200, reply(valid_context(), 'MAX_TOKENS')),
    (200, reply({'goal': {'task_type': 'made-up-task'}})),
    (200, reply({**valid_context(), 'goal': {'task_type': 'classification', 'max_missing_fraction': {'text': .15}}})),
])
def test_gemini_failure_falls_back_once(gemini, monkeypatch, status, body):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)
    mock_api(monkeypatch, handler)
    result = extractor.interpret(TEXT, TAGS)
    assert result.provider.startswith('deterministic')
    assert result.goal.prediction_target == 'label'
    assert not result.goal.max_missing_fraction and not result.confirmed
    assert len(calls) == 1  # No retry and no alternate paid model.

def test_missing_key_and_disabled_provider_never_request(gemini, monkeypatch):
    def fail(request):
        pytest.fail('An unconfigured/disabled provider must not send a request')
    mock_api(monkeypatch, fail)
    monkeypatch.delenv('GEMINI_API_KEY')
    assert extractor.interpret(TEXT, TAGS).provider.startswith('deterministic')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-key')
    monkeypatch.setenv('CONTEXT_PROVIDER', 'deterministic')
    assert extractor.interpret(TEXT, TAGS).provider == 'deterministic'

def test_legacy_provider_selection_is_preserved(monkeypatch):
    monkeypatch.delenv('CONTEXT_PROVIDER', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('CONTEXT_MODEL', 'existing-model')
    monkeypatch.setattr(extractor.OpenAICompatibleProvider, 'extract', lambda self, text, tags: extractor.DatasetContext(
        original_text=text, selected_tags=tags, goal=extractor.Goal(task_type='classification'), provider='legacy-test'))
    assert extractor.interpret(TEXT, TAGS).provider == 'legacy-test'


@pytest.mark.parametrize('status,body,code', [
    (400, {'error': {'details': [{'reason': 'API_KEY_INVALID'}]}}, 'invalid_key'),
    (401, {'error': {'message': 'secret-must-not-leak'}}, 'invalid_key'),
    (403, {'error': {'message': 'secret-must-not-leak'}}, 'access_denied'),
    (404, {'error': {'message': 'secret-must-not-leak'}}, 'model_unavailable'),
    (429, {'error': {'message': 'secret-must-not-leak'}}, 'quota_exceeded'),
])
def test_provider_failure_exposes_safe_actionable_reason(gemini, monkeypatch, status, body, code):
    mock_api(monkeypatch, lambda request: httpx.Response(status, json=body))
    result = extractor.interpret(TEXT, TAGS)
    assert result.provider_error_code == code
    assert result.provider_error
    encoded = result.model_dump_json()
    assert 'secret-must-not-leak' not in encoded and 'test-only-key' not in encoded
    assert result.goal.prediction_target == 'label'


def test_connection_error_never_exposes_exception_or_request(gemini, monkeypatch):
    def handler(request):
        raise httpx.ConnectError('secret-must-not-leak', request=request)
    mock_api(monkeypatch, handler)
    result = extractor.interpret(TEXT, TAGS)
    assert result.provider_error_code == 'connection_error'
    assert 'secret-must-not-leak' not in result.model_dump_json()
    assert 'test-only-key' not in result.model_dump_json()


def test_network_permission_failure_is_distinguished_from_invalid_key(gemini, monkeypatch):
    def handler(request):
        raise httpx.ConnectError('secret-must-not-leak', request=request) from PermissionError('private path')
    mock_api(monkeypatch, handler)
    result = extractor.interpret(TEXT, TAGS)
    assert result.provider_error_code == 'network_denied'
    assert 'private path' not in result.model_dump_json()
