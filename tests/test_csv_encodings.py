"""Small synthetic regressions for legacy spreadsheet CSV encodings.

The real user's dataset is deliberately not copied into the repository.
"""
import pytest
from fastapi.testclient import TestClient
from backend.ingestion import IngestionError, load_dataset
from main import app


def test_windows_1252_csv_preserves_smart_punctuation_and_all_rows():
    text = 'S#,Comments,Temperature(C)\r\n1,"A witness\u2019s account, recorded",28.475\r\n2,Another account,30\r\n'
    ds = load_dataset('legacy.csv', text.encode('cp1252'))
    assert ds.profile['row_count'] == 2
    assert ds.profile['column_count'] == 3
    assert ds.frame.loc[0, 'Comments'] == 'A witness\u2019s account, recorded'
    assert ds.frame.loc[0, 'Temperature(C)'] == 28.475
    assert ds.profile['parsing']['encoding'] == 'cp1252'
    assert ds.profile['parsing']['warnings']


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-8-sig', 'utf-16', 'utf-32'])
def test_unicode_csv_encodings(encoding):
    text = 'id,comment\n1,caf\u00e9\n2,hello\n'
    ds = load_dataset('unicode.csv', text.encode(encoding))
    assert ds.frame.loc[0, 'comment'] == 'caf\u00e9'
    assert ds.profile['row_count'] == 2
    assert not ds.profile['parsing']['warnings']


def test_explicit_csv_encoding_overrides_fallback():
    ds = load_dataset('legacy.csv', b'id,comment\n1,caf\xe9\n', {'csv_encoding': 'windows-1252'})
    assert ds.frame.loc[0, 'comment'] == 'caf\u00e9'
    assert ds.profile['parsing']['encoding'] == 'cp1252'
    assert ds.profile['parsing']['encoding_source'] == 'user_declared'


def test_wrong_declared_encoding_is_not_silently_overridden():
    with pytest.raises(IngestionError, match='UTF|utf'):
        load_dataset('legacy.csv', b'id,comment\n1,it\x92s\n', {'csv_encoding': 'utf-8'})


@pytest.mark.parametrize('encoding', ['not-an-encoding', 'unicode_escape', 123])
def test_unsupported_declared_encoding_is_rejected(encoding):
    with pytest.raises(IngestionError, match='encoding'):
        load_dataset('legacy.csv', b'id,label\n1,car\n', {'csv_encoding': encoding})


@pytest.mark.parametrize('data', [b'id,label\n1,a\x00b\n', b'id,label\n1,a\x01b\n', b'id,label\n1,a\x81b\n'])
def test_binary_controls_and_undecodable_bytes_are_rejected(data):
    with pytest.raises(IngestionError):
        load_dataset('bad.csv', data)


def test_legacy_encoding_does_not_disable_structural_checks():
    with pytest.raises(IngestionError, match='number of fields'):
        load_dataset('bad.csv', 'id,label\n1,it\u2019s,extra\n'.encode('cp1252'))


def test_upload_api_accepts_windows_1252_and_exposes_encoding():
    client = TestClient(app)
    response = client.post('/api/datasets/upload', files={'file': ('legacy.csv', b'id,comment\n1,it\x92s valid\n', 'text/csv')})
    assert response.status_code == 200, response.text
    uploaded = response.json()
    try:
        assert uploaded['profile']['parsing']['encoding'] == 'cp1252'
        assert uploaded['profile']['row_count'] == 1
    finally:
        client.delete('/api/datasets/' + uploaded['id'])
