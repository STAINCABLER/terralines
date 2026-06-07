from __future__ import annotations

import base64
import copy
import importlib
import io
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


app_module = importlib.import_module('app.app')
flask_app = app_module.app
DEFAULT_PARAMS = app_module.DEFAULT_PARAMS


def _preview_params() -> dict:
    params = copy.deepcopy(DEFAULT_PARAMS)
    params.update(
        {
            'width': 320,
            'height': 180,
            'preview_scale': 0.1,
            'octaves': 1,
            'levels': 2,
            'smoothing': 0.0,
            'seed': 42,
        }
    )
    return params


def _client():
    flask_app.config['TESTING'] = True
    with app_module._rate_limit_lock:
        app_module._rate_limit_hits.clear()
    return flask_app.test_client()


def test_health_endpoint_is_available():
    client = _client()

    response = client.get('/health')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'worker_mode' in payload


def test_generate_requires_json_content_type():
    client = _client()

    response = client.post('/api/generate', data='{}', headers={'Content-Type': 'text/plain'})

    assert response.status_code == 400


def test_generate_rejects_invalid_json_body():
    client = _client()

    response = client.post('/api/generate', data='{"broken":', headers={'Content-Type': 'application/json'})

    assert response.status_code == 400


def test_generate_returns_busy_when_semaphore_is_saturated():
    client = _client()

    acquired = app_module.GENERATE_SEMAPHORE.acquire(blocking=False)
    assert acquired is True
    try:
        response = client.post('/api/generate', json=_preview_params())
    finally:
        app_module.GENERATE_SEMAPHORE.release()

    assert response.status_code == 429


def test_defaults_rate_limit_can_trigger_429():
    client = _client()
    old_limit = app_module.RATE_LIMIT_MAX_REQUESTS

    try:
        app_module.RATE_LIMIT_MAX_REQUESTS = 1
        with app_module._rate_limit_lock:
            app_module._rate_limit_hits.clear()

        first = client.get('/api/defaults')
        second = client.get('/api/defaults')
    finally:
        app_module.RATE_LIMIT_MAX_REQUESTS = old_limit
        with app_module._rate_limit_lock:
            app_module._rate_limit_hits.clear()

    assert first.status_code == 200
    assert second.status_code == 429


def test_export_png_returns_attachment():
    client = _client()

    response = client.post('/api/export', json=_preview_params())

    assert response.status_code == 200
    assert response.mimetype == 'image/png'
    assert 'attachment' in response.headers.get('Content-Disposition', '')


def test_export_svg_returns_svg_file():
    client = _client()

    response = client.post('/api/export/svg', json=_preview_params())

    assert response.status_code == 200
    assert response.mimetype == 'image/svg+xml'
    assert 'attachment' in response.headers.get('Content-Disposition', '')


def test_heightmap_requires_file():
    client = _client()

    response = client.post('/api/generate/heightmap', data={'params': '{}'})

    assert response.status_code == 400


def test_heightmap_rejects_non_image_upload():
    client = _client()

    response = client.post(
        '/api/generate/heightmap',
        data={
            'params': '{}',
            'heightmap': (io.BytesIO(b'not-an-image'), 'file.txt', 'text/plain'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 415


def test_heightmap_rejects_oversized_upload():
    client = _client()
    oversized = b'0' * (app_module.MAX_HEIGHTMAP_BYTES + 1)

    response = client.post(
        '/api/generate/heightmap',
        data={
            'params': '{}',
            'heightmap': (io.BytesIO(oversized), 'heightmap.png', 'image/png'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 413


def test_generate_async_enqueues_job_and_returns_job_id():
    client = _client()

    with patch.object(app_module, 'enqueue_job', return_value='job-123') as enqueue_mock:
        response = client.post('/api/generate_async', json=_preview_params())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['job_id'] == 'job-123'
    enqueue_mock.assert_called_once()


def test_job_status_returns_404_for_unknown_job():
    client = _client()

    with patch.object(app_module, 'get_job_status', return_value=None):
        response = client.get('/api/job/missing')

    assert response.status_code == 404


def test_job_status_exposes_download_url_for_finished_job():
    client = _client()
    fake_job = {
        'status': 'finished',
        'created_at': 1,
        'started_at': 2,
        'finished_at': 3,
        'result': {'image_path': '/tmp/terralines_results/job_abc.png'},
    }

    with patch.object(app_module, 'get_job_status', return_value=fake_job):
        response = client.get('/api/job/abc')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'finished'
    assert payload['download_url'].endswith('/api/job/abc/download')


def test_job_download_returns_409_when_job_not_finished():
    client = _client()

    with patch.object(app_module, 'get_job_status', return_value={'status': 'running'}):
        response = client.get('/api/job/abc/download')

    assert response.status_code == 409


def test_job_download_rejects_path_outside_results_dir():
    client = _client()
    fake_job = {
        'status': 'finished',
        'result_path': str(Path(tempfile.gettempdir()).parent / 'outside.png'),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {'TERRALINES_RESULTS_DIR': tmpdir}):
            with patch.object(app_module, 'get_job_status', return_value=fake_job):
                response = client.get('/api/job/abc/download')
                _ = response.get_data()
                response.close()

    assert response.status_code == 403


def test_job_download_serves_persisted_result_file():
    client = _client()

    tmpdir = tempfile.mkdtemp(prefix='terralines-test-')
    try:
        result_path = Path(tmpdir) / 'job_abc.png'
        result_path.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='))

        fake_job = {
            'status': 'finished',
            'result_path': str(result_path),
        }

        with patch.dict(os.environ, {'TERRALINES_RESULTS_DIR': tmpdir}):
            with patch.object(app_module, 'get_job_status', return_value=fake_job):
                response = client.get('/api/job/abc/download')
                _ = response.get_data()
                response.close()

        assert response.status_code == 200
        assert response.mimetype == 'image/png'
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_queue_status_endpoint_returns_counts():
    client = _client()

    response = client.get('/api/queue_status')

    assert response.status_code == 200
    payload = response.get_json()
    assert 'queued' in payload
    assert 'running' in payload
    assert 'finished' in payload


def test_generate_rq_returns_503_when_rq_unavailable():
    client = _client()

    with patch.object(app_module, '_rq_available', False):
        response = client.post('/api/generate_rq', json=_preview_params())

    assert response.status_code == 503


def test_generate_rq_returns_job_id_when_available():
    client = _client()
    mock_tasks = SimpleNamespace(enqueue_preview=lambda _params: 'rq-job-1')

    with patch.object(app_module, '_rq_available', True):
        with patch.object(app_module, 'rq_tasks', mock_tasks):
            response = client.post('/api/generate_rq', json=_preview_params())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['job_id'] == 'rq-job-1'
    assert payload['status_url'].endswith('/api/rq/job/rq-job-1')


def test_rq_job_status_returns_503_when_rq_unavailable():
    client = _client()

    with patch.object(app_module, '_rq_available', False):
        response = client.get('/api/rq/job/abc')

    assert response.status_code == 503


def test_results_endpoint_returns_404_for_missing_file():
    client = _client()

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {'TERRALINES_RESULTS_DIR': tmpdir}):
            response = client.get('/results/missing.png')

    assert response.status_code == 404


def test_results_endpoint_serves_existing_file():
    client = _client()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'job_demo.png'
        file_path.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='))

        with patch.dict(os.environ, {'TERRALINES_RESULTS_DIR': tmpdir}):
            response = client.get('/results/job_demo.png')
            response.close()

    assert response.status_code == 200
    assert response.mimetype == 'image/png'
