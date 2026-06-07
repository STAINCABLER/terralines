from __future__ import annotations

import base64
import importlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.tasks as tasks


def test_enqueue_preview_uses_named_rq_callable():
    fake_job = SimpleNamespace(id='job-1')
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = fake_job

    with patch.object(tasks, '_get_queue', return_value=fake_queue):
        job_id = tasks.enqueue_preview({'width': 320})

    assert job_id == 'job-1'
    fake_queue.enqueue.assert_called_once_with('tasks._run_generate_preview', {'width': 320})


def test_enqueue_export_uses_named_rq_callable():
    fake_job = SimpleNamespace(id='job-2')
    fake_queue = MagicMock()
    fake_queue.enqueue.return_value = fake_job

    with patch.object(tasks, '_get_queue', return_value=fake_queue):
        job_id = tasks.enqueue_export({'width': 640})

    assert job_id == 'job-2'
    fake_queue.enqueue.assert_called_once_with('tasks._run_generate_export', {'width': 640})


def test_run_generate_preview_persists_png_when_job_context_exists():
    png_b64 = base64.b64encode(b'png-bytes').decode('utf-8')
    fake_result = {'image': png_b64, 'width': 1, 'height': 1, 'time_ms': 1}

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(tasks, 'RESULTS_DIR', Path(tmpdir)):
            with patch.object(tasks, 'generate_topography', return_value=dict(fake_result)) as generate_mock:
                with patch.object(tasks, 'get_current_job', return_value=SimpleNamespace(id='abc')):
                    result = tasks._run_generate_preview({'seed': 1})

        persisted = Path(result['result_path'])
        assert persisted.exists()

    generate_mock.assert_called_once_with({'seed': 1}, preview=True)
    assert result['result_path'].endswith('job_abc.png')


def test_run_generate_export_uses_full_resolution_mode():
    with patch.object(tasks, 'generate_topography', return_value={'ok': True}) as generate_mock:
        result = tasks._run_generate_export({'seed': 7})

    assert result == {'ok': True}
    generate_mock.assert_called_once_with({'seed': 7}, preview=False)


def test_worker_module_uses_redis_url_env_on_import():
    with patch.dict(os.environ, {'REDIS_URL': 'redis://example:6379/2'}):
        with patch('redis.Redis.from_url', return_value='MOCK_CONN') as from_url_mock:
            worker_module = importlib.reload(importlib.import_module('app.worker'))

    assert worker_module.redis_url == 'redis://example:6379/2'
    assert worker_module.conn == 'MOCK_CONN'
    assert from_url_mock.call_count >= 1
    assert all(call.args == ('redis://example:6379/2',) for call in from_url_mock.call_args_list)
