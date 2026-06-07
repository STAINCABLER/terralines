from __future__ import annotations

import base64
import os
import tempfile
import time
from unittest.mock import patch

from app.job_queue import JobQueue


def _wait_for_status(job_queue: JobQueue, job_id: str, wanted: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = job_queue.get_job(job_id)
        if job and job.get('status') == wanted:
            return job
        time.sleep(0.05)
    return job_queue.get_job(job_id)


def test_job_queue_finishes_enqueued_job():
    jq = JobQueue(worker_count=1)
    try:
        job_id = jq.enqueue(lambda x: x + 1, 2)
        job = _wait_for_status(jq, job_id, 'finished')

        assert job is not None
        assert job['status'] == 'finished'
        assert job['result'] == 3
    finally:
        jq.shutdown(wait=True)


def test_job_queue_persists_image_results_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {'TERRALINES_RESULTS_DIR': tmpdir}):
            jq = JobQueue(worker_count=1)
            try:
                img_b64 = base64.b64encode(b'fake-png-bytes').decode('utf-8')
                job_id = jq.enqueue(lambda: {'image': img_b64, 'width': 1, 'height': 1, 'time_ms': 1})
                job = _wait_for_status(jq, job_id, 'finished')

                assert job is not None
                assert job['status'] == 'finished'
                assert 'result_path' in job
                assert os.path.exists(job['result_path'])
                assert 'image_path' in job['result']
                assert 'image' not in job['result']
            finally:
                jq.shutdown(wait=True)
