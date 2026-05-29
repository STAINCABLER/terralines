"""
tasks.py — RQ task wrappers for Topografie-Generator

This module exposes functions that can be enqueued by RQ workers.
Workers import this module and execute jobs in a separate process.
"""
import os
import base64
import tempfile
from pathlib import Path
from redis import Redis
from rq import Queue
import matplotlib
matplotlib.use('Agg')

try:
    from .generator import generate_topography
except ImportError:
    from generator import generate_topography
from rq import get_current_job

# Results directory for persisted previews
RESULTS_DIR = Path(os.environ.get('TERRALINES_RESULTS_DIR', str(Path(tempfile.gettempdir()) / 'terralines_results')))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_queue():
    _redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    conn = Redis.from_url(_redis_url)
    return Queue('terralines', connection=conn)


def enqueue_preview(params: dict) -> str:
    """Enqueue a preview generation; returns RQ job id."""
    q = _get_queue()
    # Enqueue by import path so worker processes can import the callable
    job = q.enqueue('tasks._run_generate_preview', params)
    return job.id


def _run_generate_preview(params: dict) -> dict:
    """Worker-executed function that returns the result dict from generator."""
    result = generate_topography(params, preview=True)
    job = get_current_job()
    if job is not None and 'image' in result:
        img_b64 = result['image']
        img_bytes = base64.b64decode(img_b64)
        out_path = RESULTS_DIR / f"job_{job.id}.png"
        with out_path.open('wb') as fh:
            fh.write(img_bytes)
        result['result_path'] = str(out_path)
    return result


def enqueue_export(params: dict) -> str:
    q = _get_queue()
    job = q.enqueue('tasks._run_generate_export', params)
    return job.id


def _run_generate_export(params: dict) -> dict:
    result = generate_topography(params, preview=False)
    return result
