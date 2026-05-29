"""
job_queue.py — Lightweight in-process job queue
------------------------------------------------
Provides a simple queue for long-running generation tasks
without external brokers. Suitable as a short-term job queue
for small deployments (single container / single worker).

API:
 - enqueue(func, *args, **kwargs) -> job_id
 - get_job(job_id) -> dict with status/result/error

Job statuses: 'pending', 'running', 'finished', 'failed'
"""
from __future__ import annotations

import threading
import queue
import time
import uuid
import os
import pathlib
import base64
from typing import Callable, Dict


class JobQueue:
    def __init__(self, worker_count: int = 1):
        self._tasks: "queue.Queue[tuple[str, Callable, tuple, dict]]" = queue.Queue()
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._worker_count = max(1, int(worker_count))
        self._stop = threading.Event()
        # results directory for finished jobs (files). Configurable by env.
        self.results_dir = pathlib.Path(os.environ.get('TERRALINES_RESULTS_DIR', '/tmp/terralines_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Job result TTL (seconds)
        try:
            self.result_ttl = int(os.environ.get('TERRALINES_RESULT_TTL', '3600'))
        except Exception:
            self.result_ttl = 3600
        # start GC thread
        self._gc_thread = threading.Thread(target=self._gc_loop, name='job-queue-gc', daemon=True)
        self._gc_thread.start()
        for i in range(self._worker_count):
            t = threading.Thread(target=self._worker_loop, name=f"job-queue-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def enqueue(self, func: Callable, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                'status': 'pending',
                'result': None,
                'error': None,
                'created_at': time.time(),
                'started_at': None,
                'finished_at': None,
            }
        self._tasks.put((job_id, func, args, kwargs))
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                job_id, func, args, kwargs = self._tasks.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    # Shouldn't happen, but skip
                    continue
                job['status'] = 'running'
                job['started_at'] = time.time()

            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # capture exception and mark failed
                with self._lock:
                    job['status'] = 'failed'
                    job['error'] = str(exc)
                    job['finished_at'] = time.time()
            else:
                # Persist small image results to file when possible
                persisted = None
                try:
                    if isinstance(result, dict) and result.get('image'):
                        # result['image'] expected to be base64-encoded PNG
                        img_b64 = result.get('image')
                        img_bytes = base64.b64decode(img_b64)
                        fname = f"job_{job_id}.png"
                        fpath = self.results_dir / fname
                        with open(fpath, 'wb') as fh:
                            fh.write(img_bytes)
                        persisted = str(fpath)
                        # Replace image in result with path reference
                        result_copy = dict(result)
                        result_copy.pop('image', None)
                        result_copy['image_path'] = persisted
                        result = result_copy
                except Exception:
                    # If persistence fails, continue storing the raw result
                    persisted = None

                with self._lock:
                    job['status'] = 'finished'
                    job['result'] = result
                    if persisted:
                        job['result_path'] = persisted
                    job['finished_at'] = time.time()

    def _gc_loop(self):
        """Periodically delete result files older than result_ttl."""
        while not self._stop.is_set():
            try:
                now = time.time()
                for p in list(self.results_dir.glob('job_*.png')):
                    try:
                        mtime = p.stat().st_mtime
                        if now - mtime > self.result_ttl:
                            p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
            # sleep for a while
            time.sleep(60)

    def shutdown(self, wait: bool = True):
        self._stop.set()
        # wake workers
        for _ in self._workers:
            self._tasks.put((None, lambda: None, (), {}))
        if wait:
            for t in self._workers:
                t.join(timeout=1.0)


# Module-level singleton (will start workers on import)
_default_worker_count = int(__import__('os').environ.get('TERRALINES_JOB_WORKERS', '1'))
job_q = JobQueue(worker_count=_default_worker_count)

def enqueue(func: Callable, *args, **kwargs) -> str:
    return job_q.enqueue(func, *args, **kwargs)

def get_job(job_id: str) -> dict | None:
    return job_q.get_job(job_id)
