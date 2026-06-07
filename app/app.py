"""
app.py — Flask-Webserver für den Topografie-Generator
======================================================
Stellt eine REST-API und das HTML-Frontend bereit.

Endpunkte:
    GET  /              → Haupt-Frontend (index.html)
    POST /api/generate  → Vorschau generieren (skaliert, schnell)
    POST /api/export    → Vollauflösung exportieren
    GET  /api/presets   → Alle Presets als JSON

Starten:
    python app.py
    → http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import io
import base64
import os
import tempfile
import time
import threading
import logging
import secrets
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, g, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
TMP_DIR = Path(tempfile.gettempdir())
DEFAULT_RESULTS_DIR = TMP_DIR / 'terralines_results'

try:
    from . import job_queue as inproc_job_queue
    from .job_queue import enqueue as enqueue_job, get_job as get_job_status
    from . import tasks as rq_tasks
    from .generator import (
        generate_topography,
        generate_topography_svg,
        load_templates,
        DEFAULT_PARAMS,
    )
except ImportError:
    try:
        import job_queue as inproc_job_queue
        from job_queue import enqueue as enqueue_job, get_job as get_job_status
    except Exception:
        inproc_job_queue = None

        def enqueue_job(*_args, **_kwargs):
            raise RuntimeError('In-process queue nicht verfügbar')

        def get_job_status(_job_id):
            return None

    try:
        import tasks as rq_tasks
        from redis import Redis
        _rq_available = True
    except Exception:
        rq_tasks = None
        _rq_available = False

    from generator import (
        generate_topography,
        generate_topography_svg,
        load_templates,
        DEFAULT_PARAMS,
    )
else:
    from redis import Redis
    _rq_available = True


# ─────────────────────────────────────────────────────────────────────────────
# Flask-App Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = 'terralines'

app = Flask(
    APP_NAME,
    template_folder=str(BASE_DIR),
    static_folder=str(BASE_DIR / 'static'),
    static_url_path='/static',
)
app.logger.setLevel(logging.DEBUG)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MiB pro Request
app.config['SECRET_KEY'] = os.getenv('TERRALINES_SECRET_KEY', secrets.token_hex(32))


MAX_HEIGHTMAP_BYTES = 8 * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('TERRALINES_RATE_LIMIT_WINDOW_SECONDS', '60'))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv('TERRALINES_RATE_LIMIT_MAX_REQUESTS', '60'))
TRUSTED_PROXIES = [ip.strip() for ip in os.getenv('TERRALINES_TRUSTED_PROXIES', '127.0.0.1,::1').split(',') if ip.strip()]

_rate_limit_lock = threading.Lock()
_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)

# Concurrency control for generation tasks (default: 1 concurrent)
GENERATE_CONCURRENCY = int(os.getenv('TERRALINES_MAX_CONCURRENCY', '1'))
GENERATE_SEMAPHORE = threading.Semaphore(GENERATE_CONCURRENCY)


def _run_with_semaphore(func, *args, **kwargs):
    """Run `func` while holding the global generation semaphore."""
    GENERATE_SEMAPHORE.acquire()
    try:
        return func(*args, **kwargs)
    finally:
        try:
            GENERATE_SEMAPHORE.release()
        except Exception:
            pass


def _json_error(message: str, status: int):
    return jsonify({'error': message}), status


def _read_json_payload():
    if not request.is_json:
        return None, _json_error('Content-Type muss application/json sein', 400)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, _json_error('Ungültiger JSON-Body', 400)

    return payload, None


def _get_client_ip() -> str:
    """
    Holt die Client-IP.
    X-Forwarded-For wird nur akzeptiert, wenn der Request von einem Trusted Proxy kommt.
    Verhindert IP-Spoofing bei Rate Limiting.
    """
    if request.remote_addr in TRUSTED_PROXIES:
        forwarded_for = request.headers.get('X-Forwarded-For', '').strip()
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def _is_rate_limited(bucket_key: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        hits = _rate_limit_hits[bucket_key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            return True

        hits.append(now)
        return False


def _handle_api_exception(exc: Exception, route_name: str):
    if isinstance(exc, ValueError):
        app.logger.warning("Ungültige Eingabe bei %s: %s", route_name, exc)
        return _json_error('Ungültige Eingabe', 422)
    app.logger.error("Fehler bei %s: %s", route_name, exc)
    return _json_error('Interner Serverfehler', 500)


def _read_limited_upload(file_storage, max_bytes: int) -> bytes:
    data = file_storage.stream.read(max_bytes + 1)
    if not data:
        raise ValueError('Leere Datei hochgeladen')
    if len(data) > max_bytes:
        raise ValueError(f'Datei zu groß (max. {max_bytes // (1024 * 1024)} MB)')
    return data


@app.errorhandler(413)
def payload_too_large(_exc):
    return _json_error('Request zu groß', 413)


@app.before_request
def apply_security_headers():
    """Origin-Check für POST-Requests + Rate Limiting"""
    # Pro Request ein CSP-Nonce für erlaubte Inline-Script-Tags.
    g.csp_nonce = secrets.token_urlsafe(16)

    # CSRF-Check für teure State-Change-Operationen
    if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
        origin = request.headers.get('Origin')
        if origin:
            allowed_origin = request.host_url.rstrip('/')
            if origin != allowed_origin:
                return _json_error('Ungültige Origin (CSRF-Schutz)', 403)

    # Rate Limiting für API-Endpunkte
    if not request.path.startswith('/api/'):
        return None

    client_ip = _get_client_ip()
    bucket_key = f'{client_ip}:{request.path}'

    if _is_rate_limited(bucket_key):
        resp = jsonify({'error': 'Zu viele Anfragen, bitte später erneut versuchen'})
        resp.status_code = 429
        resp.headers['Retry-After'] = str(RATE_LIMIT_WINDOW_SECONDS)
        return resp

    return None


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    # CSP ohne unsafe-inline für Scripts und Styles
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{g.csp_nonce}' https://unpkg.com/lucide@0.419.0/dist/umd/lucide.min.js; "
        "script-src-attr 'unsafe-inline'; "
        f"style-src 'self' https://fonts.googleapis.com; "
        "style-src-attr 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Routen
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """
    Liefert das Haupt-Frontend.
    Gibt Default-Parameter als JSON an die Template weiter,
    damit das Frontend mit sinnvollen Werten initialisiert wird.
    """
    return render_template(
        'index.html',
        default_params=DEFAULT_PARAMS,
        presets=load_templates(),
        csp_nonce=g.csp_nonce,
    )


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    Generiert eine skalierte Vorschau (preview_scale des Clients).

    Erwartet: JSON-Body mit TopoParams-Feldern
    Gibt zurück: { image: base64-PNG, width, height, time_ms }
    """
    params, error = _read_json_payload()
    if error:
        return error

    # Try to serve synchronously if a concurrency slot is free; otherwise
    # inform caller to use async endpoint.
    acquired = GENERATE_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        resp = jsonify({'error': 'Server busy — zu viele parallele Generierungen', 'hint': 'Nutze /api/generate_async'}), 429
        return resp

    try:
        try:
            result = generate_topography(params, preview=True)
            return jsonify(result)
        except Exception as exc:
            return _handle_api_exception(exc, '/api/generate')
    finally:
        try:
            GENERATE_SEMAPHORE.release()
        except Exception:
            pass


@app.route('/api/export', methods=['POST'])
def api_export():
    """
    Generiert das Bild in voller Auflösung und gibt es als PNG-Datei zurück.
    Dieser Endpunkt kann je nach Parametern länger dauern (mehrere Sekunden).

    Erwartet: JSON-Body mit TopoParams-Feldern
    Gibt zurück: PNG-Datei (Content-Disposition: attachment)
    """
    params, error = _read_json_payload()
    if error:
        return error

    # For exports, behave similarly: if no slot is available, ask client to use async
    acquired = GENERATE_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        resp = jsonify({'error': 'Server busy — zu viele parallele Exporte', 'hint': 'Nutze /api/generate_async'}), 429
        return resp

    try:
        try:
            result = generate_topography(params, preview=False)
        except Exception as exc:
            return _handle_api_exception(exc, '/api/export')
    finally:
        try:
            GENERATE_SEMAPHORE.release()
        except Exception:
            pass

    # Base64 → Bytes → Flask-Response
    img_bytes = base64.b64decode(result['image'])
    buf = io.BytesIO(img_bytes)
    buf.seek(0)

    filename = f"topography_{result.get('seed', 42)}_{result['width']}x{result['height']}.png"
    return send_file(
        buf,
        mimetype='image/png',
        as_attachment=True,
        download_name=filename,
    )


@app.route('/api/export/svg', methods=['POST'])
def api_export_svg():
    """
    SVG-Export in voller Auflösung.
    SVG-Dateien können bei vielen Ebenen groß werden.
    """
    params, error = _read_json_payload()
    if error:
        return error

    try:
        result = generate_topography_svg(params)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/export/svg')

    filename = f"terralines_seed{result.get('seed', 42)}_{result['width']}x{result['height']}.svg"
    buf = io.BytesIO(result['svg'].encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf,
        mimetype='image/svg+xml',
        as_attachment=True,
        download_name=filename,
    )


@app.route('/api/generate/heightmap', methods=['POST'])
def api_generate_heightmap():
    """
    Wie /api/generate, aber akzeptiert multipart/form-data mit params + Bild.
    """
    if 'heightmap' not in request.files:
        return jsonify({'error': 'Kein Heightmap-Feld in Request'}), 400

    params_str = request.form.get('params', '{}')
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        return _json_error('Ungültiger params-JSON', 400)

    if not isinstance(params, dict):
        return _json_error('params muss ein JSON-Objekt sein', 400)

    heightmap_file = request.files['heightmap']
    if heightmap_file.mimetype and not heightmap_file.mimetype.startswith('image/'):
        return _json_error('Heightmap muss ein Bild sein', 415)

    try:
        heightmap_bytes = _read_limited_upload(heightmap_file, MAX_HEIGHTMAP_BYTES)
    except ValueError as exc:
        return _json_error(str(exc), 413)

    # Attempt synchronous handling with semaphore
    acquired = GENERATE_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        resp = jsonify({'error': 'Server busy — zu viele parallele Generierungen', 'hint': 'Nutze /api/generate_async'}), 429
        return resp

    try:
        try:
            result = generate_topography(params, preview=True, heightmap_data=heightmap_bytes)
            return jsonify(result)
        except Exception as exc:
            return _handle_api_exception(exc, '/api/generate/heightmap')
    finally:
        try:
            GENERATE_SEMAPHORE.release()
        except Exception:
            pass


@app.route('/api/generate_async', methods=['POST'])
def api_generate_async():
    """Enqueue a generation job and return a job id for polling."""
    params, error = _read_json_payload()
    if error:
        return error

    # Enqueue a wrapper that acquires the semaphore (blocks until available)
    try:
        job_id = enqueue_job(_run_with_semaphore, generate_topography, params, True)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/generate_async')

    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>', methods=['GET'])
def api_get_job(job_id: str):
    job = get_job_status(job_id)
    if job is None:
        return _json_error('Job nicht gefunden', 404)

    # Return the job status and result/error when finished.
    out = {
        'status': job.get('status'),
        'created_at': job.get('created_at'),
        'started_at': job.get('started_at'),
        'finished_at': job.get('finished_at'),
    }
    if job.get('status') == 'finished':
        out['result'] = job.get('result')
        # If a persisted result file exists, expose a download URL
        result_path = job.get('result_path') or (job.get('result') or {}).get('image_path')
        if result_path:
            out['download_url'] = url_for('api_download_job_result', job_id=job_id, _external=True)
    if job.get('status') == 'failed':
        out['error'] = job.get('error')
    return jsonify(out)


@app.route('/api/job/<job_id>/download', methods=['GET'])
def api_download_job_result(job_id: str):
    app.logger.info('api_download_job_result called for %s (method=%s)', job_id, request.method)

    def _serve_path(path: Path):
        try:
            p = path.resolve()
        except Exception:
            app.logger.warning('Invalid path for job %s: %s', job_id, path)
            return _json_error('Ungültiger Pfad', 400)

        results_dir = Path(os.environ.get('TERRALINES_RESULTS_DIR', str(DEFAULT_RESULTS_DIR))).resolve()
        if not p.is_relative_to(results_dir):
            app.logger.warning('Attempt to access file outside results dir: %s', p)
            return _json_error('Zugriff verweigert', 403)

        if not p.exists() or not p.is_file():
            return _json_error('Datei nicht gefunden', 404)

        return send_file(str(p), mimetype='image/png', as_attachment=True, download_name=p.name)

    # 1) Check in-process job queue
    job = get_job_status(job_id)
    app.logger.debug('in-process job lookup returned: %s', bool(job))
    if job:
        if job.get('status') != 'finished':
            return _json_error('Job noch nicht fertig', 409)

        result_path = (
            job.get('result_path')
            or (job.get('result') or {}).get('result_path')
            or (job.get('result') or {}).get('image_path')
        )
        # If we have a persisted file path, serve it
        if result_path:
            candidate = Path(result_path)
            app.logger.debug('In-process job result_path: %s', candidate)
            return _serve_path(candidate)

        # If the in-process job returned an inline base64 image, decode & return
        inline_image_b64 = (job.get('result') or {}).get('image')
        if inline_image_b64:
            try:
                img_bytes = base64.b64decode(inline_image_b64)
                buf = io.BytesIO(img_bytes)
                buf.seek(0)
                return send_file(buf, mimetype='image/png', as_attachment=True, download_name=f'job_{job_id}.png')
            except Exception as exc:
                app.logger.exception('Failed to decode inline image for job %s: %s', job_id, exc)
                return _json_error('Ergebnis nicht verfügbar', 500)

        return _json_error('Kein Ergebnis verfügbar', 404)

    # 2) Fallback: direct shared file (job_<id>.png)
    results_dir = Path(os.environ.get('TERRALINES_RESULTS_DIR', str(DEFAULT_RESULTS_DIR)))
    candidate = results_dir / f"job_{job_id}.png"
    app.logger.debug('Download fallback check shared file: %s', candidate)
    if candidate.exists():
        return _serve_path(candidate)

    # 3) Fallback: RQ/Redis job
    if _rq_available:
        try:
            redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
            conn = Redis.from_url(redis_url)
            from rq.job import Job
            rq_job = Job.fetch(job_id, connection=conn)
            if rq_job is None:
                app.logger.debug('RQ job fetch returned None for %s', job_id)
                return _json_error('Job nicht gefunden', 404)

            rq_status = rq_job.get_status()
            app.logger.debug('RQ job %s status=%s result=%s', job_id, rq_status, type(rq_job.result))

            # If RQ task wrote a file path within results dir, serve it
            rp = (rq_job.result or {}).get('result_path') if isinstance(rq_job.result, dict) else None
            if rp:
                candidate2 = Path(rp)
                app.logger.debug('RQ job result_path: %s', candidate2)
                if candidate2.exists():
                    return _serve_path(candidate2)

            # If RQ returned inline base64 image in result dict
            inline_b64 = (rq_job.result or {}).get('image') if isinstance(rq_job.result, dict) else None
            if inline_b64:
                try:
                    img_bytes = base64.b64decode(inline_b64)
                    buf = io.BytesIO(img_bytes)
                    buf.seek(0)
                    return send_file(buf, mimetype='image/png', as_attachment=True, download_name=f'job_{job_id}.png')
                except Exception:
                    app.logger.exception('Failed to decode RQ inline image for %s', job_id)
                    return _json_error('Ergebnis nicht verfügbar', 500)

            return _json_error('Job nicht gefunden', 404)
        except Exception as exc:
            app.logger.exception('RQ fallback failed for job %s: %s', job_id, exc)
            return _json_error('Job nicht gefunden', 404)

    return _json_error('Job nicht gefunden', 404)


@app.route('/api/queue_status', methods=['GET'])
def api_queue_status():
    """Gibt Zahlen zur in-process-Queue zurück: queued, running, finished."""
    if inproc_job_queue is None:
        return jsonify({'error': 'Queue nicht verfügbar'}), 503

    jobs = inproc_job_queue.job_q

    with jobs._lock:
        counts = {'queued': jobs._tasks.qsize()}
        running = sum(1 for j in jobs._jobs.values() if j.get('status') == 'running')
        finished = sum(1 for j in jobs._jobs.values() if j.get('status') == 'finished')
    counts.update({'running': running, 'finished': finished})
    return jsonify(counts)


@app.route('/api/presets', methods=['GET'])
def api_presets():
    """Gibt alle Presets aus dem templates-Ordner zurück."""
    return jsonify(load_templates())


@app.route('/api/generate_rq', methods=['POST'])
def api_generate_rq():
    if not _rq_available:
        return _json_error('RQ/Redis nicht konfiguriert', 503)
    params, error = _read_json_payload()
    if error:
        return error

    try:
        job_id = rq_tasks.enqueue_preview(params)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/generate_rq')

    status_url = url_for('api_rq_job_status', job_id=job_id, _external=True)
    return jsonify({'job_id': job_id, 'status_url': status_url})


@app.route('/api/rq/job/<job_id>', methods=['GET'])
def api_rq_job_status(job_id: str):
    if not _rq_available:
        return _json_error('RQ/Redis nicht konfiguriert', 503)
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
        conn = Redis.from_url(redis_url)
        from rq.job import Job
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return _json_error('Job nicht gefunden oder Redis nicht erreichbar', 404)

    status = job.get_status()
    out = {'job_id': job.id, 'status': status}
    if status == 'finished':
        out['result'] = job.result
    if status == 'failed':
        out['error'] = str(job.exc_info)
    return jsonify(out)


@app.route('/api/defaults', methods=['GET'])
def api_defaults():
    """Gibt die Standard-Parameter zurück."""
    return jsonify(DEFAULT_PARAMS)


@app.route('/health', methods=['GET'])
def health():
    """Simple runtime health endpoint for container probes and monitoring."""
    worker_mode = str(os.getenv('TERRALINES_WORKER', 'false')).lower() in {'1', 'true', 'yes', 'on'}
    return jsonify({'status': 'ok', 'worker_mode': worker_mode})


@app.route('/results/<path:filename>', methods=['GET'])
def serve_result_file(filename: str):
    """Serve persisted result files from the shared results directory.

    Example: /results/job_<id>.png
    """
    app.logger.info('serve_result_file called for %s', filename)
    results_dir = os.environ.get('TERRALINES_RESULTS_DIR', str(DEFAULT_RESULTS_DIR))
    candidate = Path(results_dir) / filename
    app.logger.debug('serve_result_file candidate=%s exists=%s', candidate, candidate.exists())
    if not candidate.exists():
        return _json_error('Datei nicht gefunden', 404)
    try:
        return send_file(str(candidate), mimetype='image/png', as_attachment=True, download_name=filename)
    except FileNotFoundError:
        return _json_error('Datei nicht gefunden', 404)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    debug_enabled = os.getenv('TERRALINES_DEBUG', '0') == '1'
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    mode = 'DEBUG' if debug_enabled else 'PRODUKTION'
    print("\n" + "="*55)
    print(f"  Topografie-Generator ({mode})  ·  http://{host}:{port}")
    print("="*55 + "\n")
    app.run(host=host, debug=debug_enabled, port=port, threaded=True, use_reloader=debug_enabled)
