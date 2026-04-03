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
import time
import threading
import secrets
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, g
from generator import (
    generate_topography,
    generate_topography_svg,
    load_templates,
    DEFAULT_PARAMS,
)


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
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MiB pro Request
app.config['SECRET_KEY'] = os.getenv('TERRALINES_SECRET_KEY', secrets.token_hex(32))

MAX_HEIGHTMAP_BYTES = 8 * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('TERRALINES_RATE_LIMIT_WINDOW_SECONDS', '60'))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv('TERRALINES_RATE_LIMIT_MAX_REQUESTS', '60'))
TRUSTED_PROXIES = [ip.strip() for ip in os.getenv('TERRALINES_TRUSTED_PROXIES', '127.0.0.1,::1').split(',') if ip.strip()]

_rate_limit_lock = threading.Lock()
_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)


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
        return _json_error(str(exc), 422)
    app.logger.exception("Fehler bei %s", route_name)
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

    try:
        result = generate_topography(params, preview=True)
        return jsonify(result)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/generate')


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

    try:
        result = generate_topography(params, preview=False)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/export')

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

    try:
        result = generate_topography(params, preview=True, heightmap_data=heightmap_bytes)
        return jsonify(result)
    except Exception as exc:
        return _handle_api_exception(exc, '/api/generate/heightmap')


@app.route('/api/presets', methods=['GET'])
def api_presets():
    """Gibt alle Presets aus dem template-Ordner zurück."""
    return jsonify(load_templates())


@app.route('/api/defaults', methods=['GET'])
def api_defaults():
    """Gibt die Standard-Parameter zurück."""
    return jsonify(DEFAULT_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    debug_enabled = os.getenv('TERRALINES_DEBUG', '0') == '1'
    mode = 'DEBUG' if debug_enabled else 'PRODUKTION'
    print("\n" + "="*55)
    print(f"  Topografie-Generator ({mode})  ·  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=debug_enabled, port=5000, threaded=True, use_reloader=debug_enabled)
