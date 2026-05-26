
# Terralines

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Flask](https://img.shields.io/badge/flask-3.0+-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

Contour art generator with a web UI. Tweak noise, colors and line styles, import a heightmap, then export PNG or SVG.


---

## What it does

Terralines generates topographic contour patterns using fractal Brownian motion noise. You get a live preview in the browser, eight presets to start from, optional heightmap input, and PNG or SVG export. The preview runs at reduced scale for speed, the export uses the full configured size.

## Requirements

- Python 3.10+
- pip
- Docker Desktop if you want to build and run the container locally

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

If you want to start it again later, activate the venv first:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Alternative without activating the environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.

## Docker

The container image is published as `ghcr.io/staincabler/ltm_ptb_terralines`.

- Nightly builds run on every push and publish `nightly` plus a short SHA tag.
- Release builds are manual only and can publish `latest` plus an optional full version tag.

Local build and run:

```powershell
docker build -t ltm_ptb_terralines:local .
docker run --rm -p 8000:8000 ltm_ptb_terralines:local
```

Open `http://127.0.0.1:8000`.

## Containerbetrieb (Docker Compose)

Kurzanleitung — lokal (Produktionsnah):

```powershell
# Build & start (erstes Mal / nach Code-Änderungen)
docker compose build
docker compose up -d

# Logs eines Dienstes
docker compose logs -f web

# Neustart mit Rebuild
docker compose up -d --build web
```

Wichtiges zur Konfiguration

- `web`: Flask-Anwendung, hört auf Port `8000`. Liefert UI und API. Das Image enthält den Quellcode (kein automatisches Mount). Bei Code-Änderungen das Image neu bauen.
- `redis`: Redis-Server (RQ-Backend).
- `worker`: RQ-Worker, liest Jobs aus Redis, rendert asynchron (falls aktiviert) und schreibt Ergebnis-PNGs nach `TERRALINES_RESULTS_DIR`.
- Gemeinsamer Ordner: `./tmp/terralines_results:/tmp/terralines_results` — hier schreibt der Worker persistente PNG-Dateien, die der `web`-Service ausliefert.

Wichtige Umgebungsvariablen

- `REDIS_URL` — z.B. `redis://redis:6379/0` (für `web` und `worker`).
- `TERRALINES_RESULTS_DIR` — Verzeichnis für persistente Resultate (Container-intern: `/tmp/terralines_results`).
- `TERRALINES_MAX_CONCURRENCY` — Anzahl paralleler Generierungen (Semaphore).
- `MPLCONFIGDIR` — beschreibbares Verzeichnis für Matplotlib (z.B. `/tmp/matplotlib`).

Dev-Workflow (schnell testen)

- Lokales Development ohne Image-Rebuild: binde den Quellcode in den Container (nur für lokales Debugging):

```powershell
docker run --rm -it -p 8000:8000 -v ${PWD}:/app -v ${PWD}/tmp/terralines_results:/tmp/terralines_results \
	--env MPLCONFIGDIR=/tmp/matplotlib \
	python:3.12-slim-bookworm bash
# im Container:
pip install -r requirements.txt
python app.py
```

Production / CI

- Build image in CI and push zu einer Registry (z. B. GitHub Container Registry):

```powershell
docker build -t ghcr.io/<owner>/<repo>:<tag> .
docker push ghcr.io/<owner>/<repo>:<tag>
```

- Auf dem Server: `docker compose pull && docker compose up -d` oder orchestrieren via Kubernetes/nomad, wobei `TERRALINES_RESULTS_DIR` als persistent volume gemountet werden muss.

Wie die Container zusammenarbeiten (Kurz)

1. Das Frontend/`web` liefert UI und synchronen API-Endpunkt (`/api/generate`) sowie asynchrone Endpunkte (`/api/generate_async`, `/api/generate_rq`).
2. Für leichte Lasten bearbeitet `web` Jobs in-process (Semaphore + `job_queue.py`).
3. Für langlebige/asynchrone Jobs kann `web` Jobs in RQ/Redis enqueuen; der `worker` nimmt Jobs aus Redis, führt die Generierung durch und schreibt PNGs nach `TERRALINES_RESULTS_DIR`.
4. `web` bietet `/results/<file>` und `/api/job/<id>/download` an, die die Dateien aus `TERRALINES_RESULTS_DIR` (oder Inline-Resultate) sicher ausliefern.

Fehlerbehebung

- Wenn `/results/...` 404 liefert: prüfen, ob `./tmp/terralines_results` existiert und die Datei drin ist; prüfen Sie `docker compose logs web` und `docker compose logs worker`.
- Nach Code-Änderungen: `docker compose build web && docker compose up -d web` (weil das Image den Code enthält).


## Parameters

| Parameter | What it does |
|---|---|
| `scale` | Size of terrain features. Higher = wider, smoother hills |
| `octaves` | Detail layers in the noise. More = finer texture |
| `persistence` | How much each octave contributes. Lower = smoother |
| `lacunarity` | Frequency increase per octave |
| `levels` | Number of contour lines |
| `smoothing` | Final blur pass over the terrain |
| `seed` | Reproduces the exact same pattern |

Colors, line widths, line styles (solid/dashed/dotted), highlight intervals, grid overlays, SVG export and heightmap upload are all available in the UI.

## Presets

| Name | Description |
|---|---|
| Dark Minimal | Black background, tight grey lines |
| Dark Dimmed | Slightly lighter, softer contrast |
| Neon Pink | Dark navy with pink contours |
| Forest | Deep green with yellow-green highlights |
| Cyberpunk | Dark teal, cyan highlights, cross grid |
| Kali Dark | Near-black on black, very dense lines |

## Export

Click "PNG exportieren" to download the full-resolution PNG file or "SVG exportieren" for a vector export. The preview runs at 50% scale by default for speed and the export uses the full dimensions set in the output panel.

Supported output sizes: up to 3840x2160 px at up to 300 dpi.

## Project structure

```
terralines/
├── app.py           # Flask server, REST endpoints
├── Dockerfile       # Minimal production container image
├── generator.py     # Noise generation and rendering
├── .github/
│   ├── dependabot.yml
│   └── workflows/   # Nightly, release, functionality and security workflows
├── tests/           # Unit and security regression tests
├── static/          # Stylesheet and favicon
├── templates/       # Server-side presets
├── requirements.txt
└── index.html       # Frontend (Material You dark theme)
```

## License


MIT