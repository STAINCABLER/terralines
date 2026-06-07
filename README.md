
# Terralines

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Flask](https://img.shields.io/badge/flask-3.0+-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

Contour art generator with a web UI. Tweak noise, colors and line styles, import a heightmap, then export PNG or SVG.


---

## What it does

Terralines generates topographic contour patterns using fractal Brownian motion noise. You get a live preview in the browser, eight presets to start from, optional heightmap input, and PNG or SVG export. The preview runs at reduced scale for speed, the export uses the full configured size.

## Requirements

- Python 3.10+
- pip
- Docker/ -Desktop if you want to run the container locally

## Project structure

```
terralines/
├── .github/              # CI workflows, actions and configs
├── .vscode/              # editor config (not all contents are tracked)
├── scripts/              # helper scripts for CI and development
├── tests/                # Unit and security regression tests
├── app/                  # Flask server, UI, templates and assets
│   └── templates/        # bundled presets (see Presets section)
├── Dockerfile            # Production multi-stage container image
├── docker-compose.yml    # Local compose setup with web, Redis and worker service
├── requirements.txt      # runtime dependencies
├── requirements-dev.txt  # dev dependencies (testing, linting, formatting)
├── README.md
└── LICENSE
```

## Setup

### Docker

GHCR Image:

```bash
docker pull ghcr.io/staincabler/ltm_ptb_terralines:latest
docker run -p 8000:8000 ghcr.io/staincabler/ltm_ptb_terralines:latest
```

Self-Build Image:

```bash
docker build -t terralines:latest .
docker run -p 8000:8000 terralines:latest
```

### Lokal

Linux Bash-Shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app/app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app/app.py
```

Open `http://127.0.0.1:8000`.

## Docker Images

The container image is published as `ghcr.io/staincabler/ltm_ptb_terralines`.

- Nightly builds run on every push and publish `nightly` plus a short SHA tag.
- Release builds are manual triggered only and can publish `latest` plus an optional full version tag.

## Bekannte Sicherheitsschwachstellen

Die folgenden Findings sind aktuell als Ausnahme dokumentiert, weil sie sich nur durch ein Base-Image update beheben lassen. Sie sind in `.trivyignore` hinterlegt, damit Trivy die CI/CD-Pipeline nicht blockiert. Stand des Eintrags: 2026-06-02.

| CVE | Betroffene Pakete | Problem | Hinzugefügt am | Voraussichtlich reparierbar |
|---|---|---|---|---|
| CVE-2025-69720 | libncursesw6, libtinfo6, ncurses-base, ncurses-bin | ncurses Buffer Overflow, potenziell Codeausführung | 2026-06-02 | 2026-06-30, sobald ein weiterer Base-Image-Refresh ein gepatchtes ncurses-Paket enthält |
| CVE-2026-42496 | perl-base | Archive::Tar kann Symlinks beim Entpacken unsicher verarbeiten | 2026-06-02 | 2026-06-30, sobald ein neueres Python/Debian-Base-Image das Fix-Paket enthält |
| CVE-2026-42497 | perl-base | Archive::Tar kann Hardlinks unsicher verarbeiten | 2026-06-02 | 2026-06-30, sobald ein neueres Python/Debian-Base-Image das Fix-Paket enthält |
| CVE-2026-48962 | perl-base | perl-IO-Compress kann unter attacker-controlled output zu Codeausführung führen | 2026-06-02 | 2026-06-30, sobald ein neueres Python/Debian-Base-Image das Fix-Paket enthält |
| CVE-2026-8376 | perl-base | Perl-Heap-Overflow beim Kompilieren | 2026-06-02 | 2026-06-30, sobald ein neueres Python/Debian-Base-Image das Fix-Paket enthält |
| CVE-2026-9538 | perl-base | Archive::Tar kann Speicher erschöpfen | 2026-06-02 | 2026-06-30, sobald ein neueres Python/Debian-Base-Image das Fix-Paket enthält |

## Containerbetrieb (Docker Compose)

### Kurzanleitung:

```powershell
# Build & start (erstes Mal / nach Code-Änderungen)
docker compose build
docker compose up -d

# Logs eines Dienstes
docker compose logs -f web

# Neustart mit Rebuild
docker compose up -d --build web
```

### Wichtiges zur Konfiguration

- `web`: Flask-Anwendung, hört auf Port `8000`. Liefert UI und API. Das Image enthält den Quellcode (kein automatisches Mount). Bei Code-Änderungen das Image neu bauen.
- `redis`: Redis-Server (RQ-Backend).
- `worker`: RQ-Worker, liest Jobs aus Redis, rendert asynchron (falls aktiviert) und schreibt Ergebnis-PNGs nach `TERRALINES_RESULTS_DIR`. Der gleiche Container kann per `TERRALINES_WORKER=true` in den Worker-Modus geschaltet werden.
- Gemeinsamer Ordner: `./tmp/terralines_results:/tmp/terralines_results` — hier schreibt der Worker persistente PNG-Dateien, die der `web`-Service ausliefert.

### Wichtige Umgebungsvariablen

- `REDIS_URL` — z.B. `redis://redis:6379/0` (für `web` und `worker`).
- `TERRALINES_RESULTS_DIR` — Verzeichnis für persistente Resultate (Container-intern: `/tmp/terralines_results`).
- `TERRALINES_MAX_CONCURRENCY` — Anzahl paralleler Generierungen (Semaphore).

**Configurable Settings**

Die folgenden Einstellungen sind zur Laufzeit konfigurierbar. Sie sind als sinnvolle Defaults im `Dockerfile` gesetzt, können aber per `docker compose` / Umgebungsvariablen oder beim Start überschrieben werden.

| Typ | Name / Label | Default | Erläuterung / Mögliche Werte |
|---:|---|---|---|
| env | `REDIS_URL` | `redis://redis:6379/0` | URL für Redis. Auf Prod: `redis://<host>:6379/0` oder Redis-Cluster-URI |
| env | `TERRALINES_SECRET_KEY` | `(none)` | Flask `SECRET_KEY` für Sessions/Signaturen; setze in Produktion per Secret/Env. Wird beim Containerstart generiert, falls unset. |
| env | `TERRALINES_RATE_LIMIT_WINDOW_SECONDS` | `60` | Zeitfenster in Sekunden für Rate-Limiting (z. B. 60) |
| env | `TERRALINES_RATE_LIMIT_MAX_REQUESTS` | `30` | Max. Anfragen pro Window (Rate-Limit), z. B. 30 |
| env | `TERRALINES_TRUSTED_PROXIES` | `127.0.0.1,::1` | Komma-separierte Liste vertrauenswürdiger Proxy-IPs/Subnetze für `X-Forwarded-*` Header |
| env | `TERRALINES_RESULTS_DIR` | `/tmp/terralines_results` | Ort, an dem Worker PNG-Dateien persistiert; sollte als Volume/Persistenter Speicher gemountet werden |
| env | `TERRALINES_WORKER` | `false` | Wenn `true`, startet der Container automatisch `rq worker terralines --url <REDIS_URL>` statt des Webservers |
| env | `TERRALINES_REDIS_WAIT_SECONDS` | `30` | Zeit in Sekunden, die der Worker beim Start auf Redis wartet, bevor er den RQ-Listener startet |
| env | `TERRALINES_JOB_WORKERS` | `1` | Anzahl der Hintergrund-Threads der internen in-process-Queue (nur relevant, wenn in-process-Queue verwendet wird) |
| env | `TERRALINES_MAX_CONCURRENCY` | `1` | Semaphore für parallele Generierungen (web Sync/Export) |
| command | Worker start command | `rq worker terralines --url redis://redis:6379/0` | Standard-RQ-Worker-Aufruf; wird vom Container über `TERRALINES_WORKER=true` automatisch gestartet |

Hinweis: Wenn du die Umgebungswerte in `docker-compose.yml` entfernst, nutzt der Container die im `Dockerfile` und im Entry-Point gesetzten Defaults. Die Volume-Mount `./tmp/terralines_results:/tmp/terralines_results` sollte bestehen bleiben, damit persistente Ergebnisse auch nach Container-Neustarts erhalten bleiben.

Worker horizontal skalieren kannst du dann mit:

```powershell
docker compose up -d --scale worker=2
```

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
| Army Land | Layered terrain with muted army/earth tones and subtle gradients |
| Cyberpunk | Dark teal/teal highlights with cross-grid accents and neon highlights |
| Dark Dimmed | Slightly lighter dark theme with softer contrast and muted lines |
| Dark Minimal | Very dark background with tight, subtle grey contours |
| Forest | Deep green background with yellow-green highlights and optional grid overlay |
| Kali Dark | Near-black theme with dense, high-count contour lines |
| Neon Lavalamp | Dark gradient background with bold, colorful lava-lamp style contours |
| Neon Pink | Dark navy background with vivid pink contour highlights |
| Cozy Sway | Warm, swaying gradient with soft highlights and subtle grid |
| Dark Forest | Deep layered forest tones with strong smoothing and detail |
| Green Lands | Bright green elevation palette with rich layering |
| Neon Sway | Neon-tinged gradient with pronounced highlights and grid accents |
| Neon Sway 2 | Radial neon gradient variant with dense contour styling |
| Red Dot | Dark background with focused red highlights and high-detail lines |
| Red Lands | Warm red/orange elevation palette with bold contours |
| Spooky | Eerie purple/pink highlights and dense, dramatic contours |

## Export

Click "PNG exportieren" to download the full-resolution PNG file or "SVG exportieren" for a vector export. The preview runs at 50% scale by default for speed and the export uses the full dimensions set in the output panel.

Supported output sizes: up to 3840x2160 px at up to 300 dpi.

## License

MIT
