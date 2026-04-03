
# Terralines

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Flask](https://img.shields.io/badge/flask-3.0+-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

Contour art generator with a web UI. Tweak noise, colors and line styles, import a heightmap, then export PNG or SVG.


---

## What it does

Terralines generates topographic contour patterns using fractal Brownian motion noise. You get a live preview in the browser, eight presets to start from, optional heightmap input, and PNG or SVG export. The preview runs at reduced scale for speed, the export uses the full configured size.

## Requirements

- Python 3.10+
- pip

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Wenn du das Programm später erneut startest, aktiviere zuerst die venv:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Alternative ohne Aktivierung:

```bash
git clone https://github.com/yourname/terralines
cd terralines
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.

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
├── generator.py     # Noise generation and rendering
├── static/          # Stylesheet and favicon
├── template/        # Server-side presets
├── requirements.txt
└── index.html       # Frontend (Material You dark theme)
```

## License


MIT