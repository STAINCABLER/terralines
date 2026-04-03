
# Terralines

![Python](https://img.shields.io/badge/python-3.10+-blue) ![Flask](https://img.shields.io/badge/flask-3.0+-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

Contour art generator with a web UI. Tweak noise, colors and line styles, then export at any resolution.


---

## What it does

Terralines generates topographic contour patterns using fractal Brownian motion noise. You get a live preview in the browser, six presets to start from, and a PNG export at whatever resolution you need. Then screen, print, wallpaper.

## Requirements

- Python 3.10+
- pip

## Setup

```bash
git clone https://github.com/yourname/terralines
cd terralines
pip install -r requirements.txt
python app.py
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

Colors, line widths, line styles (solid/dashed/dotted), highlight intervals and grid overlays are all adjustable in the sidebar.

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

Click "PNG exportieren" to download the full-resolution file. The preview runs at 50% scale by default for speed and the export always uses the full dimensions set in the output panel.

Supported output sizes: up to 7680x4320 px at up to 300 dpi.

## Project structure

```
terralines/
├── app.py           # Flask server, REST endpoints
├── generator.py     # Noise generation and rendering
├── requirements.txt
└── index.html       # Frontend (Material You dark theme)
```

## License


MIT