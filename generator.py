"""
generator.py — Topographic Pattern Generation Engine
======================================================
Erstellt topografische Höhenlinienmuster (Contour-Art) mithilfe von
fraktalem Brownschem Rauschen (fBm) und matplotlib-Konturdiagrammen.

Algorithmus:
    1. Erzeuge ein 2D-Höhenfeld via fBm (überlagerte gaußgeglättete Zufallsfelder)
    2. Rendere Isolinien (Contours) mit konfigurierbarem Styling via matplotlib
    3. Exportiere als Base64-kodiertes PNG

Abhängigkeiten: numpy, scipy, matplotlib
"""

from __future__ import annotations

import io
import base64
import time
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')          # Server-taugliches Non-GUI-Backend
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter


MAX_RENDER_WIDTH = 3840
MAX_RENDER_HEIGHT = 2160
MAX_RENDER_PIXELS = MAX_RENDER_WIDTH * MAX_RENDER_HEIGHT
MAX_HEIGHTMAP_PIXELS = 16_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Parameterdatenklasse — vollständige API-Dokumentation aller Einstellungen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopoParams:
    """
    Alle Einstellungsparameter für die topografische Bildgenerierung.

    Farbwerte sind CSS-Hex-Strings (#rrggbb).
    Alle numerischen Ranges sind im Frontend per Slider einstellbar.
    """

    # ── Ausgabe / Canvas ─────────────────────────────────────────────────────
    width: int          = 1920    # Bildbreite in Pixeln
    height: int         = 1080   # Bildhöhe in Pixeln
    dpi: int            = 96     # Rendering-DPI (96 = Bildschirm-Standard)
    preview_scale: float= 0.5   # Skalierungsfaktor für Vorschau (< 1 = schneller)

    # ── Farben ───────────────────────────────────────────────────────────────
    bg_color: str        = "#0d0d0d"  # Hintergrundfarbe
    bg_mode: str         = "flat"     # "flat" | "linear" | "radial"
    bg_color_2: str      = "#1a1a1a"  # Zweite Hintergrundfarbe für Verlauf
    gradient_angle: float= 135.0       # Winkel für linearen Verlauf
    line_color: str      = "#2d2d2d"  # Normale Linienfarbe
    highlight_color: str = "#3a3a3a"  # Farbe der Hervorhebunslinien
    grid_color: str      = "#ffffff"  # Farbe der Gitteroverlay-Elemente

    # ── Linienappearance ─────────────────────────────────────────────────────
    line_width: float    = 0.7    # Normlinien-Stärke (Punkte)
    highlight_width: float= 1.4   # Hervorhebungs-Linienstärke
    line_alpha: float    = 1.0    # Deckkraft (0.0–1.0)
    highlight_every: int = 5      # Jede N-te Konturlinie hervorheben (0=aus)

    # ── Linienstil ───────────────────────────────────────────────────────────
    line_style: str      = "solid"  # "solid" | "dashed" | "dotted"
    dash_every: int      = 0        # 0=aus, N= jede N-te Linie gestrichelt

    # ── Terrain / Rauschgenerator ─────────────────────────────────────────────
    levels: int          = 15     # Anzahl Konturebenen
    scale: float         = 200.0  # Rauschskala (Pixel pro Terrain-Feature)
    octaves: int         = 6      # fBm Oktaven (mehr = feiner Detail)
    persistence: float   = 0.55   # Amplitudenabfall pro Oktave (0–1)
    lacunarity: float    = 2.0    # Frequenzanstieg pro Oktave
    seed: int            = 42     # Zufalls-Seed für Reproduzierbarkeit
    smoothing: float     = 1.0    # Finaler Gaußglättungs-Faktor

    # ── Zweite Terrainschicht ────────────────────────────────────────────────
    layer2_enabled: bool = False
    layer2_scale: float  = 80.0
    layer2_weight: float = 0.35
    layer2_seed_offset: int = 1000

    # ── Gitter-Overlay ────────────────────────────────────────────────────────
    show_grid: bool      = False    # Gitterüberlagerung aktivieren
    grid_divisions: int  = 8        # Gitterzellen pro Achse
    grid_style: str      = "cross"  # "cross" | "dashes" | "lines"
    grid_alpha: float    = 0.25     # Gitter-Deckkraft
    grid_line_width: float= 0.5    # Gitterlinien-Stärke

    # ── Farbmodus ─────────────────────────────────────────────────────────────
    color_mode: str      = "flat"   # "flat" | "elevation"
    # Bei "elevation": Linienfarbe variiert von line_color → highlight_color

    @classmethod
    def from_dict(cls, d: dict) -> "TopoParams":
        """Erstellt validierte TopoParams aus einem Dictionary."""
        if not isinstance(d, dict):
            raise ValueError('Parameter müssen ein JSON-Objekt sein')

        valid_keys = cls.__dataclass_fields__.keys()
        params = cls(**{k: v for k, v in d.items() if k in valid_keys})
        _validate_topo_params(params)
        return params

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = TopoParams().to_dict()


def _validate_topo_params(p: TopoParams) -> None:
    def _int_in_range(name: str, value, minimum: int, maximum: int) -> int:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'Ungültiger Parameter: {name}')
        if ivalue < minimum or ivalue > maximum:
            raise ValueError(f'Parameter {name} muss zwischen {minimum} und {maximum} liegen')
        return ivalue

    def _float_in_range(name: str, value, minimum: float, maximum: float) -> float:
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Ungültiger Parameter: {name}')
        if fvalue < minimum or fvalue > maximum:
            raise ValueError(f'Parameter {name} muss zwischen {minimum} und {maximum} liegen')
        return fvalue

    def _one_of(name: str, value: str, allowed: set[str]) -> str:
        if value not in allowed:
            allowed_str = ', '.join(sorted(allowed))
            raise ValueError(f'Parameter {name} muss einer der Werte sein: {allowed_str}')
        return value

    p.width = _int_in_range('width', p.width, 320, MAX_RENDER_WIDTH)
    p.height = _int_in_range('height', p.height, 180, MAX_RENDER_HEIGHT)
    if p.width * p.height > MAX_RENDER_PIXELS:
        raise ValueError('Rendergröße überschreitet 4K-Limit (3840x2160)')

    p.dpi = _int_in_range('dpi', p.dpi, 72, 300)
    p.preview_scale = _float_in_range('preview_scale', p.preview_scale, 0.1, 1.0)
    p.levels = _int_in_range('levels', p.levels, 2, 128)
    p.octaves = _int_in_range('octaves', p.octaves, 1, 10)
    p.highlight_every = _int_in_range('highlight_every', p.highlight_every, 0, 64)
    p.dash_every = _int_in_range('dash_every', p.dash_every, 0, 64)
    p.grid_divisions = _int_in_range('grid_divisions', p.grid_divisions, 2, 64)
    p.seed = _int_in_range('seed', p.seed, -2_147_483_648, 2_147_483_647)

    p.scale = _float_in_range('scale', p.scale, 10.0, 2000.0)
    p.persistence = _float_in_range('persistence', p.persistence, 0.05, 0.95)
    p.lacunarity = _float_in_range('lacunarity', p.lacunarity, 1.1, 4.0)
    p.smoothing = _float_in_range('smoothing', p.smoothing, 0.0, 20.0)
    p.line_width = _float_in_range('line_width', p.line_width, 0.1, 6.0)
    p.highlight_width = _float_in_range('highlight_width', p.highlight_width, 0.1, 10.0)
    p.line_alpha = _float_in_range('line_alpha', p.line_alpha, 0.0, 1.0)
    p.grid_alpha = _float_in_range('grid_alpha', p.grid_alpha, 0.0, 1.0)
    p.grid_line_width = _float_in_range('grid_line_width', p.grid_line_width, 0.1, 5.0)
    p.gradient_angle = _float_in_range('gradient_angle', p.gradient_angle, 0.0, 360.0)
    p.layer2_scale = _float_in_range('layer2_scale', p.layer2_scale, 10.0, 2000.0)
    p.layer2_weight = _float_in_range('layer2_weight', p.layer2_weight, 0.0, 1.0)
    p.layer2_seed_offset = _int_in_range('layer2_seed_offset', p.layer2_seed_offset, -1_000_000, 1_000_000)

    p.bg_mode = _one_of('bg_mode', p.bg_mode, {'flat', 'linear', 'radial'})
    p.line_style = _one_of('line_style', p.line_style, {'solid', 'dashed', 'dotted'})
    p.grid_style = _one_of('grid_style', p.grid_style, {'cross', 'dashes', 'lines'})
    p.color_mode = _one_of('color_mode', p.color_mode, {'flat', 'elevation'})


TEMPLATE_DIR = Path(__file__).resolve().parent / 'template'


def _normalize_template_payload(template_key: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError('Template-Datei muss ein JSON-Objekt enthalten')

    raw_params = payload.get('params', payload)
    if not isinstance(raw_params, dict):
        raise ValueError('Template-Parameter müssen ein Objekt sein')

    params = TopoParams.from_dict(raw_params).to_dict()
    name = str(payload.get('name') or template_key.replace('_', ' ').title())
    chip_color = str(payload.get('chip_color') or params.get('line_color') or '#666666')

    return {
        'name': name,
        'chip_color': chip_color,
        'params': params,
    }


def load_templates() -> dict[str, dict]:
    templates: dict[str, dict] = {}
    if not TEMPLATE_DIR.exists() or not TEMPLATE_DIR.is_dir():
        return templates

    for template_file in sorted(TEMPLATE_DIR.glob('*.json')):
        key = template_file.stem
        with template_file.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        templates[key] = _normalize_template_payload(key, payload)

    return templates


# ─────────────────────────────────────────────────────────────────────────────
# Höhenfeld-Generierung (fBm via überlagerte Gauß-Felder)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_fbm(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
) -> np.ndarray:
    """
    Erzeugt ein fraktales Brownian Motion (fBm) Höhenfeld.

    Methode: Mehrere Oktaven gaußgeglätteten Zufallsrauschens werden mit
    abnehmender Amplitude und steigender Frequenz überlagert. Dies entspricht
    qualitativ Perlin/Simplex-Noise, ist aber wesentlich schneller da rein
    vektorisiert (kein Python-Loop über Pixel).

    Args:
        width, height: Feldgröße in Pixeln
        scale: Basisskala in Pixeln (größer = gröbere Strukturen)
        octaves: Anzahl Detailstufen
        persistence: Amplitudenabfall pro Oktave (0–1, typisch 0.5–0.65)
        lacunarity: Frequenzanstieg pro Oktave (typisch 2.0)
        seed: Zufalls-Seed

    Returns:
        2D numpy array (height × width), Werte normiert auf [-1, 1]
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float32)
    total_amplitude = 0.0
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        # Frisches Rauschfeld für diese Oktave
        noise_layer = rng.standard_normal((height, width)).astype(np.float32)

        # Gaußglättung entsprechend der aktuellen Frequenz
        # sigma = scale/frequency → hohe Frequenz = feine Details (kleines sigma)
        sigma = scale / frequency
        if sigma >= 0.5:
            noise_layer = gaussian_filter(noise_layer, sigma=sigma)

        result += noise_layer * amplitude
        total_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normierung auf [-1, 1]
    result /= total_amplitude
    vmax = np.abs(result).max()
    if vmax > 0:
        result /= vmax

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Gitter-Overlay Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _draw_grid_overlay(ax: plt.Axes, w: int, h: int, p: TopoParams) -> None:
    """
    Zeichnet ein Gitter-Overlay über das Bild.
    Unterstützt drei Stile: Kreuzmarker ('+'), Strichlierungen, volle Linien.
    """
    grid_c = to_rgba(p.grid_color, alpha=p.grid_alpha)

    # Gitterpositionen berechnen
    xs = np.linspace(0, w, p.grid_divisions + 1)[1:-1]  # vertikale Linien
    ys = np.linspace(0, h, p.grid_divisions + 1)[1:-1]  # horizontale Linien

    if p.grid_style == "lines":
        # Volle Gitterlinien
        for x in xs:
            ax.axvline(x, color=grid_c, linewidth=p.grid_line_width, zorder=2)
        for y in ys:
            ax.axhline(y, color=grid_c, linewidth=p.grid_line_width, zorder=2)

    elif p.grid_style == "dashes":
        # Gestrichelte Gitterlinien
        ls = (0, (8, 6))
        for x in xs:
            ax.axvline(x, color=grid_c, linewidth=p.grid_line_width,
                       linestyle=ls, zorder=2)
        for y in ys:
            ax.axhline(y, color=grid_c, linewidth=p.grid_line_width,
                       linestyle=ls, zorder=2)

    elif p.grid_style == "cross":
        # Kreuzmarker (+) an jedem Schnittpunkt
        marker_size = max(w, h) * 0.008
        for x in xs:
            for y in ys:
                ax.plot(
                    [x - marker_size, x + marker_size], [y, y],
                    color=grid_c, linewidth=p.grid_line_width * 1.5, zorder=3
                )
                ax.plot(
                    [x, x], [y - marker_size, y + marker_size],
                    color=grid_c, linewidth=p.grid_line_width * 1.5, zorder=3
                )


def _create_gradient_background(
    width: int,
    height: int,
    color1: str,
    color2: str,
    mode: str,
    angle_deg: float,
) -> np.ndarray:
    """
    Erzeugt ein RGBA-Hintergrundbild als NumPy-Array (height × width × 4).
    """
    r1, g1, b1, _ = to_rgba(color1)
    r2, g2, b2, _ = to_rgba(color2)

    if mode == "radial":
        cx, cy = width / 2, height / 2
        Y, X = np.mgrid[0:height, 0:width]
        dist = np.sqrt(((X - cx) / (width / 2))**2 + ((Y - cy) / (height / 2))**2)
        t = np.clip(dist, 0, 1)
    else:
        angle_rad = np.deg2rad(angle_deg)
        Y, X = np.mgrid[0:height, 0:width]
        nx = np.cos(angle_rad)
        ny = np.sin(angle_rad)
        proj = (X / width) * nx + (Y / height) * ny
        t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)

    bg = np.zeros((height, width, 4), dtype=np.float32)
    bg[..., 0] = r1 + t * (r2 - r1)
    bg[..., 1] = g1 + t * (g2 - g1)
    bg[..., 2] = b1 + t * (b2 - b1)
    bg[..., 3] = 1.0
    return bg


def heightmap_from_image(image_bytes: bytes, width: int, height: int) -> np.ndarray:
    """
    Konvertiert ein hochgeladenes Bild in ein normiertes Höhenfeld [-1, 1].
    """
    from PIL import Image as PILImage
    import io as _io

    try:
        PILImage.MAX_IMAGE_PIXELS = MAX_HEIGHTMAP_PIXELS
        with warnings.catch_warnings():
            warnings.simplefilter('error', PILImage.DecompressionBombWarning)
            img = PILImage.open(_io.BytesIO(image_bytes))
            img.load()
    except (PILImage.DecompressionBombWarning, PILImage.DecompressionBombError):
        raise ValueError('Heightmap ist zu groß oder nicht sicher dekodierbar')
    except Exception as exc:
        raise ValueError(f"Bild konnte nicht geöffnet werden: {exc}")

    img = img.convert('L')
    img = img.resize((width, height), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return arr


def _compute_contour_levels(terrain: np.ndarray, levels: int) -> np.ndarray:
    """
    Liefert streng steigende Konturebenen auch bei nahezu konstantem Terrain.
    """
    t_min = float(terrain.min())
    t_max = float(terrain.max())

    if not np.isfinite(t_min) or not np.isfinite(t_max):
        t_min, t_max = -1.0, 1.0

    if abs(t_max - t_min) < 1e-9:
        center = t_min
        half_span = 1e-3
        t_min = center - half_span
        t_max = center + half_span

    return np.linspace(t_min, t_max, levels + 2)[1:-1]


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Render-Funktion
# ─────────────────────────────────────────────────────────────────────────────

def generate_topography(
    params_dict: dict,
    preview: bool = True,
    heightmap_data: Optional[bytes] = None,
) -> dict:
    """
    Hauptfunktion: Erzeugt ein topografisches Muster als Base64-PNG.

    Args:
        params_dict: Parameter-Dictionary (wird in TopoParams umgewandelt)
        preview: True → skalierte Vorschau (schnell), False → volle Auflösung

    Returns:
        dict mit:
            'image'    → base64-kodierter PNG-String (ohne data:image/png;base64, Präfix)
            'width'    → tatsächliche Bildbreite
            'height'   → tatsächliche Bildhöhe
            'time_ms'  → Renderzeit in Millisekunden
    """
    t_start = time.perf_counter()

    p = TopoParams.from_dict(params_dict)

    # ── 1. Renderauflösung bestimmen ─────────────────────────────────────────
    if preview:
        scale_factor = p.preview_scale
        w = max(int(p.width  * scale_factor), 320)
        h = max(int(p.height * scale_factor), 180)
        # Rauschskala proportional anpassen
        noise_scale = p.scale * scale_factor
    else:
        w, h = p.width, p.height
        noise_scale = p.scale

    # ── 2. Höhenfeld generieren ───────────────────────────────────────────────
    if heightmap_data is not None:
        terrain = heightmap_from_image(heightmap_data, w, h)
    else:
        terrain = _generate_fbm(
            width=w,
            height=h,
            scale=noise_scale,
            octaves=max(1, min(p.octaves, 10)),
            persistence=np.clip(p.persistence, 0.05, 0.95),
            lacunarity=np.clip(p.lacunarity, 1.1, 4.0),
            seed=p.seed,
        )

    if p.layer2_enabled and p.layer2_weight > 0 and heightmap_data is None:
        terrain2 = _generate_fbm(
            width=w,
            height=h,
            scale=p.layer2_scale * (scale_factor if preview else 1.0),
            octaves=max(1, min(p.octaves, 10)),
            persistence=np.clip(p.persistence, 0.05, 0.95),
            lacunarity=np.clip(p.lacunarity, 1.1, 4.0),
            seed=p.seed + p.layer2_seed_offset,
        )
        terrain = terrain + terrain2 * p.layer2_weight
        vmax = np.abs(terrain).max()
        if vmax > 0:
            terrain /= vmax

    # Optionale Zusatzglättung des Geländes
    if p.smoothing > 0.1:
        terrain = gaussian_filter(terrain, sigma=p.smoothing)

    # ── 3. matplotlib-Figure aufsetzen ────────────────────────────────────────
    dpi = p.dpi
    fig, ax = plt.subplots(
        figsize=(w / dpi, h / dpi),
        dpi=dpi,
    )
    fig.patch.set_facecolor(p.bg_color)
    ax.set_facecolor(p.bg_color)

    if p.bg_mode in ("linear", "radial"):
        bg_array = _create_gradient_background(
            w, h, p.bg_color, p.bg_color_2, p.bg_mode, p.gradient_angle
        )
        ax.imshow(
            bg_array,
            extent=[0, w - 1, 0, h - 1],
            origin='lower',
            aspect='auto',
            zorder=0,
        )
        fig.patch.set_facecolor(p.bg_color)

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── 4. Konturebenen & Styling vorbereiten ─────────────────────────────────
    # Gleichmäßig verteilte Niveaus (ohne Extremwerte am Rand)
    contour_levels = _compute_contour_levels(terrain, p.levels)

    colors_list:      List = []
    linewidths_list:  List = []
    linestyles_list:  List = []

    for i in range(len(contour_levels)):
        # Hervorhebungs-Linie?
        is_highlight = (p.highlight_every > 0 and i % p.highlight_every == 0)

        # Farbe bestimmen
        if p.color_mode == "elevation":
            # Lineare Interpolation der Farbe entlang der Höhe
            t = i / max(len(contour_levels) - 1, 1)
            r1, g1, b1, _ = to_rgba(p.line_color)
            r2, g2, b2, _ = to_rgba(p.highlight_color)
            col = (
                r1 + t * (r2 - r1),
                g1 + t * (g2 - g1),
                b1 + t * (b2 - b1),
                p.line_alpha,
            )
        else:
            base_c  = to_rgba(p.highlight_color if is_highlight else p.line_color)
            col = (*base_c[:3], p.line_alpha)

        colors_list.append(col)
        linewidths_list.append(p.highlight_width if is_highlight else p.line_width)

        # Linienstil
        is_dashed_level = (p.dash_every > 0 and i % p.dash_every == 0)
        if p.line_style == "dashed" or is_dashed_level:
            linestyles_list.append("dashed")
        elif p.line_style == "dotted":
            linestyles_list.append("dotted")
        else:
            linestyles_list.append("solid")

    # ── 5. Konturen zeichnen (ein Aufruf für alle Ebenen = effizient) ─────────
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    ax.contour(
        X, Y, terrain,
        levels=contour_levels,
        colors=colors_list,
        linewidths=linewidths_list,
        linestyles=linestyles_list,
    )

    # ── 6. Gitter-Overlay ─────────────────────────────────────────────────────
    if p.show_grid:
        _draw_grid_overlay(ax, w, h, p)

    # ── 7. Exportieren ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(
        buf, format='png', dpi=dpi,
        bbox_inches=None,
        facecolor=p.bg_color,
        pad_inches=0,
    )
    plt.close(fig)
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    return {
        'image':    img_b64,
        'width':    w,
        'height':   h,
        'seed':     p.seed,
        'time_ms':  elapsed_ms,
    }


def generate_topography_svg(params_dict: dict) -> dict:
    """
    Wie generate_topography(), aber gibt SVG-Text zurück statt Base64-PNG.
    """
    t_start = time.perf_counter()
    p = TopoParams.from_dict(params_dict)

    w, h = p.width, p.height

    terrain = _generate_fbm(
        width=w,
        height=h,
        scale=p.scale,
        octaves=max(1, min(p.octaves, 10)),
        persistence=np.clip(p.persistence, 0.05, 0.95),
        lacunarity=np.clip(p.lacunarity, 1.1, 4.0),
        seed=p.seed,
    )

    if p.layer2_enabled and p.layer2_weight > 0:
        terrain2 = _generate_fbm(
            width=w,
            height=h,
            scale=p.layer2_scale,
            octaves=max(1, min(p.octaves, 10)),
            persistence=np.clip(p.persistence, 0.05, 0.95),
            lacunarity=np.clip(p.lacunarity, 1.1, 4.0),
            seed=p.seed + p.layer2_seed_offset,
        )
        terrain = terrain + terrain2 * p.layer2_weight
        vmax = np.abs(terrain).max()
        if vmax > 0:
            terrain /= vmax

    if p.smoothing > 0.1:
        terrain = gaussian_filter(terrain, sigma=p.smoothing)

    dpi = p.dpi
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor(p.bg_color)
    ax.set_facecolor(p.bg_color)

    if p.bg_mode in ("linear", "radial"):
        bg_array = _create_gradient_background(
            w, h, p.bg_color, p.bg_color_2, p.bg_mode, p.gradient_angle
        )
        ax.imshow(
            bg_array,
            extent=[0, w - 1, 0, h - 1],
            origin='lower',
            aspect='auto',
            zorder=0,
        )
        fig.patch.set_facecolor(p.bg_color)

    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    contour_levels = _compute_contour_levels(terrain, p.levels)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    colors_list: List = []
    linewidths_list: List = []
    linestyles_list: List = []

    for i in range(len(contour_levels)):
        is_highlight = (p.highlight_every > 0 and i % p.highlight_every == 0)
        if p.color_mode == "elevation":
            t = i / max(len(contour_levels) - 1, 1)
            r1, g1, b1, _ = to_rgba(p.line_color)
            r2, g2, b2, _ = to_rgba(p.highlight_color)
            col = (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1), p.line_alpha)
        else:
            base_c = to_rgba(p.highlight_color if is_highlight else p.line_color)
            col = (*base_c[:3], p.line_alpha)
        colors_list.append(col)
        linewidths_list.append(p.highlight_width if is_highlight else p.line_width)
        is_dashed = (p.dash_every > 0 and i % p.dash_every == 0)
        if p.line_style == "dashed" or is_dashed:
            linestyles_list.append("dashed")
        elif p.line_style == "dotted":
            linestyles_list.append("dotted")
        else:
            linestyles_list.append("solid")

    ax.contour(
        X,
        Y,
        terrain,
        levels=contour_levels,
        colors=colors_list,
        linewidths=linewidths_list,
        linestyles=linestyles_list,
    )

    if p.show_grid:
        _draw_grid_overlay(ax, w, h, p)

    svg_buf = io.StringIO()
    fig.savefig(
        svg_buf,
        format='svg',
        bbox_inches=None,
        facecolor=p.bg_color,
        pad_inches=0,
    )
    plt.close(fig)
    svg_content = svg_buf.getvalue()

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    return {
        'svg': svg_content,
        'width': w,
        'height': h,
        'seed': p.seed,
        'time_ms': elapsed_ms,
    }
