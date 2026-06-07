/* ===========================================================================
   ZUSTANDSVERWALTUNG
   =========================================================================== */

/** Aktueller Parametersatz – wird aus Flask-Template initialisiert */
let params = window.DEFAULT_PARAMS ? { ...window.DEFAULT_PARAMS } : {};

/** Alle Presets vom Server */
let SERVER_TEMPLATES = window.SERVER_TEMPLATES || {};

/** Debounce-Timer für Auto-Generate */
let autoGenTimer = null;
const AUTO_GEN_DELAY_MS = 900;  // ms nach letzter Änderung

/** Animations-Handle für den Template-Strip */
let templatesScrollAnimationFrame = null;

/** Merkt ob gerade ein Request läuft */
let isGenerating = false;

/** Terrain-Quelle und optionaler Upload-Inhalt */
let currentHeightmapBytes = null;
let terrainSource = 'noise';

/** Zoom/Pan-Zustand */
const viewport = {
  scale: 1.0,
  x: 0,
  y: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  dragStartVX: 0,
  dragStartVY: 0,
};
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 8.0;
const ZOOM_STEP = 0.12;

const LOCALSTORAGE_KEY = 'terralines_user_presets';


/* ===========================================================================
   URL-HASH SERIALISIERUNG
   =========================================================================== */

function saveParamsToHash() {
  const json = JSON.stringify(params);
  const encoded = btoa(unescape(encodeURIComponent(json)));
  history.replaceState(null, '', '#' + encoded);
}

function loadParamsFromHash() {
  const hash = window.location.hash.slice(1);
  if (!hash) return null;
  try {
    const json = decodeURIComponent(escape(atob(hash)));
    return JSON.parse(json);
  } catch (e) {
    console.warn('URL-Hash ungültig, ignoriert.', e);
    return null;
  }
}


/* ===========================================================================
   SLIDER-TRACKING (visuelles Füllgrad-Update)
   =========================================================================== */

/**
 * Aktualisiert den Slider-Hintergrund basierend auf dem aktuellen Wert.
 * Erzeugt einen "gefüllten" Look entsprechend M3-Spec.
 */
function updateSliderTrack(sliderId) {
  const slider = document.getElementById(sliderId);
  if (!slider) return;
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right,
    var(--md-sys-color-primary) ${pct}%,
    var(--md-sys-color-surface-container-highest) ${pct}%)`;
}

/** Initialisiert alle Slider-Tracks beim Laden */
function initSliderTracks() {
  document.querySelectorAll('.m3-slider').forEach(s => {
    updateSliderTrack(s.id);
  });
}


/* ===========================================================================
   PARAMETER-HANDLER
   =========================================================================== */

/**
 * Verarbeitet Slider-Änderung: aktualisiert params, zugehöriges Zahleninput
 * und schiebt Auto-Generate an.
 *
 * @param {string} key       - Parameter-Key
 * @param {string} rawValue  - Rohwert als String
 * @param {number} decimals  - Anzahl Dezimalstellen für Anzeige
 */
function handleSlider(key, rawValue, decimals = 0) {
  const val = decimals > 0 ? parseFloat(rawValue) : parseInt(rawValue, 10);
  params[key] = val;

  // Zahleninput synchronisieren
  const numInput = document.getElementById(`in-${key}`);
  if (numInput) numInput.value = val.toFixed(decimals);

  // Slider-Farbe aktualisieren
  updateSliderTrack(`sl-${key}`);

  scheduleAutoGenerate();
}

/**
 * Verarbeitet direkte Zahleingaben: synchronisiert Slider und params.
 */
function handleInput(key, rawValue) {
  const slider = document.getElementById(`sl-${key}`);
  const isFloat = slider && (parseFloat(slider.step) < 1);
  const val = isFloat ? parseFloat(rawValue) : parseInt(rawValue, 10);

  params[key] = val;

  if (slider) {
    slider.value = val;
    updateSliderTrack(`sl-${key}`);
  }

  scheduleAutoGenerate();
}

/**
 * Verarbeitet Select-Änderungen.
 */
function handleSelect(key, value) {
  params[key] = value;
  scheduleAutoGenerate();
}

/**
 * Verarbeitet Color-Picker Änderungen: aktualisiert Swatch, Hex-Input, params.
 */
function handleColorChange(key, hexValue) {
  params[key] = hexValue;

  // Swatch-Hintergrund aktualisieren
  const swatch = document.getElementById(`swatch-${key}`);
  if (swatch) swatch.style.backgroundColor = hexValue;

  // Hex-Textfeld synchronisieren
  const hexInput = document.getElementById(`hex-${key}`);
  if (hexInput) hexInput.value = hexValue;

  scheduleAutoGenerate();
}

/**
 * Verarbeitet manuelle Hex-Eingabe im Textfeld.
 */
function handleHexInput(key, hexValue) {
  // Nur gültige Hex-Farben akzeptieren
  if (!/^#[0-9a-fA-F]{6}$/.test(hexValue)) return;
  params[key] = hexValue;

  const picker = document.getElementById(`picker-${key}`);
  if (picker) picker.value = hexValue;

  const swatch = document.getElementById(`swatch-${key}`);
  if (swatch) swatch.style.backgroundColor = hexValue;

  scheduleAutoGenerate();
}

/**
 * Verarbeitet Toggle-Switch Klicks.
 */
function handleSwitchChange(key, checked) {
  params[key] = !!checked;
  const track = document.getElementById(`sw-${key}`);
  if (track) {
    if (params[key]) track.classList.add('checked');
    else track.classList.remove('checked');
  }
  scheduleAutoGenerate();
}

function toggleSwitch(key) {
  const checkbox = document.getElementById(`cb-${key}`);
  if (!checkbox) return;
  checkbox.checked = !checkbox.checked;
  handleSwitchChange(key, checkbox.checked);
}


/* ===========================================================================
   PRESETS
   =========================================================================== */

/**
 * Wendet ein Preset an: merged Preset-Werte in params und aktualisiert die UI.
 */
function applyPreset(presetKey) {
  const template = SERVER_TEMPLATES[presetKey];
  if (!template || !template.params) return;

  // Preset-Werte in params einfügen (rest behält Defaults)
  Object.assign(params, template.params);

  // UI synchronisieren
  syncUIToParams();

  // Aktiven Chip markieren
  document.querySelectorAll('.chip.preset-item').forEach(c => c.classList.remove('active'));
  const activeChip = document.getElementById(`chip-server-${presetKey}`);
  if (activeChip) activeChip.classList.add('active');

  // Sofort generieren (kein Debounce)
  generatePreview();
}

/**
 * Synchronisiert alle UI-Elemente mit dem aktuellen params-Objekt.
 * Wird nach Preset-Anwendung aufgerufen.
 */
function syncUIToParams() {
  // Slider + Zahleninputs
  const sliderParams = [
    ['line_width', 1], ['highlight_width', 1], ['highlight_every', 0],
    ['line_alpha', 2], ['dash_every', 0],
    ['levels', 0], ['scale', 0], ['octaves', 0],
    ['persistence', 2], ['lacunarity', 1], ['smoothing', 1], ['seed', 0],
    ['grid_divisions', 0], ['grid_alpha', 2], ['grid_line_width', 1],
    ['width', 0], ['height', 0],
    ['gradient_angle', 0],
    ['layer2_scale', 0], ['layer2_weight', 2], ['layer2_seed_offset', 0],
  ];
  sliderParams.forEach(([key, dec]) => {
    const val = params[key];
    if (val === undefined) return;
    const slider = document.getElementById(`sl-${key}`);
    if (slider) { slider.value = val; updateSliderTrack(`sl-${key}`); }
    const numIn = document.getElementById(`in-${key}`);
    if (numIn) numIn.value = typeof val === 'number' ? val.toFixed(dec) : val;
  });

  // Farbfelder
  ['bg_color', 'line_color', 'highlight_color', 'grid_color', 'bg_color_2'].forEach(key => {
    const val = params[key];
    if (!val) return;
    const picker = document.getElementById(`picker-${key}`);
    if (picker) picker.value = val;
    const hex = document.getElementById(`hex-${key}`);
    if (hex) hex.value = val;
    const swatch = document.getElementById(`swatch-${key}`);
    if (swatch) swatch.style.backgroundColor = val;
  });

  // Selects
  ['line_style', 'color_mode', 'grid_style', 'bg_mode'].forEach(key => {
    const sel = document.getElementById(`sel-${key}`);
    if (sel && params[key] !== undefined) sel.value = params[key];
  });

  const sourceSel = document.getElementById('sel-terrain-source');
  if (sourceSel) sourceSel.value = terrainSource;

  const bgModeSel = document.getElementById('sel-bg_mode');
  if (bgModeSel) bgModeSel.value = params.bg_mode || 'flat';
  toggleGradientControls();
  // DPI select
  const dpiSel = document.getElementById('sel-dpi');
  if (dpiSel && params.dpi) dpiSel.value = params.dpi;
  updatePreviewScaleControl();

  // Toggle
  const gridSwitch = document.getElementById('sw-show_grid');
  const gridCheckbox = document.getElementById('cb-show_grid');
  if (gridSwitch) {
    if (params.show_grid) gridSwitch.classList.add('checked');
    else gridSwitch.classList.remove('checked');
  }
  if (gridCheckbox) gridCheckbox.checked = !!params.show_grid;

  const l2sw = document.getElementById('sw-layer2_enabled');
  const l2cb = document.getElementById('cb-layer2_enabled');
  if (l2sw) {
    if (params.layer2_enabled) l2sw.classList.add('checked');
    else l2sw.classList.remove('checked');
  }
  if (l2cb) l2cb.checked = !!params.layer2_enabled;
  toggleLayer2Controls();
}


/* ===========================================================================
   GENERIERUNG
   =========================================================================== */

/** Plant eine Auto-Generierung nach Debounce-Delay. */
function scheduleAutoGenerate() {
  if (autoGenTimer) clearTimeout(autoGenTimer);
  saveParamsToHash();
  autoGenTimer = setTimeout(() => generatePreview(), AUTO_GEN_DELAY_MS);
}

function getPreviewScaleLabel(value) {
  const scale = Math.max(0.1, Math.min(1.0, Number(value) || 0.7));
  return `${Math.round(scale * 100)}%`;
}

function updatePreviewScaleControl() {
  const label = document.getElementById('preview-scale-label');
  const current = params.preview_scale ?? 0.7;
  if (label) label.textContent = getPreviewScaleLabel(current);

  const menu = document.getElementById('preview-scale-menu');
  if (menu) {
    menu.querySelectorAll('.split-dropdown-item').forEach((item) => {
      const itemScale = parseFloat(item.dataset.previewScale || '0');
      item.classList.toggle('active', Math.abs(itemScale - current) < 0.001);
    });
  }
}

function closePreviewScaleMenu() {
  const menu = document.getElementById('preview-scale-menu');
  const toggle = document.getElementById('btn-preview-scale-toggle');
  const group = document.getElementById('preview-scale-split-group');
  if (menu) menu.hidden = true;
  if (toggle) toggle.setAttribute('aria-expanded', 'false');
  if (group) {
    group.classList.remove('menu-up', 'menu-down');
  }
}

function openPreviewScaleMenu() {
  const menu = document.getElementById('preview-scale-menu');
  const toggle = document.getElementById('btn-preview-scale-toggle');
  const group = document.getElementById('preview-scale-split-group');
  if (!menu || !toggle || !group) return;

  menu.hidden = false;
  const menuHeight = menu.scrollHeight || 220;
  const toggleRect = toggle.getBoundingClientRect();
  const spaceBelow = window.innerHeight - toggleRect.bottom;
  const spaceAbove = toggleRect.top;
  const openDown = spaceBelow >= menuHeight || spaceBelow >= spaceAbove;

  group.classList.toggle('menu-down', openDown);
  group.classList.toggle('menu-up', !openDown);
  toggle.setAttribute('aria-expanded', 'true');
}

function togglePreviewScaleMenu(forceOpen = null) {
  const menu = document.getElementById('preview-scale-menu');
  if (!menu) return;
  const shouldOpen = forceOpen === null ? menu.hidden : !!forceOpen;
  if (shouldOpen) openPreviewScaleMenu();
  else closePreviewScaleMenu();
}

function applyPreviewScale(value) {
  const next = Math.max(0.1, Math.min(1.0, parseFloat(value)));
  if (Number.isNaN(next)) return;
  params.preview_scale = next;
  saveParamsToHash();
  updatePreviewScaleControl();
  closePreviewScaleMenu();
}

/** Generiert eine Vorschau via POST /api/generate. */
async function generatePreview() {
  if (terrainSource === 'heightmap' && currentHeightmapBytes !== null) {
    await generateWithHeightmap();
    return;
  }

  if (isGenerating) return;
  if (autoGenTimer) { clearTimeout(autoGenTimer); autoGenTimer = null; }

  isGenerating = true;
  setLoading(true, 'Generiere Vorschau…');
  document.getElementById('btn-generate').disabled = true;

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    displayImage(data);

  } catch (err) {
    showSnackbar(`Fehler: ${err.message}`);
    console.error('Generate error:', err);
  } finally {
    isGenerating = false;
    setLoading(false);
    document.getElementById('btn-generate').disabled = false;
  }
}

/** Exportiert das Bild in voller Auflösung. */
async function exportImage() {
  setLoading(true, `Exportiere ${params.width}×${params.height}px…`);
  document.getElementById('btn-export').disabled = true;

  try {
    const response = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    // Blob herunterladen
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `topography_seed${params.seed}_${params.width}x${params.height}.png`;
    a.click();
    URL.revokeObjectURL(url);

    showSnackbar(`✓ Exportiert: ${params.width}×${params.height} @ ${params.dpi} dpi`);

  } catch (err) {
    showSnackbar(`Export-Fehler: ${err.message}`);
    console.error('Export error:', err);
  } finally {
    setLoading(false);
    document.getElementById('btn-export').disabled = false;
  }
}

async function exportSVG() {
  setLoading(true, `Exportiere SVG ${params.width}×${params.height}px…`);
  document.getElementById('btn-export-svg').disabled = true;

  try {
    const response = await fetch('/api/export/svg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terralines_seed${params.seed}_${params.width}x${params.height}.svg`;
    a.click();
    URL.revokeObjectURL(url);
    showSnackbar('✓ SVG exportiert — kann groß sein (5–30 MB normal)');
  } catch (err) {
    showSnackbar(`SVG-Fehler: ${err.message}`);
  } finally {
    setLoading(false);
    document.getElementById('btn-export-svg').disabled = false;
  }
}


/* ===========================================================================
   UI-HELFER
   =========================================================================== */

/** Zeigt das generierte Bild im Vorschaubereich an. */
function displayImage(data) {
  const skeleton = document.getElementById('preview-skeleton');
  const img = document.getElementById('preview-img');
  const statusbar = document.getElementById('statusbar');

  skeleton.style.display = 'none';
  img.src = `data:image/png;base64,${data.image}`;
  img.style.display = 'block';
  resetViewport();
  updateCenterPreviewButtonState();

  // Statusleiste aktualisieren
  statusbar.style.display = 'flex';
  document.getElementById('status-res').textContent = `${data.width} × ${data.height} px`;
  document.getElementById('status-time').textContent = `${data.time_ms} ms`;
  document.getElementById('status-seed').textContent = params.seed;
}

/** Setzt Ladeoverlay sichtbar/unsichtbar. */
function setLoading(visible, text = '') {
  const overlay = document.getElementById('loading-overlay');
  const loadText = document.getElementById('loading-text');
  if (visible) {
    overlay.classList.add('visible');
    if (text) loadText.textContent = text;
  } else {
    overlay.classList.remove('visible');
  }
}

/** Zeigt einen M3-Snackbar-Toast. */
function showSnackbar(message, durationMs = 3000) {
  const sb = document.getElementById('snackbar');
  sb.textContent = message;
  sb.classList.add('visible');
  setTimeout(() => sb.classList.remove('visible'), durationMs);
}

/** Würfelt einen neuen Zufalls-Seed. */
function randomizeSeed() {
  playDiceAnimation('dice-seed');
  const newSeed = Math.floor(Math.random() * 9999);
  params.seed = newSeed;
  document.getElementById('sl-seed').value = newSeed;
  document.getElementById('in-seed').value = newSeed;
  updateSliderTrack('sl-seed');

  // Aktive Chip-Markierung aufheben (kein Standard-Preset mehr)
  document.querySelectorAll('.chip.preset-item').forEach(c => c.classList.remove('active'));

  scheduleAutoGenerate();
}

function randomizeAll() {
  playDiceAnimation('dice-all');
  const r = (min, max, decimals = 0) => {
    const val = Math.random() * (max - min) + min;
    return decimals > 0 ? parseFloat(val.toFixed(decimals)) : Math.round(val);
  };

  const hslToHex = (h, s, l) => {
    const a = s * Math.min(l, 100 - l) / 10000;
    const f = n => {
      const k = (n + h / 30) % 12;
      const color = l / 100 - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  };

  const baseHue = r(0, 360);
  const accentHue = (baseHue + r(30, 180)) % 360;

  params.seed = r(0, 9999);
  params.scale = r(80, 600);
  params.octaves = r(2, 9);
  params.persistence = r(0.3, 0.8, 2);
  params.lacunarity = r(1.2, 3.5, 1);
  params.smoothing = r(0, 3, 1);
  params.levels = r(6, 30);
  params.line_width = r(0.4, 2.5, 1);
  params.highlight_width = params.line_width + r(0.3, 1.5, 1);
  params.highlight_every = r(3, 10);
  params.line_alpha = r(0.6, 1.0, 2);
  params.dash_every = Math.random() > 0.7 ? r(3, 10) : 0;

  params.bg_color = hslToHex(baseHue, r(0, 20), r(3, 12));
  params.bg_color_2 = hslToHex((baseHue + r(10, 60)) % 360, r(10, 30), r(8, 18));
  params.line_color = hslToHex(baseHue, r(10, 40), r(15, 35));
  params.highlight_color = hslToHex(accentHue, r(40, 90), r(40, 70));

  params.bg_mode = ['flat', 'linear', 'radial'][r(0, 2)];
  params.gradient_angle = r(0, 360);

  params.show_grid = Math.random() > 0.7;
  if (params.show_grid) {
    params.grid_style = ['cross', 'dashes', 'lines'][r(0, 2)];
    params.grid_divisions = r(4, 16);
    params.grid_alpha = r(0.1, 0.5, 2);
    params.grid_color = hslToHex(accentHue, r(20, 60), r(50, 85));
  }

  params.layer2_enabled = Math.random() > 0.65;
  params.layer2_scale = r(10, 400);
  params.layer2_weight = r(0.05, 1.0, 2);
  params.layer2_seed_offset = r(1, 5000);

  document.querySelectorAll('.chip.preset-item').forEach(c => c.classList.remove('active'));

  syncUIToParams();
  saveParamsToHash();
  generatePreview();
}

function toggleGradientControls() {
  const mode = params.bg_mode || 'flat';
  const controls = document.getElementById('gradient-controls');
  const angleRow = document.getElementById('gradient-angle-row');
  if (controls) controls.style.display = (mode === 'flat') ? 'none' : 'flex';
  if (angleRow) angleRow.style.display = (mode === 'radial') ? 'none' : 'flex';
}

function toggleLayer2Controls() {
  const el = document.getElementById('layer2-controls');
  if (el) el.style.display = params.layer2_enabled ? 'flex' : 'none';
}

function handleTerrainSourceChange(value) {
  terrainSource = value;
  const uploadArea = document.getElementById('heightmap-upload-area');
  if (uploadArea) uploadArea.style.display = value === 'heightmap' ? 'block' : 'none';
  if (value === 'noise') {
    currentHeightmapBytes = null;
    scheduleAutoGenerate();
  }
}

function handleHeightmapUpload(input) {
  const file = input.files && input.files[0];
  if (!file) return;

  handleHeightmapFile(file);
}

function handleHeightmapFile(file) {
  if (!file) return;

  document.getElementById('heightmap-filename').textContent = file.name;
  const reader = new FileReader();
  reader.onload = (e) => {
    currentHeightmapBytes = e.target.result;
    generateWithHeightmap();
  };
  reader.readAsArrayBuffer(file);
}

async function generateWithHeightmap() {
  if (!currentHeightmapBytes || isGenerating) return;
  isGenerating = true;
  setLoading(true, 'Verarbeite Heightmap…');
  document.getElementById('btn-generate').disabled = true;

  const formData = new FormData();
  formData.append('params', JSON.stringify(params));
  formData.append('heightmap', new Blob([currentHeightmapBytes]), 'heightmap.png');

  try {
    const response = await fetch('/api/generate/heightmap', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      let msg = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        if (err && err.error) msg = err.error;
      } catch (_) {}
      throw new Error(msg);
    }
    const data = await response.json();
    displayImage(data);
  } catch (err) {
    showSnackbar(`Heightmap-Fehler: ${err.message}`);
  } finally {
    isGenerating = false;
    setLoading(false);
    document.getElementById('btn-generate').disabled = false;
  }
}

function applyViewportTransform() {
  const img = document.getElementById('preview-img');
  if (!img) return;
  img.style.transform = `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`;
  img.style.transformOrigin = 'center center';
  updateCenterPreviewButtonState();
}

function resetViewport() {
  viewport.scale = 1.0;
  viewport.x = 0;
  viewport.y = 0;
  applyViewportTransform();
}

function centerPreview() {
  resetViewport();
}

function hasViewportChanged() {
  return Math.abs(viewport.scale - 1.0) > 0.001 || Math.abs(viewport.x) > 0.5 || Math.abs(viewport.y) > 0.5;
}

function updateCenterPreviewButtonState() {
  const button = document.getElementById('btn-center-preview');
  if (!button) return;
  const img = document.getElementById('preview-img');
  const enabled = !!img && img.style.display !== 'none' && hasViewportChanged();
  button.disabled = !enabled;
}

function loadUserPresets() {
  try {
    return JSON.parse(localStorage.getItem(LOCALSTORAGE_KEY) || '{}');
  } catch (_) {
    return {};
  }
}

function saveUserPresets(presets) {
  localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(presets));
}

function saveCurrentAsPreset() {
  const name = prompt('Preset-Name:');
  if (!name || !name.trim()) return;

  const userPresets = loadUserPresets();
  userPresets[name.trim()] = { ...params };
  saveUserPresets(userPresets);
  renderPresetList();
  showSnackbar(`Preset "${name.trim()}" gespeichert`);
}

function exportCurrentSettings() {
  const defaultName = `terralines_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}`;
  const name = (prompt('Name für den Export:', defaultName) || defaultName).trim();
  const payload = {
    name,
    chip_color: params.line_color || '#666666',
    params: { ...params },
  };

  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name.replace(/[^a-z0-9_-]+/gi, '_').toLowerCase()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function applyPresetFromObject(name, preset) {
  Object.assign(params, preset);
  syncUIToParams();
  document.querySelectorAll('.chip.preset-item').forEach(c => c.classList.remove('active'));
  const chip = document.getElementById(`chip-user-${name}`);
  if (chip) chip.classList.add('active');
  saveParamsToHash();
  generatePreview();
}

function renderServerTemplateList(keepCurrentScroll = true) {
  const container = document.getElementById('server-preset-chips');
  if (!container) return;
  container.innerHTML = '';

  const scrollArea = document.getElementById('templates-scroll');
  const preserveScrollLeft = keepCurrentScroll && scrollArea ? scrollArea.scrollLeft : 0;

  const getGhostBorderColor = (hexColor) => {
    const hex = String(hexColor || '').replace('#', '').trim();
    if (!/^[0-9a-fA-F]{6}$/.test(hex)) {
      return 'rgba(240, 240, 240, 0.14)';
    }

    const red = parseInt(hex.slice(0, 2), 16);
    const green = parseInt(hex.slice(2, 4), 16);
    const blue = parseInt(hex.slice(4, 6), 16);
    const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;

    if (luminance < 0.28) return 'rgba(244, 244, 244, 0.34)';
    if (luminance < 0.45) return 'rgba(244, 244, 244, 0.22)';
    return 'rgba(244, 244, 244, 0.12)';
  };

  Object.entries(SERVER_TEMPLATES).forEach(([key, tpl]) => {
    const chip = document.createElement('button');
    chip.className = 'chip preset-item';
    chip.id = `chip-server-${key}`;

    const dot = document.createElement('span');
    dot.className = 'chip-dot';
    dot.style.background = tpl.chip_color || '#666666';
    dot.style.borderColor = getGhostBorderColor(tpl.chip_color || '#666666');

    const text = document.createElement('span');
    text.textContent = tpl.name || key;

    chip.appendChild(dot);
    chip.appendChild(text);
    chip.onclick = () => applyPreset(key);
    container.appendChild(chip);
  });

  requestAnimationFrame(() => {
    const templatesScroll = document.getElementById('templates-scroll');
    if (templatesScroll) {
      const maxScrollLeft = Math.max(0, templatesScroll.scrollWidth - templatesScroll.clientWidth);
      templatesScroll.scrollLeft = Math.min(preserveScrollLeft, maxScrollLeft);
    }
    updateTemplateScrollState();
  });
}

function renderPresetList(keepCurrentScroll = true) {
  const container = document.getElementById('user-preset-chips');
  if (!container) return;
  container.innerHTML = '';

  const scrollArea = document.getElementById('templates-scroll');
  const preserveScrollLeft = keepCurrentScroll && scrollArea ? scrollArea.scrollLeft : 0;

  const userPresets = loadUserPresets();

  Object.entries(userPresets).forEach(([name, preset]) => {
    const chip = document.createElement('button');
    chip.className = 'chip user-preset preset-item';
    chip.id = `chip-user-${name}`;

    const text = document.createElement('span');
    text.textContent = `⭐ ${name}`;

    const del = document.createElement('span');
    del.textContent = '✕';
    del.style.marginLeft = '6px';
    del.style.opacity = '.6';
    del.style.fontSize = '11px';
    del.onclick = (e) => {
      e.stopPropagation();
      deleteUserPreset(name);
    };

    chip.appendChild(text);
    chip.appendChild(del);
    chip.onclick = () => applyPresetFromObject(name, preset);
    container.appendChild(chip);
  });

  lucide.createIcons();
  requestAnimationFrame(() => {
    const templatesScroll = document.getElementById('templates-scroll');
    if (templatesScroll) {
      const maxScrollLeft = Math.max(0, templatesScroll.scrollWidth - templatesScroll.clientWidth);
      templatesScroll.scrollLeft = Math.min(preserveScrollLeft, maxScrollLeft);
    }
    updateTemplateScrollState();
  });
}

function easeOutQuint(t) {
  return 1 - Math.pow(1 - t, 5);
}

function animateTemplatesScrollTo(targetLeft, durationMs = 420) {
  const scrollArea = document.getElementById('templates-scroll');
  if (!scrollArea) return;

  const maxScrollLeft = Math.max(0, scrollArea.scrollWidth - scrollArea.clientWidth);
  const startLeft = scrollArea.scrollLeft;
  const endLeft = Math.max(0, Math.min(targetLeft, maxScrollLeft));

  if (templatesScrollAnimationFrame !== null) {
    cancelAnimationFrame(templatesScrollAnimationFrame);
    templatesScrollAnimationFrame = null;
  }

  if (Math.abs(endLeft - startLeft) < 1) {
    scrollArea.scrollLeft = endLeft;
    updateTemplateScrollState();
    return;
  }

  const startTime = performance.now();
  const step = (now) => {
    const elapsed = now - startTime;
    const progress = Math.min(1, elapsed / durationMs);
    const eased = easeOutQuint(progress);
    scrollArea.scrollLeft = startLeft + (endLeft - startLeft) * eased;

    if (progress < 1) {
      templatesScrollAnimationFrame = requestAnimationFrame(step);
    } else {
      templatesScrollAnimationFrame = null;
      updateTemplateScrollState();
    }
  };

  templatesScrollAnimationFrame = requestAnimationFrame(step);
}

function scrollTemplatesToEdge(direction) {
  const scrollArea = document.getElementById('templates-scroll');
  if (!scrollArea) return;

  const maxScrollLeft = Math.max(0, scrollArea.scrollWidth - scrollArea.clientWidth);
  if (maxScrollLeft <= 0) return;

  animateTemplatesScrollTo(direction === 'left' ? 0 : maxScrollLeft);
}

function updateTemplateScrollState() {
  const scrollArea = document.getElementById('templates-scroll');
  if (!scrollArea) return;
  const leftHint = document.querySelector('.templates-hint-edge-left');
  const rightHint = document.querySelector('.templates-hint-edge-right');

  const scrollable = scrollArea.scrollWidth > scrollArea.clientWidth + 1;
  const atStart = scrollArea.scrollLeft <= 1;
  const atEnd = scrollArea.scrollLeft + scrollArea.clientWidth >= scrollArea.scrollWidth - 1;

  scrollArea.classList.toggle('is-scrollable', scrollable);
  scrollArea.classList.toggle('at-start', atStart);
  scrollArea.classList.toggle('at-end', atEnd);

  if (leftHint) {
    const leftVisible = scrollable && !atStart;
    leftHint.classList.toggle('is-visible', leftVisible);
  }
  if (rightHint) {
    const rightVisible = scrollable && !atEnd;
    rightHint.classList.toggle('is-visible', rightVisible);
  }
}

function deleteUserPreset(name) {
  const userPresets = loadUserPresets();
  delete userPresets[name];
  saveUserPresets(userPresets);
  renderPresetList();
  showSnackbar(`Preset "${name}" gelöscht`);
}

function exportUserPresets() {
  const presets = loadUserPresets();
  const json = JSON.stringify(presets, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'terralines_presets.json';
  a.click();
  URL.revokeObjectURL(url);
}

function importUserPresets(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const imported = JSON.parse(e.target.result);
      const existing = loadUserPresets();

      // Unterstützt sowohl ein einzelnes Template-Objekt
      // als auch ein Dictionary mehrerer Presets.
      if (imported && typeof imported === 'object' && !Array.isArray(imported)) {
        if (imported.params && typeof imported.params === 'object') {
          const key = (imported.name || `import_${Date.now()}`).toString().trim();
          existing[key] = imported.params;
        } else {
          Object.entries(imported).forEach(([k, v]) => {
            if (v && typeof v === 'object' && !Array.isArray(v)) {
              existing[k] = v.params && typeof v.params === 'object' ? v.params : v;
            }
          });
        }
      } else {
        throw new Error('Ungültiges JSON-Format');
      }

      saveUserPresets(existing);
      renderPresetList();
      showSnackbar('Einstellungen importiert');
    } catch (_) {
      showSnackbar('Import fehlgeschlagen — ungültiges JSON');
    }
  };
  reader.readAsText(file);
  input.value = '';
}

function playDiceAnimation(wrapperId) {
  const el = document.getElementById(wrapperId);
  if (!el) return;
  el.classList.remove('rolling');
  void el.offsetWidth;
  el.classList.add('rolling');
  el.addEventListener('animationend', () => {
    el.classList.remove('rolling');
  }, { once: true });
}

/** Klappt eine Accordion-Sektion auf/zu. */
function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  section.classList.toggle('open');
}


/* ===========================================================================
   INITIALISIERUNG
   =========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  const fromHash = loadParamsFromHash();
  if (fromHash) Object.assign(params, fromHash);

  renderServerTemplateList(false);
  // Swatch-Farben aus initialen params setzen
  syncUIToParams();
  initSliderTracks();
  renderPresetList(false);

  const wrapper = document.getElementById('preview-wrapper');
  const previewImg = document.getElementById('preview-img');
  if (wrapper) {
    wrapper.addEventListener('wheel', (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
      viewport.scale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, viewport.scale + delta));
      applyViewportTransform();
    }, { passive: false });

    if (previewImg) {
      previewImg.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        viewport.isDragging = true;
        viewport.dragStartX = e.clientX;
        viewport.dragStartY = e.clientY;
        viewport.dragStartVX = viewport.x;
        viewport.dragStartVY = viewport.y;
        previewImg.style.cursor = 'grabbing';
      });

      previewImg.addEventListener('dragstart', (e) => e.preventDefault());
    }

    wrapper.addEventListener('dblclick', resetViewport);
  }

  window.addEventListener('mousemove', (e) => {
    if (!viewport.isDragging) return;
    viewport.x = viewport.dragStartVX + (e.clientX - viewport.dragStartX);
    viewport.y = viewport.dragStartVY + (e.clientY - viewport.dragStartY);
    applyViewportTransform();
  });

  window.addEventListener('mouseup', () => {
    viewport.isDragging = false;
    if (previewImg) previewImg.style.cursor = 'grab';
  });

  const hmInput = document.getElementById('heightmap-input');
  const hmLabel = document.getElementById('heightmap-drop-label');
  if (hmLabel && hmInput) {
    hmLabel.addEventListener('dragover', (e) => {
      e.preventDefault();
      hmLabel.style.background = 'rgba(255,255,255,0.08)';
    });
    hmLabel.addEventListener('dragleave', () => {
      hmLabel.style.background = 'transparent';
    });
    hmLabel.addEventListener('drop', (e) => {
      e.preventDefault();
      hmLabel.style.background = 'transparent';
      if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files[0]) return;
      handleHeightmapFile(e.dataTransfer.files[0]);
    });
  }

  const templatesScroll = document.getElementById('templates-scroll');
  if (templatesScroll) {
    templatesScroll.addEventListener('wheel', (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        templatesScroll.scrollLeft += e.deltaY;
        e.preventDefault();
      }
    }, { passive: false });

    templatesScroll.addEventListener('scroll', updateTemplateScrollState, { passive: true });
    window.addEventListener('resize', updateTemplateScrollState);
    updateTemplateScrollState();
  }

  const templatesScrollPrev = document.getElementById('templates-scroll-prev');
  const templatesScrollNext = document.getElementById('templates-scroll-next');
  if (templatesScrollPrev) {
    templatesScrollPrev.addEventListener('click', () => scrollTemplatesToEdge('left'));
  }
  if (templatesScrollNext) {
    templatesScrollNext.addEventListener('click', () => scrollTemplatesToEdge('right'));
  }

  const centerPreviewButton = document.getElementById('btn-center-preview');
  if (centerPreviewButton) {
    centerPreviewButton.addEventListener('click', centerPreview);
  }

  const splitToggle = document.getElementById('btn-preview-scale-toggle');
  const splitMenu = document.getElementById('preview-scale-menu');
  const splitGroup = document.getElementById('preview-scale-split-group');
  if (splitToggle) {
    splitToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      togglePreviewScaleMenu();
    });
  }
  if (splitMenu) {
    splitMenu.addEventListener('click', (e) => {
      const target = e.target.closest('[data-preview-scale]');
      if (!target) return;
      applyPreviewScale(target.dataset.previewScale);
      generatePreview();
    });
  }
  document.addEventListener('click', (e) => {
    if (!splitGroup) return;
    if (!splitGroup.contains(e.target)) closePreviewScaleMenu();
  });
  window.addEventListener('blur', closePreviewScaleMenu);

  updateCenterPreviewButtonState();
  updatePreviewScaleControl();

  // Beim Start direkt eine Vorschau generieren
  setTimeout(() => generatePreview(), 300);
  lucide.createIcons();
  updateTemplateScrollState();
});
