from __future__ import annotations

import copy

import pytest

from app.generator import DEFAULT_PARAMS, TopoParams, generate_topography, generate_topography_svg, heightmap_from_image, load_templates


def test_topo_params_reject_out_of_range_values():
    params = copy.deepcopy(DEFAULT_PARAMS)
    params['width'] = 10_000

    with pytest.raises(ValueError):
        TopoParams.from_dict(params)


def test_load_templates_returns_non_empty_collection():
    templates = load_templates()

    assert isinstance(templates, dict)
    assert len(templates) > 0
    sample = next(iter(templates.values()))
    assert 'name' in sample
    assert 'params' in sample


def test_generate_topography_preview_respects_minimum_preview_resolution():
    params = copy.deepcopy(DEFAULT_PARAMS)
    params.update({'width': 320, 'height': 180, 'preview_scale': 0.1, 'octaves': 1, 'levels': 2})

    result = generate_topography(params, preview=True)

    assert result['width'] == 320
    assert result['height'] == 180
    assert isinstance(result['image'], str)
    assert len(result['image']) > 0


def test_generate_topography_svg_returns_svg_payload():
    params = copy.deepcopy(DEFAULT_PARAMS)
    params.update({'width': 320, 'height': 180, 'octaves': 1, 'levels': 2})

    result = generate_topography_svg(params)

    assert result['width'] == 320
    assert result['height'] == 180
    assert result['svg'].startswith('<?xml')


def test_heightmap_from_invalid_bytes_raises_value_error():
    with pytest.raises(ValueError):
        heightmap_from_image(b'not-an-image', width=320, height=180)
