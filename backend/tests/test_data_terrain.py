"""Tests for terrain data: decoding, stitching, zoom selection, fetch orchestration."""
from __future__ import annotations

import io

import numpy as np
import pytest
import responses
from PIL import Image

from contour.data.terrain import (
    decode_terrain_rgb,
    fetch_heightmap,
    select_zoom,
    stitch_heightmap,
)
from contour.framing.hex import HexFrame
from contour.http.cache import TileCache
from contour.http.client import HttpClient
from contour.schema.settings import Physical


def _make_terrain_png(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- decode_terrain_rgb ----------


def test_decode_terrain_rgb_black_is_neg_10000():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    eles = decode_terrain_rgb(_make_terrain_png(rgb))
    assert eles.shape == (4, 4)
    np.testing.assert_allclose(eles, -10000.0)


def test_decode_terrain_rgb_known_pixel():
    # (R=1, G=0, B=0) -> -10000 + 65536 * 0.1 = -3446.4
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [1, 0, 0]
    eles = decode_terrain_rgb(_make_terrain_png(rgb))
    assert eles[0, 0] == pytest.approx(-3446.4)
    assert eles[1, 1] == pytest.approx(-10000.0)


def test_decode_terrain_rgb_full_range():
    # (255, 255, 255) -> -10000 + (255*65536 + 255*256 + 255) * 0.1 = 1667721.5
    rgb = np.full((1, 1, 3), 255, dtype=np.uint8)
    eles = decode_terrain_rgb(_make_terrain_png(rgb))
    assert eles[0, 0] == pytest.approx(1667721.5)


# ---------- stitch_heightmap ----------


def test_stitch_heightmap_single_tile():
    data = np.full((256, 256), 100.0, dtype=np.float32)
    hm = stitch_heightmap({(5, 7): data}, zoom=12)
    assert hm.shape == (256, 256)
    assert hm.zoom == 12
    assert hm.tile_origin_x == 5
    assert hm.tile_origin_y == 7
    np.testing.assert_array_equal(hm.elevations, data)


def test_stitch_heightmap_2x2():
    tiles = {
        (10, 20): np.full((256, 256), 1.0, dtype=np.float32),  # NW
        (11, 20): np.full((256, 256), 2.0, dtype=np.float32),  # NE
        (10, 21): np.full((256, 256), 3.0, dtype=np.float32),  # SW
        (11, 21): np.full((256, 256), 4.0, dtype=np.float32),  # SE
    }
    hm = stitch_heightmap(tiles, zoom=12)
    assert hm.shape == (512, 512)
    assert hm.tile_origin_x == 10
    assert hm.tile_origin_y == 20
    assert hm.elevations[0, 0] == 1.0
    assert hm.elevations[0, 300] == 2.0
    assert hm.elevations[300, 0] == 3.0
    assert hm.elevations[300, 300] == 4.0


def test_stitch_heightmap_empty_raises():
    with pytest.raises(ValueError, match="No tiles"):
        stitch_heightmap({}, zoom=12)


def test_stitch_heightmap_mismatched_shapes_raises():
    tiles = {
        (0, 0): np.zeros((256, 256), dtype=np.float32),
        (1, 0): np.zeros((128, 256), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="identical shape"):
        stitch_heightmap(tiles, zoom=12)


# ---------- select_zoom ----------


def test_select_zoom_small_model_picks_high_zoom():
    frame = HexFrame(centre_lon=0, centre_lat=51.5, circumradius_m=500)
    physical = Physical(size_mm=150, resolution_mm=0.2)
    zoom = select_zoom(frame, physical)
    assert zoom >= 14


def test_select_zoom_large_hex_picks_lower_zoom():
    frame = HexFrame(centre_lon=0, centre_lat=51.5, circumradius_m=50_000)
    physical = Physical(size_mm=150, resolution_mm=0.2)
    zoom = select_zoom(frame, physical)
    assert zoom <= 13


def test_select_zoom_respects_tile_budget():
    """A hex too large at every zoom-within-budget still returns a usable zoom."""
    frame = HexFrame(centre_lon=0, centre_lat=51.5, circumradius_m=200_000)
    physical = Physical(size_mm=150, resolution_mm=0.05)  # very fine
    zoom = select_zoom(frame, physical, max_tiles=64)
    assert 1 <= zoom <= 16


# ---------- fetch_heightmap (integration with mocked HTTP) ----------


@responses.activate
def test_fetch_heightmap_assembles_from_mocked_tiles(tmp_path):
    frame = HexFrame(centre_lon=0.0, centre_lat=51.5, circumradius_m=300)
    physical = Physical(size_mm=150, resolution_mm=2.0)  # coarse → few tiles

    constant_png = _make_terrain_png(np.full((256, 256, 3), [0, 0, 100], dtype=np.uint8))
    # constant pixel value (0, 0, 100) -> elevation = -10000 + 100*0.1 = -9990
    responses.add_passthru("")  # not used, just preallocate
    responses.reset()
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.terrain-rgb/.*"),
        body=constant_png,
        status=200,
    )

    client = HttpClient(backoff_factor=0.0)
    cache = TileCache(tmp_path)
    hm = fetch_heightmap(frame, physical, client, cache, mapbox_token="test-token")

    assert hm.zoom >= 1
    assert hm.elevations.min() == pytest.approx(-9990.0)
    assert hm.elevations.max() == pytest.approx(-9990.0)


@responses.activate
def test_fetch_heightmap_uses_cache_on_second_call(tmp_path):
    frame = HexFrame(centre_lon=0.0, centre_lat=51.5, circumradius_m=300)
    physical = Physical(size_mm=150, resolution_mm=2.0)

    constant_png = _make_terrain_png(np.zeros((256, 256, 3), dtype=np.uint8))
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.terrain-rgb/.*"),
        body=constant_png,
        status=200,
    )

    client = HttpClient(backoff_factor=0.0)
    cache = TileCache(tmp_path)

    fetch_heightmap(frame, physical, client, cache, mapbox_token="test-token")
    calls_after_first = len(responses.calls)

    fetch_heightmap(frame, physical, client, cache, mapbox_token="test-token")
    calls_after_second = len(responses.calls)

    assert calls_after_second == calls_after_first  # no new HTTP requests
