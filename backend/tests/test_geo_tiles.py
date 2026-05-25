"""Tests for Web Mercator tile-space math."""
from __future__ import annotations

import pytest

from contour.geo.tiles import (
    RasterTile,
    lonlat_to_pixel,
    lonlat_to_tile,
    pixel_to_lonlat,
    tiles_covering_bbox,
)


def test_zoom_zero_tile_bbox_is_world():
    t = RasterTile(zoom=0, x=0, y=0)
    west, south, east, north = t.bbox
    assert west == pytest.approx(-180)
    assert east == pytest.approx(180)
    assert north == pytest.approx(85.0511, abs=0.001)
    assert south == pytest.approx(-85.0511, abs=0.001)


def test_lonlat_to_tile_finds_containing_tile():
    t = lonlat_to_tile(-0.1, 51.5, 12)
    assert t.zoom == 12
    west, south, east, north = t.bbox
    assert west <= -0.1 <= east
    assert south <= 51.5 <= north


def test_pixel_round_trip():
    px = lonlat_to_pixel(-0.1, 51.5, 14)
    lonlat = pixel_to_lonlat(float(px[0]), float(px[1]), 14)
    assert lonlat[0] == pytest.approx(-0.1, abs=1e-9)
    assert lonlat[1] == pytest.approx(51.5, abs=1e-9)


def test_tiles_covering_tiny_bbox_returns_one_tile():
    tiles = tiles_covering_bbox(-0.10001, 51.50001, -0.09999, 51.50003, 14)
    assert len(tiles) == 1


def test_tiles_covering_larger_bbox_returns_multiple():
    tiles = tiles_covering_bbox(-1.0, 51.0, 1.0, 52.0, 10)
    assert len(tiles) > 1
    assert all(t.zoom == 10 for t in tiles)


def test_tile_bbox_consistent_with_corner_lookups():
    t = lonlat_to_tile(-0.1, 51.5, 12)
    west, south, east, north = t.bbox
    assert west < east
    assert south < north
