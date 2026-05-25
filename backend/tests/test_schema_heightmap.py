"""Tests for the Heightmap dataclass — bbox math against known tile geometry."""
from __future__ import annotations

import numpy as np
import pytest

from contour.geo.tiles import RasterTile
from contour.schema.heightmap import Heightmap


def _zeros(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)


def test_bbox_for_world_at_zoom_zero():
    """A heightmap covering tile (0, 0, 0) at zoom 0 spans the whole Web Mercator world."""
    hm = Heightmap(elevations=_zeros(256, 256), zoom=0, tile_origin_x=0, tile_origin_y=0)
    west, south, east, north = hm.bbox
    assert west == pytest.approx(-180)
    assert east == pytest.approx(180)
    assert north == pytest.approx(85.0511, abs=0.001)
    assert south == pytest.approx(-85.0511, abs=0.001)


def test_bbox_for_single_tile_matches_tile_bbox():
    """A heightmap covering exactly one tile should have that tile's bbox."""
    zoom, x, y = 12, 2048, 1364
    hm = Heightmap(elevations=_zeros(256, 256), zoom=zoom, tile_origin_x=x, tile_origin_y=y)
    expected = RasterTile(zoom=zoom, x=x, y=y).bbox
    np.testing.assert_allclose(hm.bbox, expected)


def test_bbox_for_2x2_grid_matches_outer_corners():
    """A 2x2 tile mosaic's bbox is the union of its NW and SE tiles' bboxes."""
    zoom, x, y = 14, 100, 200
    hm = Heightmap(elevations=_zeros(512, 512), zoom=zoom, tile_origin_x=x, tile_origin_y=y)
    nw = RasterTile(zoom=zoom, x=x, y=y).bbox
    se = RasterTile(zoom=zoom, x=x + 1, y=y + 1).bbox

    west, south, east, north = hm.bbox
    assert west == pytest.approx(nw[0])
    assert north == pytest.approx(nw[3])
    assert east == pytest.approx(se[2])
    assert south == pytest.approx(se[1])


def test_bbox_invariants():
    """For any heightmap, west < east and south < north."""
    hm = Heightmap(elevations=_zeros(256, 256), zoom=10, tile_origin_x=500, tile_origin_y=300)
    west, south, east, north = hm.bbox
    assert west < east
    assert south < north


def test_shape_matches_elevations():
    hm = Heightmap(elevations=_zeros(512, 768), zoom=10, tile_origin_x=0, tile_origin_y=0)
    assert hm.shape == (512, 768)
