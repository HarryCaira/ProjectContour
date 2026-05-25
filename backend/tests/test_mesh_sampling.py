"""Tests for heightmap bilinear sampling at ENU points."""
from __future__ import annotations

import math

import numpy as np
import pytest

from contour.geo.transforms import LocalENU
from contour.mesh.sampling import sample_at_enu
from contour.schema.heightmap import Heightmap


def _tile_at(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return x, y


def test_sample_returns_constant_for_uniform_heightmap():
    tx, ty = _tile_at(0.0, 0.0, 12)
    hm = Heightmap(
        elevations=np.full((512, 512), 42.5, dtype=np.float32),
        zoom=12,
        tile_origin_x=tx - 1,
        tile_origin_y=ty - 1,
    )
    enu = LocalENU(lat0=0.0, lon0=0.0)
    pts = np.array([[0.0, 0.0], [50.0, 50.0], [-30.0, 20.0]])
    z = sample_at_enu(hm, pts, enu)
    np.testing.assert_allclose(z, 42.5, atol=1e-3)


def test_sample_empty_input_returns_empty_array():
    tx, ty = _tile_at(0.0, 0.0, 12)
    hm = Heightmap(elevations=np.zeros((256, 256), dtype=np.float32), zoom=12, tile_origin_x=tx, tile_origin_y=ty)
    enu = LocalENU(lat0=0.0, lon0=0.0)
    z = sample_at_enu(hm, np.empty((0, 2)), enu)
    assert z.shape == (0,)


def test_sample_rejects_wrong_shape():
    tx, ty = _tile_at(0.0, 0.0, 12)
    hm = Heightmap(elevations=np.zeros((256, 256), dtype=np.float32), zoom=12, tile_origin_x=tx, tile_origin_y=ty)
    enu = LocalENU(lat0=0.0, lon0=0.0)
    with pytest.raises(ValueError, match="N, 2"):
        sample_at_enu(hm, np.zeros((5, 3)), enu)


def test_sample_interpolates_linear_gradient():
    """A heightmap that varies linearly in pixel-x should return a linear
    function of pixel-x for nearby ENU points."""
    tx, ty = _tile_at(0.0, 0.0, 14)
    h, w = 512, 512
    gradient = np.tile(np.linspace(0, 100, w, dtype=np.float32), (h, 1))
    hm = Heightmap(elevations=gradient, zoom=14, tile_origin_x=tx - 1, tile_origin_y=ty - 1)
    enu = LocalENU(lat0=0.0, lon0=0.0)

    # Sample at origin and at +ENU east (should give higher elevation since pixel-x increases east)
    pts = np.array([[0.0, 0.0], [50.0, 0.0]])
    z = sample_at_enu(hm, pts, enu)
    assert z[1] > z[0]


def test_sample_out_of_range_clamps():
    """Sampling far outside the heightmap should not crash; it clamps to nearest pixel."""
    tx, ty = _tile_at(0.0, 0.0, 12)
    hm = Heightmap(elevations=np.full((256, 256), 99.0, dtype=np.float32), zoom=12, tile_origin_x=tx, tile_origin_y=ty)
    enu = LocalENU(lat0=0.0, lon0=0.0)
    pts = np.array([[1_000_000.0, 1_000_000.0]])
    z = sample_at_enu(hm, pts, enu)
    assert z[0] == pytest.approx(99.0)
