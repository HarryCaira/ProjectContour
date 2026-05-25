"""Tests for the water mesh."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from contour.mesh.water import build_water_mesh


def test_no_polygons_returns_none():
    assert build_water_mesh([], top_z=0.0, bottom_z=-5.0) is None


def test_zero_height_returns_none():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert build_water_mesh([poly], top_z=0.0, bottom_z=0.0) is None


def test_single_polygon_is_watertight():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    mesh = build_water_mesh([poly], top_z=-1.0, bottom_z=-3.0)
    assert mesh is not None
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(100 * 2, rel=1e-6)


def test_two_polygons_concatenated():
    p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    p2 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    mesh = build_water_mesh([p1, p2], top_z=-1.0, bottom_z=-3.0)
    assert mesh is not None
    assert mesh.volume == pytest.approx(100 * 2 + 100 * 2, rel=1e-6)


def test_water_top_and_bottom_z():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    mesh = build_water_mesh([poly], top_z=-1.0, bottom_z=-5.0)
    assert mesh is not None
    assert mesh.bounds[0, 2] == pytest.approx(-5.0)
    assert mesh.bounds[1, 2] == pytest.approx(-1.0)


def test_invalid_polygon_skipped():
    bad = Polygon([(0, 0), (10, 0), (5, 5), (10, 10), (0, 10), (5, 5)])  # self-intersection
    good = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    mesh = build_water_mesh([bad, good], top_z=-1.0, bottom_z=-3.0)
    assert mesh is not None
    # Only the good polygon should contribute volume.
    assert mesh.volume == pytest.approx(100 * 2, rel=1e-6)
