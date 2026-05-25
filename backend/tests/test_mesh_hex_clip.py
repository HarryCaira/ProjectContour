"""Tests for hex_clip: land polygon computation and CDT triangulation."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from contour.framing.hex import HexFrame
from contour.mesh.hex_clip import land_polygon, triangulate_land


def _hex_polygon(r: float = 1000.0) -> Polygon:
    return HexFrame(centre_lon=0, centre_lat=0, circumradius_m=r).polygon_enu()


def test_land_polygon_with_no_water_is_hex():
    hex_poly = _hex_polygon()
    result = land_polygon(hex_poly, [])
    assert result.area == pytest.approx(hex_poly.area, rel=1e-9)


def test_land_polygon_with_water_has_hole():
    hex_poly = _hex_polygon()
    water = Polygon([(-200, -200), (200, -200), (200, 200), (-200, 200)])
    result = land_polygon(hex_poly, [water])
    # Expect a Polygon with one interior ring.
    assert result.geom_type == "Polygon"
    assert len(list(result.interiors)) == 1
    assert result.area == pytest.approx(hex_poly.area - 400 * 400, rel=1e-9)


def test_triangulate_land_produces_triangles():
    hex_poly = _hex_polygon()
    tri = triangulate_land(hex_poly, [], grid_points_per_side=30)
    assert tri.triangles.shape[1] == 3
    assert len(tri.triangles) > 0


def test_triangulation_area_matches_hex():
    hex_poly = _hex_polygon()
    tri = triangulate_land(hex_poly, [], grid_points_per_side=30)
    total_area = 0.0
    for a, b, c in tri.triangles:
        v = tri.vertices[[a, b, c]]
        total_area += abs(0.5 * ((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1]) - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])))
    assert total_area == pytest.approx(hex_poly.area, rel=1e-3)


def test_triangulation_area_excludes_water_hole():
    hex_poly = _hex_polygon()
    water = Polygon([(-200, -200), (200, -200), (200, 200), (-200, 200)])
    tri = triangulate_land(hex_poly, [water], grid_points_per_side=30)
    total_area = 0.0
    for a, b, c in tri.triangles:
        v = tri.vertices[[a, b, c]]
        total_area += abs(0.5 * ((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1]) - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])))
    expected = hex_poly.area - 400 * 400
    assert total_area == pytest.approx(expected, rel=1e-2)


def test_triangulation_has_boundary_segments():
    hex_poly = _hex_polygon()
    tri = triangulate_land(hex_poly, [], grid_points_per_side=20)
    # A hex has 6 outer edges; with no water there should be exactly 6 boundary segments.
    assert len(tri.boundary_segments) == 6


def test_triangulation_with_water_has_additional_boundary():
    hex_poly = _hex_polygon()
    water = Polygon([(-200, -200), (200, -200), (200, 200), (-200, 200)])
    tri = triangulate_land(hex_poly, [water], grid_points_per_side=20)
    # 6 hex edges + 4 water edges = 10
    assert len(tri.boundary_segments) == 10


def test_triangulation_grid_adds_interior_points():
    """A higher grid resolution should produce more interior vertices."""
    hex_poly = _hex_polygon()
    low = triangulate_land(hex_poly, [], grid_points_per_side=5)
    high = triangulate_land(hex_poly, [], grid_points_per_side=50)
    assert len(high.vertices) > len(low.vertices)
