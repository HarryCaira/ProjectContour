"""Tests for hex_clip: land polygon computation and CDT triangulation."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from contour.hex_frame import HexFrame
from contour.hex_clip import land_polygon, triangulate_land


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
    assert len(tri.boundary_segments) > 6
    edges = tri.vertices[np.asarray(tri.boundary_segments)]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    bounds = hex_poly.bounds
    spacing = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 20
    assert lengths.max() <= spacing * (1 + 1e-9)
    assert lengths.sum() == pytest.approx(hex_poly.length)


def test_triangulation_with_water_has_additional_boundary():
    hex_poly = _hex_polygon()
    water = Polygon([(-200, -200), (200, -200), (200, 200), (-200, 200)])
    tri = triangulate_land(hex_poly, [water], grid_points_per_side=20)
    edges = tri.vertices[np.asarray(tri.boundary_segments)]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    assert len(tri.boundary_segments) > 10
    assert lengths.sum() == pytest.approx(hex_poly.length + water.length)
    assert lengths.max() <= 100 * (1 + 1e-9)


def test_triangulation_grid_adds_interior_points():
    """A higher grid resolution should produce more interior vertices."""
    hex_poly = _hex_polygon()
    low = triangulate_land(hex_poly, [], grid_points_per_side=5)
    high = triangulate_land(hex_poly, [], grid_points_per_side=50)
    assert len(high.vertices) > len(low.vertices)


@pytest.mark.parametrize("density", [0, -1])
def test__triangulate_land__rejects_invalid_density(density: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        triangulate_land(_hex_polygon(), [], grid_points_per_side=density)


def test__triangulate_land__cleans_near_duplicate_edges_on_disconnected_land() -> None:
    frame = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    # A river divides the land; a virtually repeated bank vertex previously
    # bypassed cleanup when the two land components were triangulated separately.
    water = Polygon([(4, -1), (6, -1), (6, 5), (6, 5 + 1e-14), (6, 11), (4, 11)])
    assert land_polygon(frame, [water]).geom_type == "MultiPolygon"
    tri = triangulate_land(frame, [water], grid_points_per_side=10)
    edges = tri.vertices[np.asarray(tri.boundary_segments)]
    assert np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).min() > 1e-8
    faces = tri.vertices[tri.triangles]
    ab, ac = faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
    area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum() / 2
    assert area == pytest.approx(80)
