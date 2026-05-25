"""Tests for hex framing."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point

from contour.framing.hex import HexFrame, hex_frame_for_route
from contour.schema.route import Route


def _route(lats, lons, eles=None) -> Route:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if eles is None:
        eles = np.zeros_like(lats)
    return Route(latitudes=lats, longitudes=lons, elevations=eles)


def test_hex_contains_all_route_points():
    route = _route(
        lats=[51.50, 51.51, 51.52, 51.51, 51.50],
        lons=[-0.10, -0.11, -0.10, -0.09, -0.10],
    )
    frame = hex_frame_for_route(route, padding_ratio=0.15)
    enu = frame.local_enu()
    pts = enu.to_enu(route.latitudes, route.longitudes, 0.0)

    poly = frame.polygon_enu().buffer(0.001)
    for px, py, _ in pts:
        assert poly.contains(Point(px, py)), f"Point ({px}, {py}) not in hex"


def test_hex_size_scales_with_padding():
    route = _route([51.5, 51.51], [-0.1, -0.11])
    small = hex_frame_for_route(route, padding_ratio=0.0)
    large = hex_frame_for_route(route, padding_ratio=0.5)
    assert large.circumradius_m > small.circumradius_m


def test_hex_centred_on_route_bbox_centroid():
    route = _route([51.5, 51.52], [-0.12, -0.10])
    frame = hex_frame_for_route(route)
    assert frame.centre_lon == pytest.approx(-0.11)
    assert frame.centre_lat == pytest.approx(51.51)


def test_hex_has_six_vertices():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000, rotation_degrees=0)
    verts = frame.vertices_enu()
    assert verts.shape == (6, 2)


def test_apothem_relation():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000, rotation_degrees=0)
    assert frame.apothem_m == pytest.approx(1000 * math.sqrt(3) / 2)


def test_pointy_top_first_vertex_is_north():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000, rotation_degrees=0)
    verts = frame.vertices_enu()
    # First vertex should be at +N (E ~ 0, N = circumradius)
    assert verts[0, 0] == pytest.approx(0, abs=1e-9)
    assert verts[0, 1] == pytest.approx(1000)


def test_rotation_rotates_vertices():
    f0 = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000, rotation_degrees=0)
    f30 = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000, rotation_degrees=30)
    # 30° rotation should move the top vertex to (-500, 866) (approx)
    v0 = f0.vertices_enu()[0]
    v30 = f30.vertices_enu()[0]
    assert v30[0] < v0[0]  # rotated CCW; x decreases
    assert v30[1] < v0[1]


def test_polygon_area_matches_hex_formula():
    r = 1000.0
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=r)
    expected_area = (3 * math.sqrt(3) / 2) * r**2
    assert frame.polygon_enu().area == pytest.approx(expected_area, rel=1e-9)


def test_route_with_no_padding_inscribed_in_hex():
    """With padding=0, the hex's inscribed circle equals the route's max radius."""
    route = _route([51.5, 51.501], [-0.1, -0.1])  # very short
    frame = hex_frame_for_route(route, padding_ratio=0.0)
    enu = frame.local_enu()
    pts = enu.to_enu(route.latitudes, route.longitudes, 0.0)
    max_radius = float(np.hypot(pts[:, 0], pts[:, 1]).max())
    assert frame.apothem_m == pytest.approx(max_radius, rel=1e-9)
