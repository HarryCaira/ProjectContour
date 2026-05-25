"""Tests for the route ribbon mesh."""
from __future__ import annotations

import math

import numpy as np
import pytest

from contour.framing.hex import HexFrame
from contour.mesh.route import build_route_mesh
from contour.schema.heightmap import Heightmap
from contour.schema.route import Route


def _tile_at(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return x, y


def _flat_heightmap(value: float = 0.0, zoom: int = 14) -> Heightmap:
    tx, ty = _tile_at(0.0, 0.0, zoom)
    return Heightmap(
        elevations=np.full((512, 512), value, dtype=np.float32),
        zoom=zoom,
        tile_origin_x=tx - 1,
        tile_origin_y=ty - 1,
    )


def _straight_route(num_points: int = 20, span_deg: float = 0.001) -> Route:
    """A straight east-going route centred on (0, 0) of given lon span."""
    lons = np.linspace(-span_deg / 2, span_deg / 2, num_points)
    lats = np.zeros(num_points)
    eles = np.zeros(num_points)
    return Route(latitudes=lats, longitudes=lons, elevations=eles)


def test_short_route_returns_none():
    route = Route(
        latitudes=np.array([0.0]),
        longitudes=np.array([0.0]),
        elevations=np.array([0.0]),
    )
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _flat_heightmap()
    mesh = build_route_mesh(route, frame, hm, width_m=5.0, height_above_terrain_m=2.0)
    assert mesh is None


def test_route_mesh_is_watertight():
    route = _straight_route()
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _flat_heightmap()
    mesh = build_route_mesh(route, frame, hm, width_m=5.0, height_above_terrain_m=2.0)
    assert mesh is not None
    assert mesh.is_watertight
    assert mesh.is_winding_consistent


def test_route_mesh_top_above_terrain():
    route = _straight_route()
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _flat_heightmap(value=10.0)
    mesh = build_route_mesh(route, frame, hm, width_m=5.0, height_above_terrain_m=3.0)
    assert mesh is not None
    # Terrain is at 10; ribbon top should be at 13; bottom at 10.
    assert mesh.bounds[1, 2] == pytest.approx(13.0, abs=0.1)
    assert mesh.bounds[0, 2] == pytest.approx(10.0, abs=0.1)


def test_route_mesh_volume_approx():
    """For a straight route on flat terrain, volume ≈ length × width × height."""
    route = _straight_route(num_points=20, span_deg=0.001)
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _flat_heightmap()
    width, height = 4.0, 2.0
    mesh = build_route_mesh(route, frame, hm, width_m=width, height_above_terrain_m=height)
    assert mesh is not None
    # Length ≈ 0.001 deg * 111000 m/deg ≈ 111 m
    expected_volume = 111.0 * width * height
    assert mesh.volume == pytest.approx(expected_volume, rel=0.1)


def test_route_downsampling_caps_segments():
    """A very long route should be downsampled to max_segments."""
    long_route = _straight_route(num_points=2000)
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _flat_heightmap()
    mesh = build_route_mesh(long_route, frame, hm, width_m=4.0, height_above_terrain_m=2.0, max_segments=100)
    assert mesh is not None
    # 4 vertices per cross-section, at most (max_segments + 1) sections.
    # After trimesh processing, vertices may merge; check via face count instead.
    # Faces per segment: 8 (sides) + walls; plus 2 end caps × 2 triangles = 4.
    # Upper bound on faces: 8 * 100 + 4 = 804.
    assert len(mesh.faces) < 900
