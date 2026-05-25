"""Tests for the land terrain mesh."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from contour.framing.hex import HexFrame
from contour.mesh.terrain import build_land_mesh
from contour.schema.heightmap import Heightmap


def _tile_at(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return x, y


def _constant_heightmap(value: float, zoom: int = 14) -> Heightmap:
    tx, ty = _tile_at(0.0, 0.0, zoom)
    return Heightmap(
        elevations=np.full((512, 512), value, dtype=np.float32),
        zoom=zoom,
        tile_origin_x=tx - 1,
        tile_origin_y=ty - 1,
    )


def test_land_mesh_is_watertight():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _constant_heightmap(50.0)
    mesh = build_land_mesh(frame, hm, water_polygons=[], base_z=-10.0, grid_points_per_side=30)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent


def test_land_mesh_top_z_matches_heightmap():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _constant_heightmap(50.0)
    mesh = build_land_mesh(frame, hm, water_polygons=[], base_z=-10.0, grid_points_per_side=30)
    # Top face is at elevation 50, bottom at -10.
    assert mesh.bounds[1, 2] == pytest.approx(50.0, abs=0.01)
    assert mesh.bounds[0, 2] == pytest.approx(-10.0, abs=0.01)


def test_land_mesh_with_water_hole_has_walls_inside():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _constant_heightmap(50.0)
    water = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    mesh = build_land_mesh(frame, hm, water_polygons=[water], base_z=-10.0, grid_points_per_side=30)
    assert mesh.is_watertight
    # Volume should be (land area) * (top - base) = (hex_area - water_area) * 60
    hex_area = (3 * math.sqrt(3) / 2) * 200**2
    expected_volume = (hex_area - 100 * 100) * 60.0
    assert mesh.volume == pytest.approx(expected_volume, rel=1e-2)


def test_land_mesh_volume_for_constant_heightmap():
    """For a constant heightmap, volume = hex area × (top_z - base_z)."""
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=300)
    hm = _constant_heightmap(20.0)
    mesh = build_land_mesh(frame, hm, water_polygons=[], base_z=-5.0, grid_points_per_side=40)
    hex_area = (3 * math.sqrt(3) / 2) * 300**2
    expected = hex_area * 25.0
    assert mesh.volume == pytest.approx(expected, rel=1e-2)


def test_land_mesh_xy_bounds_within_hex():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    hm = _constant_heightmap(0.0)
    mesh = build_land_mesh(frame, hm, water_polygons=[], base_z=-1.0, grid_points_per_side=30)
    # Hex circumradius defines the maximum extent
    assert mesh.bounds[1, 0] <= 200.0 + 1e-6
    assert mesh.bounds[1, 1] <= 200.0 + 1e-6
    assert mesh.bounds[0, 0] >= -200.0 - 1e-6
    assert mesh.bounds[0, 1] >= -200.0 - 1e-6
