"""Tests for the land terrain mesh."""
from __future__ import annotations

import math

import numpy as np
import shapely
import pytest
from shapely.geometry import Polygon

from contour.hex_frame import HexFrame
from contour.terrain_mesh import build_land_mesh
from contour.heightmap import Heightmap
from contour.water_mesh import build_water_mesh


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


@pytest.fixture
def boundary_valley(monkeypatch):
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    corners = np.asarray(frame.polygon_enu().exterior.coords)[:2]
    midpoint = corners.mean(axis=0)

    def sample_valley(heightmap, points, enu):
        distance = np.linalg.norm(points - midpoint, axis=1)
        return 100.0 - 80.0 * np.exp(-(distance / 40.0) ** 2)

    monkeypatch.setattr("contour.terrain_mesh.sample_at_enu", sample_valley)
    return frame, midpoint


def test__build_land_mesh__boundary_follows_valley_and_remains_watertight(boundary_valley):
    frame, midpoint = boundary_valley
    mesh = build_land_mesh(frame, _constant_heightmap(100), [], base_z=-10, grid_points_per_side=40)
    vertices = mesh.vertices
    near_valley = np.linalg.norm(vertices[:, :2] - midpoint, axis=1) < 1e-6
    assert near_valley.any(), "Boundary needs an elevation sample in the valley"
    assert vertices[near_valley, 2].max() == pytest.approx(20)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0



def test__build_land_mesh__shoreline_is_flush_with_flat_water():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    water_polygon = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    land = build_land_mesh(frame, _constant_heightmap(50), [water_polygon],
                           base_z=-10, grid_points_per_side=30, water_levels=[40])
    water = build_water_mesh([water_polygon], top_z=[40], bottom_z=-10)
    points = shapely.points(land.vertices[:, :2])
    shoreline = (shapely.distance(points, water_polygon.boundary) < 1e-6) & (land.vertices[:, 2] > -9)
    assert shoreline.sum() > 4
    assert np.allclose(land.vertices[shoreline, 2], water.bounds[1, 2])
    assert land.bounds[1, 2] == pytest.approx(50)
    assert land.is_watertight and water.is_watertight
    assert land.is_winding_consistent and water.is_winding_consistent


def test__build_land_mesh__rejects_missing_water_level():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    water = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    with pytest.raises(ValueError, match="surface level"):
        build_land_mesh(frame, _constant_heightmap(50), [water], base_z=-10, water_levels=[])


@pytest.mark.parametrize("with_water", [False, True])
def test_adaptive_terrain_is_a_watertight_solid(monkeypatch, with_water):
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    def sampled_surface(heightmap, points, enu):
        return 50 + 3 * np.sin(points[:, 0] / 30) * np.cos(points[:, 1] / 20)
    monkeypatch.setattr("contour.terrain_mesh.sample_at_enu", sampled_surface)
    water = [Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])] if with_water else []
    mesh = build_land_mesh(frame, _constant_heightmap(50), water, base_z=-10,
                           grid_points_per_side=12, water_levels=[47] if with_water else [],
                           surface_tolerance_m=.03, sample_spacing_m=2)
    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.bounds[0, 2] == -10
