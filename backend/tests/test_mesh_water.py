"""Tests for the water mesh."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from contour.water_mesh import build_water_mesh, build_shoreline_water_mesh
from contour.hex_frame import HexFrame
from contour.heightmap import Heightmap


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


@pytest.fixture
def shoreline_scene(monkeypatch):
    frame = HexFrame(centre_lat=0, centre_lon=0, circumradius_m=1000)
    heightmap = Heightmap(elevations=np.zeros((256, 256)), zoom=14, tile_origin_x=0, tile_origin_y=0)

    def sample_banks(heightmap, points, enu):
        return np.where(points[:, 0] < 0, 100.0, 700.0)

    monkeypatch.setattr("contour.water_mesh.sample_at_enu", sample_banks)
    low = Polygon([(-300, -50), (-200, -50), (-200, 50), (-300, 50)])
    high = Polygon([(200, -50), (300, -50), (300, 50), (200, 50)])
    return frame, heightmap, low, high


def test__build_shoreline_water_mesh__separate_lakes_follow_local_banks(shoreline_scene):
    frame, heightmap, low, high = shoreline_scene
    mesh = build_shoreline_water_mesh([low, high], heightmap, frame, bottom_z=-20, recess_m=2)
    assert mesh is not None
    components = sorted(mesh.split(), key=lambda part: part.bounds[1, 2])
    assert len(components) == 2
    assert [part.bounds[1, 2] for part in components] == pytest.approx([98, 698])
    for part in components:
        assert part.is_watertight
        assert part.is_winding_consistent
        assert part.bounds[0, 2] == pytest.approx(-20)
        assert part.volume > 0


def test__build_shoreline_water_mesh__deep_recess_cannot_remove_water(shoreline_scene):
    frame, heightmap, low, _ = shoreline_scene
    mesh = build_shoreline_water_mesh([low], heightmap, frame, bottom_z=99, recess_m=1000)
    assert mesh is not None
    assert mesh.bounds[1, 2] == pytest.approx(99.5)
    assert mesh.is_watertight


def test__build_shoreline_water_mesh__zero_recess_matches_bank(shoreline_scene):
    frame, heightmap, low, _ = shoreline_scene
    mesh = build_shoreline_water_mesh([low], heightmap, frame, bottom_z=0, recess_m=0)
    assert mesh.bounds[1, 2] == pytest.approx(100)


@pytest.mark.parametrize("recess", [-1, float("nan"), float("inf")])
def test__build_shoreline_water_mesh__rejects_invalid_recess(shoreline_scene, recess):
    frame, heightmap, low, _ = shoreline_scene
    with pytest.raises(ValueError, match="recess"):
        build_shoreline_water_mesh([low], heightmap, frame, bottom_z=0, recess_m=recess)


def test__build_shoreline_water_mesh__empty_returns_none(shoreline_scene):
    frame, heightmap, _, _ = shoreline_scene
    assert build_shoreline_water_mesh([], heightmap, frame, bottom_z=0, recess_m=1) is None


def test__build_shoreline_water_mesh__rejects_bank_below_base(shoreline_scene):
    frame, heightmap, low, _ = shoreline_scene
    with pytest.raises(ValueError, match="above the model base"):
        build_shoreline_water_mesh([low], heightmap, frame, bottom_z=100, recess_m=1)


def test__build_shoreline_water_mesh__rejects_nonfinite_elevation(shoreline_scene, monkeypatch):
    frame, heightmap, low, _ = shoreline_scene
    monkeypatch.setattr("contour.water_mesh.sample_at_enu", lambda hm, points, enu: np.full(len(points), np.nan))
    with pytest.raises(ValueError, match="invalid elevations"):
        build_shoreline_water_mesh([low], heightmap, frame, bottom_z=0, recess_m=1)


def test__build_shoreline_water_mesh__full_water_frame_uses_boundary_fallback(shoreline_scene, monkeypatch):
    frame, heightmap, _, _ = shoreline_scene
    monkeypatch.setattr("contour.water_mesh.sample_at_enu", lambda hm, points, enu: np.full(len(points), 50.0))
    mesh = build_shoreline_water_mesh([frame.polygon_enu()], heightmap, frame, bottom_z=0, recess_m=1)
    assert mesh.is_watertight
    assert mesh.bounds[1, 2] == pytest.approx(49)



def test__build_water_mesh__rejects_missing_level():
    water = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    with pytest.raises(ValueError, match="surface level"):
        build_water_mesh([water], top_z=[], bottom_z=-10)
