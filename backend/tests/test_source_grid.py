"""Native-source mode retains measured raster sample locations and heights."""
from collections.abc import Callable

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

from contour.errors import MeshDetailLimitError
from contour.heightmap import Heightmap
from contour.hex_frame import HexFrame
from contour.sampling import sample_at_enu
from contour.source_grid import source_grid_points
from contour.terrain_mesh import build_land_mesh
from contour.tiles import pixel_to_lonlat


@pytest.fixture
def native_scene() -> tuple[Heightmap, HexFrame]:
    zoom = 15
    lon, lat = pixel_to_lonlat(16384 * 256 + 7.5, 16384 * 256 + 7.5, zoom)
    frame = HexFrame(float(lon), float(lat), 25)
    grid = np.arange(256, dtype=float).reshape(16, 16) / 10 + 100
    return Heightmap(grid, zoom, 16384, 16384), frame


def test__source_grid__retains_native_sample_heights_in_closed_mesh(native_scene) -> None:
    heightmap, frame = native_scene
    domain = frame.polygon_enu()
    native = source_grid_points(heightmap, frame, domain)
    assert len(native) > 20
    mesh = build_land_mesh(frame, heightmap, [], base_z=90, grid_points_per_side=12,
                           native_source_grid=True)
    top = mesh.vertices[mesh.vertices[:, 2] > 90]
    distances, indices = cKDTree(top[:, :2]).query(native)
    assert distances.max() < 1e-6
    np.testing.assert_allclose(top[indices, 2], sample_at_enu(heightmap, native, frame.local_enu()), atol=1e-6)
    assert mesh.is_watertight and mesh.is_winding_consistent


def test__source_grid__excludes_water_holes(native_scene) -> None:
    heightmap, frame = native_scene
    water = shapely.box(-10, -10, 10, 10)
    points = source_grid_points(heightmap, frame, frame.polygon_enu().difference(water))
    assert not shapely.contains_xy(water, points[:, 0], points[:, 1]).any()


def test__source_grid__reports_native_sample_budget(native_scene, monkeypatch) -> None:
    heightmap, frame = native_scene
    monkeypatch.setattr('contour.terrain_refinement.MAX_VERTICES', 1)
    with pytest.raises(MeshDetailLimitError) as caught:
        source_grid_points(heightmap, frame, frame.polygon_enu())
    assert caught.value.details['stage'] == 'native_source_grid'


def test__source_grid__honours_cancellation(native_scene) -> None:
    heightmap, frame = native_scene
    def cancel() -> None:
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        source_grid_points(heightmap, frame, frame.polygon_enu(), cancel)
