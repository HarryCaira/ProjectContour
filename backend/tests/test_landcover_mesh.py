import io
import json
import zipfile

import numpy as np
import pytest
import trimesh
from shapely.geometry import box

from contour.landcover_mesh import split_surface_material
from contour.kit import MeshKit, KitPart, Material
from contour.stl_export import to_stl_zip


@pytest.mark.parametrize('slope', [0., .5, 2.])
def test_surface_partition_preserves_shape_and_volume(slope):
    land = trimesh.creation.box(extents=[10, 10, 4])
    land.apply_translation([0, 0, 2])
    land.vertices[:, 2] += slope * (land.vertices[:, 0] + 5)
    before = land.vertices.copy()
    base, wood = split_surface_material(land, box(-3, -3, 3, 3))
    assert wood is not None and wood.is_volume and base.is_volume
    assert base.volume + wood.volume == pytest.approx(land.volume, rel=1e-5)
    overlap = trimesh.boolean.intersection([base, wood], engine='manifold')
    assert overlap.is_empty or abs(overlap.volume) < 1e-5
    reunited = trimesh.boolean.union([base, wood], engine='manifold')
    assert reunited.volume == pytest.approx(land.volume, rel=1e-5)
    offsets = wood.vertices[:, 2] - (4 + slope * (wood.vertices[:, 0] + 5))
    assert offsets.min() == pytest.approx(-.6, abs=1e-5)
    assert offsets.max() == pytest.approx(0, abs=1e-5)
    np.testing.assert_array_equal(land.vertices, before)
    data = to_stl_zip(MeshKit([KitPart('land', base, Material('#777777')), KitPart('woodland', wood, Material('#365d38'))]))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert 'woodland.stl' in archive.namelist()
        restored = trimesh.load(io.BytesIO(archive.read('woodland.stl')), file_type='stl')
        assert restored.is_volume
        assert json.loads(archive.read('manifest.json'))['parts'][1]['name'] == 'woodland'


def test_thin_terrain_keeps_foundation():
    land = trimesh.creation.box(extents=[10,10,.5]); land.apply_translation([0,0,.25])
    base, wood = split_surface_material(land, box(-10,-10,10,10))
    assert wood.bounds[0,2] == pytest.approx(.3)
    assert base.bounds[1,2] == pytest.approx(.3)
    assert base.volume + wood.volume == pytest.approx(land.volume)


@pytest.mark.parametrize('region', [box(20,20,30,30), box(0,0,.1,.1), box(-4,0,4,.1)])
def test_unprintable_or_outside_coverage_leaves_land_unchanged(region):
    land = trimesh.creation.box(extents=[10,10,4])
    base, wood = split_surface_material(land, region)
    assert base is land and wood is None
