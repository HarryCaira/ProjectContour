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


def test_winding_minimum_width_road_keeps_connectivity_and_area():
    from shapely.geometry import LineString
    from contour.stl_export import _validated_stl
    line = LineString([(float(x), float(.12*np.sin(x*3))) for x in np.linspace(-4,4,101)])
    region = line.buffer(.1)
    land = trimesh.creation.box(extents=[10,10,4])
    base, road = split_surface_material(land, region, preserve_network=True)
    assert road is not None and road.is_volume and base.is_volume
    assert len(road.split()) == 1
    assert road.volume == pytest.approx(region.area*.6, rel=1e-5)
    assert base.volume + road.volume == pytest.approx(land.volume, rel=1e-6)
    assert _validated_stl('roads', road, repair=False)


def test_bridge_surface_crosses_water_without_overlap_or_missing_volume():
    from contour.stl_export import _validated_stl
    # Flush river surface with a shallow water insert and supporting terrain.
    original = trimesh.creation.box(extents=[10,10,4])
    land, water = split_surface_material(original, box(-1,-6,1,6))
    road_region = box(-4,-.1,4,.1)
    river = box(-1,-6,1,6)
    land, approaches = split_surface_material(land, road_region.difference(river), preserve_network=True)
    water, crossing = split_surface_material(water, road_region.intersection(river), preserve_network=True)
    road = trimesh.boolean.union([approaches, crossing], engine='manifold')
    assert len(road.split()) == 1
    assert road.extents[:2] == pytest.approx([8,.2], abs=1e-6)
    assert sum(m.volume for m in (land,water,road)) == pytest.approx(original.volume, rel=1e-6)
    for name, mesh in [('land',land),('water',water),('roads',road)]:
        assert mesh.is_volume
        assert _validated_stl(name,mesh,repair=False)
    for other in (land,water):
        overlap = trimesh.boolean.intersection([road,other],engine='manifold')
        assert overlap.is_empty or abs(overlap.volume) < 1e-6


def test_bridge_clipping_dust_does_not_block_real_surface():
    from shapely.geometry import Polygon, MultiPolygon
    # A nearly collinear triangle produced at a clipped building boundary.
    dust = Polygon([(2, 2), (2.1, 2.1), (2.2, 2.2 + 1e-14)])
    region = MultiPolygon([box(-4,-.1,4,.1), dust])
    land = trimesh.creation.box(extents=[10,10,4])
    base, road = split_surface_material(land, region, preserve_network=True)
    assert base.is_volume and road.is_volume
    assert len(road.split()) == 1
    assert road.volume == pytest.approx(8*.2*.6, rel=1e-6)


@pytest.mark.parametrize('width', [1e-10, 1e-8, 1e-7])
def test_road_clipping_slivers_collapse_at_solid_precision(width):
    from shapely.geometry import Polygon, MultiPolygon
    from contour.stl_export import _validated_stl
    # Long, numerically thin fragments can exceed an area-only dust cutoff.
    sliver = Polygon([(2,2), (3,2), (3,2+width)])
    region = MultiPolygon([box(-4,-.1,4,.1), sliver])
    land = trimesh.creation.box(extents=[10,10,4])
    base, road = split_surface_material(land, region, preserve_network=True)
    assert base.is_volume and road.is_volume
    assert len(road.split()) == 1
    assert road.volume == pytest.approx(8*.2*.6, rel=1e-6)
    assert _validated_stl('roads',road,repair=False)
