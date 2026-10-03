import numpy as np
import pytest
import trimesh
from shapely.geometry import box

from contour.landcover_mesh import split_surface_material
from contour.woodland_texture import add_woodland_texture, BUMP_HEIGHT_MM


@pytest.mark.parametrize('slope', [0., 1.5])
def test_texture_is_raised_closed_and_keeps_interface(slope):
    land = trimesh.creation.box(extents=[8,8,4]); land.apply_translation([0,0,2])
    land.vertices[:,2] += slope*(land.vertices[:,0]+4)
    region = box(-3,-3,3,3)
    base, insert = split_surface_material(land, region)
    textured = add_woodland_texture(insert, region, base)
    assert textured.is_volume and textured.volume > insert.volume
    heights = textured.vertices[:,2] - (4+slope*(textured.vertices[:,0]+4))
    assert .1 < heights.max() <= BUMP_HEIGHT_MM + 1e-5
    assert heights.min() == pytest.approx(-.6, abs=1e-5)
    overlap = trimesh.boolean.intersection([base, textured], engine='manifold')
    assert overlap.is_empty or abs(overlap.volume) < 1e-5


def test_route_corridor_stays_clear():
    land = trimesh.creation.box(extents=[8,8,4]); land.apply_translation([0,0,2])
    region = box(-3,-3,3,3)
    base, insert = split_surface_material(land, region)
    route = trimesh.creation.box(extents=[1,8,1]); route.apply_translation([0,0,4.5])
    textured = add_woodland_texture(insert, region, base, route)
    raised = textured.vertices[textured.vertices[:,2] > 4.001]
    assert len(raised) and np.abs(raised[:,0]).min() >= .65-1e-5


def test_canopy_scales_with_geographic_extent_and_stays_printable():
    from contour.woodland_texture import woodland_scale
    # Same 100 mm print, showing 5 km versus 10 km.
    assert woodland_scale(100/10000) == woodland_scale(100/5000)/2
    assert woodland_scale(150/5000) == woodland_scale(100/5000)*1.5
    assert woodland_scale(100/100000) == .5
    assert woodland_scale(100/100) == 2


def test_larger_geographic_extent_produces_lower_canopies():
    land = trimesh.creation.box(extents=[8,8,4]); land.apply_translation([0,0,2])
    region = box(-3,-3,3,3)
    base, insert = split_surface_material(land, region)
    near = add_woodland_texture(insert, region, base, mm_per_m=.02)
    far = add_woodland_texture(insert, region, base, mm_per_m=.01)
    assert near.is_volume and far.is_volume
    assert .5 < near.bounds[1,2]-4 <= .7+1e-5
    assert .2 < far.bounds[1,2]-4 <= .35+1e-5
