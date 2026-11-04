import pytest
import shapely
import trimesh
from shapely.affinity import scale
from contour.rock_transition import natural_rock_region, EDGE_DEPTH_MM
from contour.landcover_mesh import split_surface_material
from contour.stl_export import _validated_stl


@pytest.mark.parametrize('size', [100, 150])
def test_transition_is_repeatable_contained_and_preserves_core(size):
    domain = shapely.box(-size/2, -size/2, size/2, size/2)
    rock = shapely.box(-size*.3, -size*.3, size*.3, size*.3)
    result = natural_rock_region(rock, domain)
    assert result.is_valid
    assert result.equals(natural_rock_region(rock, domain))
    assert 0 < result.area < rock.area
    assert result.difference(rock).area < 1e-8
    assert rock.buffer(-EDGE_DEPTH_MM-.05).difference(result).area < 1e-8


def test_geographic_extent_does_not_change_physical_texture():
    expected = natural_rock_region(shapely.box(-20,-20,20,20), shapely.box(-50,-50,50,50))
    for metres_per_mm in [2, 100, 5000]:
        source = shapely.box(-20*metres_per_mm,-20*metres_per_mm,20*metres_per_mm,20*metres_per_mm)
        physical = scale(source, 1/metres_per_mm, 1/metres_per_mm, origin=(0,0))
        assert natural_rock_region(physical, shapely.box(-50,-50,50,50)).equals(expected)


def test_small_patches_and_shoreline_are_protected():
    domain = shapely.box(-50,-50,50,50)
    small = shapely.box(0,0,1,1)
    assert natural_rock_region(small, domain).equals(small)
    rock = shapely.box(-50,-20,20,20)
    water = shapely.box(20,-20,40,20)
    result = natural_rock_region(rock, domain, water)
    assert result.intersection(shapely.box(19.9,-19,20,19)).equals(rock.intersection(shapely.box(19.9,-19,20,19)))
    assert result.intersection(shapely.box(-50,-19,-49.9,19)).equals(rock.intersection(shapely.box(-50,-19,-49.9,19)))


@pytest.mark.parametrize("size", [100, 150])
def test_textured_rock_exports_complementary_closed_solids(size):
    land = trimesh.creation.box(extents=[size,size,4])
    land.vertices[:,2] += land.vertices[:,0]*.2
    region = natural_rock_region(shapely.box(-size*.3,-size*.3,size*.3,size*.3), shapely.box(-size/2,-size/2,size/2,size/2))
    base, rock = split_surface_material(land, region)
    assert base.volume + rock.volume == pytest.approx(land.volume, rel=1e-5)
    for name, mesh in [('land',base),('rock',rock)]:
        assert mesh.is_volume
        assert _validated_stl(name, mesh)
