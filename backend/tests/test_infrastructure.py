import numpy as np
import pytest
import shapely
from shapely.geometry import box
import trimesh
from contour.infrastructure import build_buildings, fetch_infrastructure


def test_building_has_flat_roof_closed_base_and_no_terrain_overlap():
    land = trimesh.creation.box(extents=[10, 10, 2])
    land.vertices[:, 2] += land.vertices[:, 0]*.1 + 2
    building, footprint = build_buildings(land, [(box(-1,-1,1,1), .8)], shapely.Polygon())
    assert building is not None and building.is_volume
    assert footprint.area == pytest.approx(4)
    overlap = trimesh.boolean.intersection([land, building], engine='manifold')
    assert overlap.is_empty or abs(overlap.volume) < 1e-5
    offsets = building.vertex_attributes['_route_offset'][:,2]
    roof = offsets > 0
    assert roof.any() and (~roof).any()
    assert np.ptp(building.vertices[roof,2]) < 1e-6
    assert offsets.max() == pytest.approx(.8)


def test_buildings_union_overlaps_and_avoid_water():
    land = trimesh.creation.box(extents=[10,10,2])
    candidates = [(box(-2,-2,0,0),1), (box(-1,-1,1,1),1), (box(2,2,3,3),1)]
    mesh, footprint = build_buildings(land, candidates, box(2,2,3,3))
    assert mesh.is_volume
    assert footprint.area == pytest.approx(7)


def test_feature_fetch_keeps_small_roads_but_keeps_bridges_and_filters_small_buildings(monkeypatch):
    import contour.infrastructure as module
    from contour.hex_frame import HexFrame
    from contour.tiles import RasterTile
    tile = RasterTile(14,8192,8192)
    features = {
      'road': {'extent':4096, 'features': [
        {'geometry':{'type':'LineString','coordinates':[[0,4000],[1000,4000]]},'properties':{'class':'primary'}},
        {'geometry':{'type':'LineString','coordinates':[[0,3000],[1000,3000]]},'properties':{'class':'street'}},
        {'geometry':{'type':'LineString','coordinates':[[0,2000],[1000,2000]]},'properties':{'class':'primary','structure':'bridge'}},
      ]},
      'building': {'extent':4096, 'features': [
        {'id':1,'geometry':{'type':'Polygon','coordinates':[[[100,3500],[400,3500],[400,3800],[100,3800],[100,3500]]]},'properties':{'height':12}},
        {'id':2,'geometry':{'type':'Polygon','coordinates':[[[500,3500],[501,3500],[501,3501],[500,3501],[500,3500]]]},'properties':{}},
      ]}}
    monkeypatch.setattr(module,'tiles_covering_bbox',lambda *args:[tile])
    monkeypatch.setattr(module,'_fetch_tile_with_cache',lambda *args:b'')
    monkeypatch.setattr(module.mapbox_vector_tile,'decode',lambda *args:features)
    frame=HexFrame(centre_lon=0,centre_lat=0,circumradius_m=2000)
    roads, buildings, bridges=fetch_infrastructure(frame,None,None,'token',.025)
    assert not bridges.is_empty
    assert len(buildings)==1
    assert buildings[0][1]==pytest.approx(.4)  # print-height minimum
    assert not roads.is_empty
    assert len(list(shapely.get_parts(roads))) == 2  # Primary and street, no bridge.
    assert all(p.bounds[3]-p.bounds[1] == pytest.approx(.2,abs=.001) for p in shapely.get_parts(roads))


def test_minimum_width_road_survives_surface_filter():
    from contour.landcover_mesh import split_surface_material
    land = trimesh.creation.box(extents=[10,10,2])
    base, road = split_surface_material(land, box(-4,-.2,4,.2), min_feature_mm=.3)
    assert road is not None and road.is_volume and base.is_volume
    assert road.extents[1] == pytest.approx(.4, abs=1e-5)


def test_shared_walls_keep_stepped_roofs_and_export_cleanly():
    from contour.stl_export import _validated_stl
    land = trimesh.creation.box(extents=[10,10,2])
    mesh, footprint = build_buildings(land, [(box(-1,-1,0,1),1), (box(0,-1,1,1),2)], shapely.Polygon())
    assert footprint.area == pytest.approx(4)
    assert mesh.is_volume and mesh.volume == pytest.approx(6)
    assert set(np.round(mesh.vertex_attributes['_route_offset'][:,2], 5)) == {0,1,2}
    assert _validated_stl('buildings', mesh, repair=False)


def test_small_terrace_houses_survive_as_connected_block():
    land = trimesh.creation.box(extents=[10,10,2])
    candidates = [(box(i*.2,0,(i+1)*.2,1), .5) for i in range(5)]
    mesh, footprint = build_buildings(land, candidates, shapely.Polygon())
    assert mesh.is_volume and mesh.volume == pytest.approx(.5)
    assert footprint.area == pytest.approx(1)


def test_route_trims_building_instead_of_discarding_it():
    land = trimesh.creation.box(extents=[10,10,2])
    clearance = box(.5,-2,2,2)
    mesh, footprint = build_buildings(land, [(box(-1,-1,1,1), 1)], clearance)
    assert mesh.is_volume
    assert footprint.area == pytest.approx(3)
    assert footprint.intersection(clearance).area == 0


def test_trimming_discards_unprintable_sliver():
    land = trimesh.creation.box(extents=[10,10,2])
    mesh, footprint = build_buildings(land, [(box(-1,-1,1,1),1)], box(-.9,-2,2,2))
    assert mesh is None and footprint.is_empty


def test_ground_height_uses_footprint_not_distant_triangle_corners():
    land = trimesh.creation.box(extents=[10,10,2])
    land.vertices[:,2] += land.vertices[:,0]*.1
    mesh, _ = build_buildings(land, [(box(-1,-1,1,1),1)], shapely.Polygon())
    assert mesh.bounds[1,2] == pytest.approx(2.1)


def test_raised_building_part_preserves_minimum_height():
    from contour.infrastructure import BuildingCandidate
    land = trimesh.creation.box(extents=[10,10,2])
    mesh, _ = build_buildings(land, [BuildingCandidate(box(-1,-1,1,1), 2, .5)], shapely.Polygon())
    assert mesh.is_volume
    assert mesh.bounds[:,2] == pytest.approx([1.5,3])


def test_building_source_uses_z16_keeps_parts_and_tracks_fallback(monkeypatch):
    import contour.infrastructure as module
    from contour.hex_frame import HexFrame
    from contour.tiles import RasterTile
    requested = []
    def tiles(west, south, east, north, zoom):
        requested.append(zoom)
        return [RasterTile(zoom, 2**(zoom-1), 2**(zoom-1))]
    features = []
    for i, props in enumerate(({'type':'building', 'extrude':'false'},
                                {'type':'building:part', 'height':12, 'min_height':3},
                                {'height':'invalid'}, {'underground':'true'})):
        features.append({'id':i, 'geometry':shapely.geometry.mapping(box(100+i*500,3000,500+i*500,3400)),
                         'properties':props})
    monkeypatch.setattr(module,'tiles_covering_bbox',tiles)
    monkeypatch.setattr(module,'_fetch_tile_with_cache',lambda *args:b'')
    monkeypatch.setattr(module.mapbox_vector_tile,'decode',lambda *args:{'building':{'extent':4096,'features':features}})
    _, candidates, _ = fetch_infrastructure(HexFrame(centre_lon=0,centre_lat=0,circumradius_m=1000),None,None,'token',.05)
    assert requested == [16,16]
    assert len(candidates)==2
    part, fallback = candidates
    assert part.height == pytest.approx(.6) and part.min_height == pytest.approx(.15)
    assert not part.estimated_height and fallback.estimated_height
    assert fallback.height == pytest.approx(.4)


def test_courtyard_and_nearly_duplicate_vertices_remain_valid():
    from contour.stl_export import _validated_stl
    polygon = shapely.Polygon([(-2,-2),(2,-2),(2,2),(2-1e-12,2),(-2,2)],
                              holes=[[(-1,-1),(-1,1),(1,1),(1,-1)]])
    land = trimesh.creation.box(extents=[10,10,2])
    mesh, footprint = build_buildings(land, [(polygon,1)], shapely.Polygon())
    assert mesh.is_volume and mesh.volume == pytest.approx(12)
    assert not footprint.contains(shapely.Point(0,0))
    assert _validated_stl('buildings',mesh)


@pytest.mark.parametrize('width,retained', [(.30,True),(.26,True),(.20,False)])
def test_small_building_density_filter_preserves_valid_solids(width, retained):
    from contour.stl_export import _validated_stl
    land = trimesh.creation.box(extents=[10,10,2])
    mesh, footprint = build_buildings(land, [(box(0,0,width,.5),.4)], shapely.Polygon())
    if retained:
        assert mesh is not None and mesh.is_volume
        assert footprint.area == pytest.approx(width*.5)
        assert _validated_stl('buildings',mesh,repair=False)
    else:
        assert mesh is None and footprint.is_empty


@pytest.mark.parametrize('road_class,width', [('street',6),('service',3),('track',3),('pedestrian',4),('path',1.5)])
def test_smaller_roads_follow_print_scale(road_class,width):
    from contour.infrastructure import printable_road_width
    props={'class':road_class}
    assert printable_road_width(props,.199/width)*(.199/width) == pytest.approx(.2)
    assert printable_road_width(props,.2/width) == width
    assert printable_road_width(props,.3/width) == width


def test_minimum_road_width_and_tunnel_exclusion():
    from contour.infrastructure import printable_road_width
    assert printable_road_width({'class':'motorway'},.01) == pytest.approx(20)
    assert printable_road_width({'class':'street','structure':'bridge'},1) == 6
    for structure in ['tunnel']:
        assert printable_road_width({'class':'street','structure':structure},1) is None
    assert printable_road_width({'class':'rail'},1) is None


def test_point_two_mm_road_is_separate_printable_solid():
    from contour.infrastructure import MIN_ROAD_MM
    from contour.landcover_mesh import split_surface_material
    from contour.stl_export import _validated_stl
    land=trimesh.creation.box(extents=[10,10,2])
    base,road=split_surface_material(land,box(-4,-.1,4,.1),min_feature_mm=MIN_ROAD_MM-.001)
    assert road is not None and road.is_volume and base.is_volume
    assert road.extents[1] == pytest.approx(.2,abs=1e-6)
    assert _validated_stl('roads',road)
