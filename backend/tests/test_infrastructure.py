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


def test_buildings_avoid_route_water_and_each_other():
    land = trimesh.creation.box(extents=[10,10,2])
    candidates = [(box(-2,-2,0,0),1), (box(-1,-1,1,1),1), (box(2,2,3,3),1)]
    mesh, footprint = build_buildings(land, candidates, box(2,2,3,3))
    assert mesh.is_volume
    assert footprint.area == pytest.approx(4)


def test_feature_fetch_filters_minor_roads_and_small_buildings(monkeypatch):
    import contour.infrastructure as module
    from contour.hex_frame import HexFrame
    from contour.tiles import RasterTile
    tile = RasterTile(14,8192,8192)
    features = {
      'road': {'extent':4096, 'features': [
        {'geometry':{'type':'LineString','coordinates':[[0,4000],[1000,4000]]},'properties':{'class':'primary'}},
        {'geometry':{'type':'LineString','coordinates':[[0,3000],[1000,3000]]},'properties':{'class':'residential'}},
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
    roads, buildings=fetch_infrastructure(frame,None,None,'token',.02)
    assert len(buildings)==1
    assert buildings[0][1]==pytest.approx(.4)  # print-height minimum
    assert not roads.is_empty
    assert roads.bounds[3]-roads.bounds[1] == pytest.approx(.4,abs=.001)


def test_minimum_width_road_survives_surface_filter():
    from contour.landcover_mesh import split_surface_material
    land = trimesh.creation.box(extents=[10,10,2])
    base, road = split_surface_material(land, box(-4,-.2,4,.2), min_feature_mm=.3)
    assert road is not None and road.is_volume and base.is_volume
    assert road.extents[1] == pytest.approx(.4, abs=1e-5)
