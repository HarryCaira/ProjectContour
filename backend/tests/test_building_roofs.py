import json
from unittest.mock import Mock

import numpy as np
import pytest
import requests
import shapely
import trimesh

from contour.building_roofs import RoofFeature, apply_roof, fetch_roofs, matched_roof, roof_spec
from contour.hex_frame import HexFrame
from contour.infrastructure import BuildingCandidate, build_buildings
from contour.stl_export import _validated_stl
from contour.tile_cache import TileCache


@pytest.fixture
def land():
    return trimesh.creation.box(extents=[20,20,2])


@pytest.mark.parametrize('kind,expected_volume', [('gabled',20),('hipped',56/3),('pyramidal',56/3),('skillion',20)])
def test_mapped_roofs_are_closed_and_preserve_total_height(land, kind, expected_volume):
    polygon = shapely.box(-2,-1,2,1)
    tags = {'roof:shape':kind,'roof:height':'1','roof:direction':'N'}
    roof = roof_spec(polygon,tags,1,3)
    mesh, _ = build_buildings(land,[BuildingCandidate(polygon,3,roof=roof)],shapely.Polygon())
    assert roof is not None and mesh.is_volume
    assert mesh.bounds[1,2] == pytest.approx(4)
    # Hipped roof has a finite ridge, so more volume than a pyramid.
    if kind != 'hipped':
        assert mesh.volume == pytest.approx(expected_volume)
    else:
        assert 56/3 < mesh.volume < 20
    assert _validated_stl('buildings',mesh,repair=False)
    assert mesh.vertex_attributes['_route_offset'][:,2].max() == pytest.approx(3)


@pytest.mark.parametrize('tags',[
    {}, {'roof:shape':'dome','roof:height':'1'}, {'roof:shape':'gabled'},
    {'roof:shape':'gabled','roof:height':'nan'}, {'roof:shape':'gabled','roof:height':'.01'},
    {'roof:shape':'gabled','roof:height':'4'}, {'roof:shape':'skillion','roof:height':'1'},
    {'roof:shape':'gabled','roof:height':'1','roof:direction':'bad'},
])
def test_missing_unprintable_or_ambiguous_tags_stay_flat(tags):
    assert roof_spec(shapely.box(-2,-1,2,1),tags,1,3) is None


def test_complex_footprint_stays_flat():
    polygon=shapely.box(0,0,4,4).difference(shapely.box(2,2,4,4))
    assert roof_spec(polygon,{'roof:shape':'gabled','roof:height':'1'},1,3) is None


def test_match_requires_close_unambiguous_footprints():
    p=shapely.box(-2,-1,2,1)
    features=[RoofFeature(p,{'roof:shape':'gabled','roof:height':'1'})]
    assert matched_roof(p,features,shapely.STRtree([p]),1,3,0) is not None
    assert matched_roof(shapely.box(0,0,1,1),features,shapely.STRtree([p]),1,3,0) is None
    assert matched_roof(p,features*2,shapely.STRtree([p,p]),1,3,0) is None


def test_clipping_does_not_move_ridge(land):
    p=shapely.box(-2,-1,2,1)
    roof=roof_spec(p,{'roof:shape':'gabled','roof:height':'1','roof:direction':'N'},1,3)
    mesh,_=build_buildings(land,[BuildingCandidate(p,3,roof=roof)],shapely.box(-3,.5,3,3))
    ridge=mesh.vertices[np.isclose(mesh.vertices[:,2],4)]
    assert len(ridge)>0 and np.allclose(ridge[:,1],0)
    assert _validated_stl('buildings',mesh,repair=False)


def test_roof_lookup_cached_and_scale_independent(tmp_path):
    frame=HexFrame(0,0,100)
    cache=TileCache(tmp_path)
    source=Mock()
    source.post.return_value=json.dumps({'elements':[{'tags':{'building':'yes','roof:shape':'gabled','roof:height':'2'},
        'geometry':[{'lon':x,'lat':y} for x,y in [(0,0),(.0001,0),(.0001,.0001),(0,.0001),(0,0)]]}]}).encode()
    first=fetch_roofs(frame,cache,.1,source)
    second=fetch_roofs(frame,cache,.2,source)
    assert len(first)==len(second)==1
    assert second[0].footprint.area == pytest.approx(first[0].footprint.area*4)
    source.post.assert_called_once()


@pytest.mark.parametrize('reply',[b'broken',b'{"remark":"timeout","elements":[]}',b'[]'])
def test_bad_response_is_not_cached(tmp_path,reply):
    source=Mock();source.post.return_value=reply
    for _ in range(2):
        assert fetch_roofs(HexFrame(0,0,100),TileCache(tmp_path),.1,source)==[]
    assert source.post.call_count==2


def test_timeout_and_regional_models_keep_existing_buildings(tmp_path):
    source=Mock();source.post.side_effect=requests.Timeout()
    assert fetch_roofs(HexFrame(0,0,100),TileCache(tmp_path),.1,source)==[]
    assert fetch_roofs(HexFrame(0,0,10000),TileCache(tmp_path),.1,source)==[]
    source.post.assert_called_once()


def test_malformed_features_are_ignored(tmp_path):
    source=Mock()
    source.post.return_value=json.dumps({'elements':[{}, {'tags':'building'},
        {'tags':{'building':'yes'},'geometry':[]}, {'tags':{'building':'yes'},'geometry':[{'lat':0}]}]}).encode()
    assert fetch_roofs(HexFrame(0,0,100),TileCache(tmp_path),.1,source)==[]


@pytest.mark.parametrize('kind', ['dome', 'cone', 'round'])
def test_curved_roof_is_printable_and_matches_height(land, kind):
    polygon = shapely.box(-2,-1,2,1) if kind == 'round' else shapely.Point(0,0).buffer(2, quad_segs=32)
    roof = roof_spec(polygon, {'roof:shape':kind,'roof:height':'1'}, 1, 3)
    assert roof is not None
    mesh, _ = build_buildings(land, [BuildingCandidate(polygon,3,roof=roof)], shapely.Polygon())
    assert mesh.is_volume and mesh.bounds[1,2] == pytest.approx(4, abs=.01)
    assert mesh.volume < polygon.area*3
    assert _validated_stl('buildings',mesh)
    assert mesh.vertex_attributes['_route_offset'][:,2].max() == pytest.approx(3, abs=.01)


@pytest.mark.parametrize('kind,expected', [('gabled',1), ('hipped',1), ('skillion',2)])
def test_mapped_pitch_determines_rise(kind, expected):
    roof = roof_spec(shapely.box(-2,-1,2,1),
                     {'roof:shape':kind,'roof:angle':'45','roof:direction':'N'},1,4)
    assert roof is not None and roof.rise == pytest.approx(expected)


@pytest.mark.parametrize('angle',['nan','0','90','-5','bad'])
def test_invalid_pitch_does_not_invent_roof(angle):
    assert roof_spec(shapely.box(-2,-1,2,1),{'roof:shape':'gabled','roof:angle':angle},1,3) is None


def test_curved_roof_rejects_wrong_footprint_and_missing_height():
    assert roof_spec(shapely.box(-2,-1,2,1),{'roof:shape':'dome','roof:height':'1'},1,3) is None
    assert roof_spec(shapely.Point(0,0).buffer(2),{'roof:shape':'dome'},1,3) is None


def test_roof_pitch_respects_physical_print_threshold():
    tags={'roof:shape':'gabled','roof:angle':'10','roof:direction':'N'}
    assert roof_spec(shapely.box(-.4,-.3,.4,.3), tags, .1, .5) is None
    assert roof_spec(shapely.box(-4,-3,4,3), tags, 1, 5) is not None


def test_fractional_mapped_roof_height_without_leading_zero():
    roof = roof_spec(shapely.box(-2,-1,2,1),{'roof:shape':'gabled','roof:height':'.5'},1,3)
    assert roof is not None and roof.rise == pytest.approx(.5)


@pytest.mark.parametrize('base', [0, .5])
def test_irregular_single_slope_can_reach_base(land, base):
    polygon=shapely.Polygon([(-2,-1),(2,-1),(0,1)])
    roof=roof_spec(polygon,{'roof:shape':'skillion','roof:height':'3','roof:direction':'N'},1,3+base,base)
    assert roof is not None
    mesh,_=build_buildings(land,[BuildingCandidate(polygon,3+base,base,roof=roof)],shapely.Polygon())
    assert mesh.is_volume
    assert mesh.bounds[1,2] == pytest.approx(4+base)
    assert mesh.volume < polygon.area*3
    assert _validated_stl('buildings',mesh,repair=False)


def test_irregular_slope_still_requires_direction_and_consistent_heights():
    polygon=shapely.Polygon([(-2,-1),(2,-1),(0,1)])
    tags={'roof:shape':'skillion','roof:height':'3'}
    assert roof_spec(polygon,tags,1,3) is None
    assert roof_spec(polygon,{**tags,'roof:direction':'N'},1,2) is None
    assert roof_spec(polygon,{**tags,'roof:direction':'N','roof:shape':'gabled'},1,4) is None


def test_ground_reaching_slopes_merge_and_trim_on_sloping_terrain():
    land=trimesh.creation.box(extents=[20,20,2])
    land.vertices[:,2] += land.vertices[:,0]*.05
    candidates=[]
    for p,direction in [(shapely.Polygon([(-2,-1),(2,-1),(0,1)]),'N'),
                        (shapely.Polygon([(-2,1),(2,1),(0,-1)]),'S')]:
        roof=roof_spec(p,{'roof:shape':'skillion','roof:height':'3','roof:direction':direction},1,3)
        candidates.append(BuildingCandidate(p,3,roof=roof))
    mesh,_=build_buildings(land,candidates,shapely.box(1,-3,3,3))
    assert mesh.is_volume
    assert _validated_stl('buildings',mesh)
