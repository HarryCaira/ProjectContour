import json
import numpy as np
import pytest
from requests import Response, HTTPError
from contour.terrain_data import _fetch_tile_with_cache, decode_terrain_rgb
from contour.tile_cache import TileCache
from contour.tiles import RasterTile, tiles_covering_bbox
from contour.biome_data import vector_zoom, _hex_geographic_bbox
from contour.hex_frame import HexFrame

@pytest.mark.parametrize('message', ['Tile not found', 'Tile does not exist'])
def test_ocean_elevation_is_zero_and_cached(tmp_path, message):
    class Missing:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            response = Response(); response.status_code = 404
            response._content = json.dumps({'message': message}).encode()
            raise HTTPError(response=response)
    client = Missing(); cache = TileCache(tmp_path); tile = RasterTile(7,58,43)
    data = _fetch_tile_with_cache(client, cache, 'token', tile)
    assert np.all(decode_terrain_rgb(data) == 0)
    assert _fetch_tile_with_cache(client, cache, 'token', tile) == data
    assert client.calls == 1

@pytest.mark.parametrize('status,message', [(401,'Unauthorized'),(404,'Tileset does not exist'),(500,'Internal error')])
def test_other_provider_failures_not_replaced_with_ocean(tmp_path,status,message):
    class Broken:
        def get(self,*args,**kwargs):
            response=Response();response.status_code=status
            response._content=json.dumps({'message':message}).encode()
            raise HTTPError(response=response)
    with pytest.raises(HTTPError):
        _fetch_tile_with_cache(Broken(),TileCache(tmp_path),'token',RasterTile(7,58,43))

def test_vector_budget_keeps_local_detail_and_bounds_continental_queries():
    assert vector_zoom(HexFrame(-4,57,2000)) == 14
    frame=HexFrame(-4,55,625000)
    zoom=vector_zoom(frame)
    assert zoom < 14
    assert len(tiles_covering_bbox(*_hex_geographic_bbox(frame),zoom)) <= 256

def test_contact_repair_does_not_buffer_distant_coastlines():
    from shapely.geometry import box
    from contour.biome_data import _join_water_contacts
    first, second, distant = box(0,0,1,1), box(1,1,2,2), box(100,100,101,101)
    repaired = _join_water_contacts([first,second,distant])
    assert repaired.is_valid
    assert len(repaired.geoms) == 2
    assert repaired.intersection(distant).equals(distant)
    assert repaired.area - 3 < .001

def test_triangulation_merges_shared_boundary_vertices(monkeypatch):
    import contour.hex_clip as clip
    from shapely.geometry import Polygon, box
    original = clip.tr.triangulate
    def checked(data, options):
        assert len(data['vertices']) == len(np.unique(data['vertices'], axis=0))
        assert np.all(data['segments'][:,0] != data['segments'][:,1])
        return original(data, options)
    monkeypatch.setattr(clip.tr, 'triangulate', checked)
    # A lake touches the enclosing coastline at one vertex.
    result = clip.triangulate_land(box(0,0,10,10), [Polygon([(0,5),(2,4),(2,6)])], 10)
    assert len(result.triangles) > 0

def test_indexed_coastline_preserves_exact_shoreline_blend():
    import shapely
    from shapely.geometry import Point
    from contour.surface_processing import shoreline_sampler
    polygon=Point(0,0).buffer(100,quad_segs=256)
    sample=lambda xy: np.full(len(xy),10.)
    blend=shoreline_sampler(sample,[polygon],[5.],minimum_width=30.)
    xy=np.column_stack((np.linspace(60,140,1000), np.full(1000,.2)))
    distance=shapely.distance(shapely.points(xy),polygon.boundary)
    t=np.clip(distance/30.,0,1)
    expected=10.-5.*(1-t*t*(3-2*t))
    np.testing.assert_allclose(blend(xy),expected,atol=1e-10)

def test_water_repairs_island_touching_its_own_coastline():
    from shapely.geometry import Polygon, Point
    from contour.biome_data import _join_water_contacts
    water=Polygon([(0,0),(10,0),(10,10),(0,10)], holes=[[(0,5),(2,4),(2,6)]])
    repaired=_join_water_contacts([water])
    assert repaired.is_valid
    assert repaired.contains(Point(.001,5))
    assert not repaired.contains(Point(1,5))

def test_large_map_cleanup_preserves_narrow_coastal_gap():
    import shapely
    from shapely.geometry import box, Point
    from contour.hex_clip import triangulate_land
    result=triangulate_land(box(-500000,-500000,500000,500000),
                            [box(0,0,100,100),box(100.0001,0,200,100)],24)
    triangles=shapely.polygons(result.vertices[result.triangles])
    assert shapely.covers(triangles,Point(100.00005,50)).any()
