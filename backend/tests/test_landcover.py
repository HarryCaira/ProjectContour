"""Land cover preserves holes, handles overlaps, and only colours mapped areas."""
import base64
import io

import mapbox_vector_tile
import numpy as np
from PIL import Image
from shapely.geometry import box, Polygon

import contour.landcover as coverage
from contour.hex_frame import HexFrame
from contour.tile_cache import TileCache


def test_mask_preserves_holes_and_axis_orientation():
    domain = box(0, 0, 10, 10)
    wood = box(0, 5, 5, 10).difference(box(1, 6, 2, 7))
    rock = box(6, 0, 10, 4)
    result = coverage.coverage_preview({'wood': wood, 'rock': rock}, domain, 101)
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1]))))
    assert tuple(pixels[10, 10]) == (255, 0, 0)
    assert tuple(pixels[35, 15]) == (0, 0, 0)  # hole
    assert tuple(pixels[90, 90]) == (0, 255, 0)
    assert tuple(pixels[10, 90]) == (0, 0, 0)  # unknown stays ordinary terrain
    assert result['percentages'] == {'wood': 24, 'rock': 16}


def test_fetch_reads_only_mapped_classes_and_resolves_overlaps(monkeypatch, tmp_path):
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=100)
    wood, rock, water = box(-40,-40,40,40), box(0,-20,60,20), box(10,-10,30,10)
    tiles = mapbox_vector_tile.encode([
        {'name':'landuse','features':[
            {'geometry':wood,'properties':{'class':'wood'}},
            {'geometry':rock,'properties':{'class':'rock'}},
            {'geometry':box(-80,-80,80,80),'properties':{'class':'park'}}]},
        {'name':'water','features':[{'geometry':water,'properties':{}}]},
    ])
    monkeypatch.setattr(coverage, 'tiles_covering_bbox', lambda *args: [object(), object()])
    monkeypatch.setattr(coverage, '_fetch_tile_with_cache', lambda *args: tiles)
    monkeypatch.setattr(coverage, '_convert_to_lonlat', lambda geom,*args: geom)
    monkeypatch.setattr(coverage, '_project_to_enu', lambda geom,*args: geom)
    result = coverage.fetch_landcover(frame, object(), TileCache(tmp_path), 'test')
    area = frame.polygon_enu().area
    assert result['percentages']['wood'] == round(100*wood.difference(rock.union(water)).area/area,2)
    assert result['percentages']['rock'] == round(100*rock.difference(water).area/area,2)
