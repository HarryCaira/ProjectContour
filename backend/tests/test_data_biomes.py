"""Tests for biome data: vector tile decoding and water polygon extraction."""
from __future__ import annotations

import math

import mapbox_vector_tile
import numpy as np
import pytest
from shapely.geometry import Polygon

import responses

from contour.data.biomes import (
    _convert_to_lonlat,
    extract_water_polygons_enu,
    fetch_water_polygons,
)
from contour.framing.hex import HexFrame
from contour.geo.tiles import RasterTile
from contour.geo.transforms import LocalENU
from contour.http.cache import TileCache
from contour.http.client import HttpClient


def _encode_water_tile(polygons_in_extent: list[Polygon], extent: int = 4096) -> bytes:
    """Encode a synthetic vector tile with a single 'water' layer."""
    return mapbox_vector_tile.encode(
        [
            {
                "name": "water",
                "features": [
                    {"geometry": poly, "properties": {}} for poly in polygons_in_extent
                ],
            }
        ],
        default_options={"extents": extent},
    )


# ---------- _convert_to_lonlat ----------


def test_convert_polygon_corners_match_tile_bbox():
    tile = RasterTile(zoom=14, x=8192, y=5446)
    extent = 4096
    full_tile_polygon = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (extent, 0), (extent, extent), (0, extent), (0, 0)]],
    }
    converted = _convert_to_lonlat(full_tile_polygon, tile, extent)
    ring = converted["coordinates"][0]

    west, south, east, north = tile.bbox

    # MVT y=0 is south, y=extent is north.
    sw = ring[0]
    se = ring[1]
    ne = ring[2]
    nw = ring[3]

    assert sw[0] == pytest.approx(west, abs=1e-9)
    assert sw[1] == pytest.approx(south, abs=1e-9)
    assert se[0] == pytest.approx(east, abs=1e-9)
    assert se[1] == pytest.approx(south, abs=1e-9)
    assert ne[0] == pytest.approx(east, abs=1e-9)
    assert ne[1] == pytest.approx(north, abs=1e-9)
    assert nw[0] == pytest.approx(west, abs=1e-9)
    assert nw[1] == pytest.approx(north, abs=1e-9)


# ---------- extract_water_polygons_enu ----------


def _tile_at_lonlat(lon: float, lat: float, zoom: int) -> RasterTile:
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return RasterTile(zoom=zoom, x=x, y=y)


def test_extract_returns_empty_when_no_water_layer():
    encoded = mapbox_vector_tile.encode([{"name": "roads", "features": []}])
    tile = _tile_at_lonlat(-0.1, 51.5, 14)
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    assert extract_water_polygons_enu(encoded, tile, enu) == []


def test_extract_single_polygon():
    extent = 4096
    polygon = Polygon([(500, 500), (3500, 500), (3500, 3500), (500, 3500)])
    encoded = _encode_water_tile([polygon], extent=extent)

    tile = _tile_at_lonlat(-0.1, 51.5, 14)
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    result = extract_water_polygons_enu(encoded, tile, enu)

    assert len(result) == 1
    p = result[0]
    assert p.is_valid
    assert not p.is_empty
    assert p.area > 0  # in m^2


def test_extract_multiple_polygons():
    encoded = _encode_water_tile(
        [
            Polygon([(100, 100), (1000, 100), (1000, 1000), (100, 1000)]),
            Polygon([(2000, 2000), (3000, 2000), (3000, 3000), (2000, 3000)]),
        ]
    )
    tile = _tile_at_lonlat(-0.1, 51.5, 14)
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    result = extract_water_polygons_enu(encoded, tile, enu)
    assert len(result) == 2


def test_extract_polygon_is_in_local_enu_scale():
    """A polygon spanning ~half the tile should produce a result on the order of
    tile_size_metres ** 2 / 4 in area, not lat/lon-scaled."""
    extent = 4096
    polygon = Polygon(
        [(extent / 4, extent / 4), (3 * extent / 4, extent / 4), (3 * extent / 4, 3 * extent / 4), (extent / 4, 3 * extent / 4)]
    )
    encoded = _encode_water_tile([polygon], extent=extent)

    zoom = 14
    tile = _tile_at_lonlat(-0.1, 51.5, zoom)
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    result = extract_water_polygons_enu(encoded, tile, enu)
    assert len(result) == 1

    # At lat=51.5, zoom=14, a tile is roughly 1.5 km wide. Half-tile polygon -> ~0.75 km wide -> ~0.5 km^2.
    area_m2 = result[0].area
    assert 100_000 < area_m2 < 1_500_000  # 0.1 - 1.5 km^2


def test_extract_skips_non_polygon_geometries():
    """A LineString feature in the water layer should be ignored."""
    from shapely.geometry import LineString

    encoded = mapbox_vector_tile.encode(
        [
            {
                "name": "water",
                "features": [
                    {"geometry": LineString([(0, 0), (1000, 1000)]), "properties": {}},
                    {"geometry": Polygon([(100, 100), (500, 100), (500, 500), (100, 500)]), "properties": {}},
                ],
            }
        ]
    )
    tile = _tile_at_lonlat(-0.1, 51.5, 14)
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    result = extract_water_polygons_enu(encoded, tile, enu)
    assert len(result) == 1


def test_extract_polygon_near_enu_origin():
    """A polygon centred on (lon0, lat0) should produce ENU coords near zero."""
    extent = 4096
    zoom = 14
    lon, lat = -0.1, 51.5
    tile = _tile_at_lonlat(lon, lat, zoom)

    # Place a small polygon at the centre of the tile.
    cx, cy = extent / 2, extent / 2
    polygon = Polygon([(cx - 50, cy - 50), (cx + 50, cy - 50), (cx + 50, cy + 50), (cx - 50, cy + 50)])
    encoded = _encode_water_tile([polygon], extent=extent)

    enu = LocalENU(lat0=lat, lon0=lon)
    result = extract_water_polygons_enu(encoded, tile, enu)
    assert len(result) == 1

    # Centroid should be within a few hundred metres of (0, 0).
    centroid = result[0].centroid
    assert abs(centroid.x) < 1500
    assert abs(centroid.y) < 1500


# ---------- fetch_water_polygons (integration with mocked HTTP) ----------


def _add_mvt_response(encoded: bytes) -> None:
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.mapbox-streets-v8/.*"),
        body=encoded,
        status=200,
    )


@responses.activate
def test_fetch_water_polygons_end_to_end(tmp_path):
    frame = HexFrame(centre_lon=-0.1, centre_lat=51.5, circumradius_m=400)
    extent = 4096
    encoded = mapbox_vector_tile.encode(
        [
            {
                "name": "water",
                "features": [
                    {
                        "geometry": Polygon([(200, 200), (3800, 200), (3800, 3800), (200, 3800)]),
                        "properties": {},
                    }
                ],
            }
        ],
        default_options={"extents": extent},
    )
    _add_mvt_response(encoded)

    client = HttpClient(backoff_factor=0.0)
    cache = TileCache(tmp_path)
    polygons = fetch_water_polygons(frame, client, cache, mapbox_token="test-token", zoom=14)

    assert len(polygons) > 0
    for p in polygons:
        assert p.is_valid
        assert not p.is_empty
        # Every clipped polygon must lie inside the hex.
        assert frame.polygon_enu().buffer(0.01).contains(p)


@responses.activate
def test_fetch_water_polygons_uses_cache(tmp_path):
    frame = HexFrame(centre_lon=-0.1, centre_lat=51.5, circumradius_m=400)
    encoded = mapbox_vector_tile.encode(
        [
            {
                "name": "water",
                "features": [
                    {"geometry": Polygon([(200, 200), (3800, 200), (3800, 3800), (200, 3800)]), "properties": {}}
                ],
            }
        ],
    )
    _add_mvt_response(encoded)

    client = HttpClient(backoff_factor=0.0)
    cache = TileCache(tmp_path)

    fetch_water_polygons(frame, client, cache, mapbox_token="test-token", zoom=14)
    calls_first = len(responses.calls)

    fetch_water_polygons(frame, client, cache, mapbox_token="test-token", zoom=14)
    calls_second = len(responses.calls)

    assert calls_first > 0
    assert calls_second == calls_first  # no new HTTP requests on second pass


@responses.activate
def test_fetch_water_polygons_returns_empty_when_no_water(tmp_path):
    frame = HexFrame(centre_lon=-0.1, centre_lat=51.5, circumradius_m=400)
    encoded_no_water = mapbox_vector_tile.encode([{"name": "roads", "features": []}])
    _add_mvt_response(encoded_no_water)

    client = HttpClient(backoff_factor=0.0)
    cache = TileCache(tmp_path)
    polygons = fetch_water_polygons(frame, client, cache, mapbox_token="test-token", zoom=14)

    assert polygons == []
