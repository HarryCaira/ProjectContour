"""Biome data acquisition: Mapbox vector tiles -> water polygons in ENU."""
from __future__ import annotations

import math

import mapbox_vector_tile
import numpy as np
import shapely
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from contour.framing.hex import HexFrame
from contour.geo.tiles import RasterTile, tiles_covering_bbox
from contour.geo.transforms import LocalENU
from contour.http.cache import TileCache
from contour.http.client import HttpClient

PROVIDER = "mapbox"
LAYER = "streets-v8"
BASE_URL = "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8"
DEFAULT_ZOOM = 14


def fetch_water_polygons(
    hex_frame: HexFrame,
    client: HttpClient,
    cache: TileCache,
    mapbox_token: str,
    zoom: int = DEFAULT_ZOOM,
) -> list[Polygon]:
    """Fetch water polygons covering the hex from Mapbox vector tiles, clipped to the hex.

    Returns a list of Polygons in ENU coordinates anchored on the hex centre.
    """
    west, south, east, north = _hex_geographic_bbox(hex_frame)
    tiles = tiles_covering_bbox(west, south, east, north, zoom)

    enu = hex_frame.local_enu()
    hex_polygon = hex_frame.polygon_enu()

    polygons_enu: list[Polygon] = []
    for tile in tiles:
        mvt_bytes = _fetch_tile_with_cache(client, cache, mapbox_token, tile)
        polygons_enu.extend(extract_water_polygons_enu(mvt_bytes, tile, enu))

    if not polygons_enu:
        return []

    merged = unary_union(polygons_enu)
    clipped = merged.intersection(hex_polygon)
    return _flatten_polygons(clipped)


def extract_water_polygons_enu(
    mvt_bytes: bytes, tile: RasterTile, enu: LocalENU
) -> list[Polygon]:
    """Decode a vector tile and return its water polygons reprojected to ENU.

    Independent from network and cache concerns so it can be unit-tested with
    synthetic encoded tiles.
    """
    decoded = mapbox_vector_tile.decode(mvt_bytes)
    water_layer = decoded.get("water")
    if not water_layer:
        return []
    extent = water_layer.get("extent", 4096)

    polygons: list[Polygon] = []
    for feature in water_layer["features"]:
        geom = feature["geometry"]
        if geom["type"] not in ("Polygon", "MultiPolygon"):
            continue
        geom_lonlat = shape(_convert_to_lonlat(geom, tile, extent))
        if not geom_lonlat.is_valid or geom_lonlat.is_empty:
            continue
        geom_enu = _project_to_enu(geom_lonlat, enu)
        polygons.extend(_flatten_polygons(geom_enu))
    return polygons


def _convert_to_lonlat(geom: dict, tile: RasterTile, extent: int) -> dict:
    """Recursively convert MVT extent coordinates to (lon, lat).

    mapbox_vector_tile decodes with y-axis flipped by default (y=0 is south, y=extent
    is north), so we convert that to a tile-coord-space y-down value first.
    """
    n = 2**tile.zoom

    def point(p: tuple[float, float]) -> list[float]:
        x_ext, y_ext = p
        tile_x_frac = tile.x + x_ext / extent
        tile_y_frac = tile.y + (1 - y_ext / extent)
        lon = tile_x_frac / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y_frac / n))))
        return [lon, lat]

    def rings(rs):
        return [[point(p) for p in ring] for ring in rs]

    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": rings(geom["coordinates"])}
    return {"type": "MultiPolygon", "coordinates": [rings(p) for p in geom["coordinates"]]}


def _project_to_enu(geom, enu: LocalENU):
    """Project a shapely geometry from (lon, lat) to ENU (E, N)."""

    def project(coords: np.ndarray) -> np.ndarray:
        lons = coords[:, 0]
        lats = coords[:, 1]
        enu_xyz = enu.to_enu(lats, lons, 0.0)
        return enu_xyz[:, :2]

    return shapely.transform(geom, project)


def _flatten_polygons(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if g.geom_type == "Polygon"]
    return []


def _fetch_tile_with_cache(
    client: HttpClient, cache: TileCache, token: str, tile: RasterTile
) -> bytes:
    cached = cache.get(PROVIDER, LAYER, tile.zoom, tile.x, tile.y, "mvt")
    if cached is not None:
        return cached
    url = f"{BASE_URL}/{tile.zoom}/{tile.x}/{tile.y}.mvt"
    data = client.get(url, params={"access_token": token})
    cache.set(PROVIDER, LAYER, tile.zoom, tile.x, tile.y, "mvt", data)
    return data


def _hex_geographic_bbox(hex_frame: HexFrame) -> tuple[float, float, float, float]:
    enu = hex_frame.local_enu()
    r = hex_frame.circumradius_m
    corners = enu.to_geodetic(
        np.array([-r, r, r, -r]),
        np.array([r, r, -r, -r]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    lats = corners[:, 0]
    lons = corners[:, 1]
    return float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())
