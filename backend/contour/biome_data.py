"""Biome data acquisition: Mapbox vector tiles -> water polygons in ENU."""
from __future__ import annotations

import math

import mapbox_vector_tile
import numpy as np
import shapely
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from contour.hex_frame import HexFrame
from contour.tiles import RasterTile, tiles_covering_bbox, lonlat_to_tile
from contour.coordinates import LocalENU
from contour.tile_cache import TileCache
from contour.http_client import HttpClient

PROVIDER = "mapbox"
LAYER = "streets-v8"
BASE_URL = "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8"
DEFAULT_ZOOM = 14


def vector_zoom(frame: HexFrame, requested: int = DEFAULT_ZOOM, max_tiles: int = 256) -> int:
    """Keep regional models bounded; retain requested detail on local models."""
    west, south, east, north = _hex_geographic_bbox(frame)
    for zoom in range(requested, -1, -1):
        nw, se = lonlat_to_tile(west, north, zoom), lonlat_to_tile(east, south, zoom)
        if (abs(se.x - nw.x) + 1) * (abs(se.y - nw.y) + 1) <= max_tiles:
            return zoom
    return 0


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
    zoom = vector_zoom(hex_frame, zoom)
    tiles = tiles_covering_bbox(west, south, east, north, zoom)

    enu = hex_frame.local_enu()
    hex_polygon = hex_frame.polygon_enu()

    polygons_enu: list[Polygon] = []
    for tile in tiles:
        mvt_bytes = _fetch_tile_with_cache(client, cache, mapbox_token, tile)
        polygons_enu.extend(extract_water_polygons_enu(mvt_bytes, tile, enu))

    if not polygons_enu:
        return []

    if zoom < DEFAULT_ZOOM:
        # Discard sub-visible coastline noise before expensive contact repair.
        # At 100 mm this is at most 0.001 mm of horizontal displacement.
        tolerance = 2 * hex_frame.circumradius_m * 1e-5
        polygons_enu = [p.simplify(tolerance, preserve_topology=True) for p in polygons_enu]
    merged = _join_water_contacts(polygons_enu)
    clipped = merged.intersection(hex_polygon)
    return _flatten_polygons(clipped)


def _join_water_contacts(polygons: list[Polygon]):
    """Give point contacts a tiny finite neck so extruded solids are manifold.

    A one-centimetre closing is far below source precision and print tolerance;
    it leaves separated lakes apart while removing zero-width land slivers.
    """
    merged = unary_union(polygons)
    pieces = _flatten_polygons(merged)
    # Include pinches within one polygon (e.g. an island touching its coastline),
    # not just contacts between different polygons. Ring-closing points are not
    # contacts and must not be counted twice.
    rings = [np.asarray(ring.coords)[:-1, :2] for piece in pieces
             for ring in (piece.exterior, *piece.interiors)]
    if not rings:
        return merged
    # Node linework first: a hole may meet the middle of an exterior edge,
    # where the exterior has no explicit vertex yet.
    linework = shapely.node(shapely.MultiLineString([
        np.vstack([ring, ring[0]]) for ring in rings]))
    lines = shapely.get_parts(linework)
    endpoints = np.concatenate([np.asarray(line.coords)[[0, -1], :2] for line in lines])
    coordinates, counts = np.unique(endpoints, axis=0, return_counts=True)
    contacts = coordinates[counts > 2]
    if not len(contacts):
        return merged
    patches = shapely.buffer(shapely.points(contacts), 0.01)
    return unary_union([merged, *patches])


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
