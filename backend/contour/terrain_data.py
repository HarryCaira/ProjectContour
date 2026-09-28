"""Terrain data acquisition: Mapbox Terrain-RGB tiles into a stitched heightmap."""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from contour.errors import MeshDetailLimitError
from contour.hex_frame import HexFrame
from contour.tiles import RasterTile, tiles_covering_bbox
from contour.tile_cache import TileCache
from contour.http_client import HttpClient
from contour.heightmap import Heightmap
from contour.settings import Physical
from contour.production import TERRAIN_SAMPLE_MM, MAX_TERRAIN_ZOOM

EARTH_CIRCUMFERENCE_M = 40_075_017.0
PROVIDER = "mapbox"
LAYER = "terrain-rgb"
BASE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb"


def fetch_heightmap(
    hex_frame: HexFrame,
    physical: Physical,
    client: HttpClient,
    cache: TileCache,
    mapbox_token: str,
    max_concurrent: int = 8,
    maximum_source_detail: bool = False,
    sample_mm: float = TERRAIN_SAMPLE_MM, max_zoom: int = MAX_TERRAIN_ZOOM, max_tiles: int = 1024,
) -> Heightmap:
    """Pick a zoom level, fetch all covering Terrain-RGB tiles, stitch into a Heightmap."""
    zoom = select_zoom(hex_frame, physical, maximum_source_detail=maximum_source_detail, sample_mm=sample_mm, max_zoom=max_zoom, max_tiles=max_tiles)
    west, south, east, north = _hex_geographic_bbox(hex_frame)
    tiles = tiles_covering_bbox(west, south, east, north, zoom)

    def fetch_one(tile: RasterTile) -> tuple[RasterTile, np.ndarray]:
        png_bytes = _fetch_tile_with_cache(client, cache, mapbox_token, tile)
        return tile, decode_terrain_rgb(png_bytes)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = list(pool.map(fetch_one, tiles))

    tile_arrays = {(t.x, t.y): arr for t, arr in results}
    return stitch_heightmap(tile_arrays, zoom)


def select_zoom(hex_frame: HexFrame, physical: Physical, max_tiles: int = 1024, maximum_source_detail: bool = False, sample_mm: float = TERRAIN_SAMPLE_MM, max_zoom: int = MAX_TERRAIN_ZOOM) -> int:
    """Pick the lowest zoom meeting our fixed production sampling target.
    Stop at native terrain detail or the tile budget, whichever comes first.
    The legacy resolutionMm field is accepted but no longer controls quality."""
    centre_lat_rad = math.radians(hex_frame.centre_lat)
    minx, miny, maxx, maxy = hex_frame.polygon_enu().bounds
    model_world_diameter_m = max(maxx - minx, maxy - miny)
    model_pixel_count = physical.size_mm / sample_mm
    target_meters_per_pixel = model_world_diameter_m / model_pixel_count

    last_within_budget: int | None = None
    for zoom in range(1, max_zoom + 1):
        meters_per_pixel = EARTH_CIRCUMFERENCE_M * math.cos(centre_lat_rad) / (256 * (2**zoom))
        tile_size_m = 256 * meters_per_pixel
        n_tiles_axis = math.ceil(model_world_diameter_m / tile_size_m) + 1
        n_tiles = n_tiles_axis**2

        if n_tiles > max_tiles:
            break
        last_within_budget = zoom
        if not maximum_source_detail and meters_per_pixel <= target_meters_per_pixel:
            return zoom

    if maximum_source_detail and last_within_budget != max_zoom:
        native_pixel_size = EARTH_CIRCUMFERENCE_M * math.cos(centre_lat_rad) / (256 * 2**max_zoom)
        estimate = (math.ceil(model_world_diameter_m / (256 * native_pixel_size)) + 1)**2
        raise MeshDetailLimitError("tiles", estimate, max_tiles, details={"zoom": max_zoom})

    if last_within_budget is None:
        raise MeshDetailLimitError("tiles", n_tiles, max_tiles, details={"zoom": zoom})
    return last_within_budget


def decode_terrain_rgb(png_bytes: bytes) -> np.ndarray:
    """Decode a Terrain-RGB PNG into an elevation array (metres above sea level)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return -10000.0 + (r * 65536 + g * 256 + b) * 0.1


def stitch_heightmap(tiles: dict[tuple[int, int], np.ndarray], zoom: int) -> Heightmap:
    """Stitch a dict of {(x, y): tile_array} into a single contiguous Heightmap."""
    if not tiles:
        raise ValueError("No tiles to stitch")

    xs = [x for x, _ in tiles]
    ys = [y for _, y in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    sample = next(iter(tiles.values()))
    tile_h, tile_w = sample.shape
    if not all(arr.shape == (tile_h, tile_w) for arr in tiles.values()):
        raise ValueError("All tiles must have identical shape")

    h = (max_y - min_y + 1) * tile_h
    w = (max_x - min_x + 1) * tile_w
    elevations = np.zeros((h, w), dtype=np.float32)
    for (x, y), tile in tiles.items():
        row = y - min_y
        col = x - min_x
        elevations[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile

    return Heightmap(
        elevations=repair_tile_seams(elevations, tile_h, tile_w),
        zoom=zoom,
        tile_origin_x=min_x,
        tile_origin_y=min_y,
        tile_size=tile_h,
    )


def repair_tile_seams(elevations: np.ndarray, tile_height: int, tile_width: int) -> np.ndarray:
    """Repair long, one-pixel zero stripes exactly on internal tile edges.

    Zero is a valid elevation, so require 32 consecutive suspect samples with
    comparable, same-sign terrain on both sides, each at least one metre from
    zero. Interpolate only those samples; leave coastlines and ambiguous data
    untouched. Source tiles and the caller's array remain unchanged.
    """
    repaired = elevations.copy()
    for source, target, tile_size in ((elevations, repaired, tile_width),
                                      (elevations.T, repaired.T, tile_height)):
        for boundary in range(tile_size, source.shape[1], tile_size):
            for column in (boundary - 1, boundary):
                if column < 1 or column + 1 >= source.shape[1]:
                    continue
                left, centre, right = source[:, column - 1], source[:, column], source[:, column + 1]
                minimum = np.minimum(np.abs(left), np.abs(right))
                suspect = ((centre == 0) & np.isfinite(left) & np.isfinite(right)
                           & (left * right > 0) & (minimum >= 1)
                           & (np.abs(left - right) <= np.maximum(1, minimum * .5)))
                transitions = np.diff(np.r_[False, suspect, False].astype(np.int8))
                for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                    if stop - start >= 32:
                        target[start:stop, column] = (left[start:stop] + right[start:stop]) / 2
    return repaired


def _fetch_tile_with_cache(
    client: HttpClient, cache: TileCache, token: str, tile: RasterTile
) -> bytes:
    cached = cache.get(PROVIDER, LAYER, tile.zoom, tile.x, tile.y, "png")
    if cached is not None:
        return cached
    url = f"{BASE_URL}/{tile.zoom}/{tile.x}/{tile.y}.pngraw"
    data = client.get(url, params={"access_token": token})
    cache.set(PROVIDER, LAYER, tile.zoom, tile.x, tile.y, "png", data)
    return data


def _hex_geographic_bbox(hex_frame: HexFrame) -> tuple[float, float, float, float]:
    """Geographic bbox containing the entire hex: (west, south, east, north)."""
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
