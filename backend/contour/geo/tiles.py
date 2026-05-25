"""Web Mercator tile-space math: lat/lon <-> tile coords at a given zoom."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

TILE_SIZE = 256


@dataclass(frozen=True)
class RasterTile:
    """A discrete Web Mercator tile, identified by (zoom, x, y)."""

    zoom: int
    x: int
    y: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Geographic bbox of this tile: (west_lon, south_lat, east_lon, north_lat)."""
        west_lon, north_lat = _corner_to_lonlat(self.zoom, self.x, self.y)
        east_lon, south_lat = _corner_to_lonlat(self.zoom, self.x + 1, self.y + 1)
        return west_lon, south_lat, east_lon, north_lat


def _corner_to_lonlat(zoom: int, x: float, y: float) -> tuple[float, float]:
    """Convert a tile-space (x, y) at the given zoom to the (lon, lat) of that point's NW corner."""
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> RasterTile:
    """Return the tile containing (lon, lat) at the given zoom, clamped to the valid range."""
    n = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return RasterTile(zoom=zoom, x=x, y=y)


def lonlat_to_pixel(
    lon: ArrayLike, lat: ArrayLike, zoom: int, tile_size: int = TILE_SIZE
) -> NDArray[np.float64]:
    """Convert lat/lon to fractional global pixel coordinates at the given zoom.

    Returns an array stacked on the last axis: scalar input -> (2,); array (...,) -> (..., 2).
    """
    lon_a = np.asarray(lon, dtype=np.float64)
    lat_a = np.asarray(lat, dtype=np.float64)
    n = 2**zoom
    px = (lon_a + 180.0) / 360.0 * n * tile_size
    lat_rad = np.radians(lat_a)
    py = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n * tile_size
    return np.stack([px, py], axis=-1)


def pixel_to_lonlat(
    px: ArrayLike, py: ArrayLike, zoom: int, tile_size: int = TILE_SIZE
) -> NDArray[np.float64]:
    """Inverse of lonlat_to_pixel. Returns (lon, lat) stacked on the last axis."""
    px_a = np.asarray(px, dtype=np.float64)
    py_a = np.asarray(py, dtype=np.float64)
    n = 2**zoom
    lon = px_a / (n * tile_size) * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * py_a / (n * tile_size)))))
    return np.stack([lon, lat], axis=-1)


def tiles_covering_bbox(
    west: float, south: float, east: float, north: float, zoom: int
) -> list[RasterTile]:
    """Return every tile that intersects the given geographic bbox at the given zoom."""
    nw = lonlat_to_tile(west, north, zoom)
    se = lonlat_to_tile(east, south, zoom)
    min_x, max_x = min(nw.x, se.x), max(nw.x, se.x)
    min_y, max_y = min(nw.y, se.y), max(nw.y, se.y)
    return [RasterTile(zoom=zoom, x=x, y=y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
