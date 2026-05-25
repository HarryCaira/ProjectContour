"""Bilinear sampling of a Heightmap at ENU points."""
from __future__ import annotations

import numpy as np

from contour.geo.tiles import lonlat_to_pixel
from contour.geo.transforms import LocalENU
from contour.schema.heightmap import Heightmap


def sample_at_enu(
    heightmap: Heightmap,
    enu_points: np.ndarray,
    local_enu: LocalENU,
) -> np.ndarray:
    """Bilinear interpolation of heightmap elevations at ENU (E, N) points.

    Returns a (N,) array of elevations in metres. Out-of-range points are clamped
    to the nearest valid pixel.
    """
    if enu_points.ndim != 2 or enu_points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) ENU points, got shape {enu_points.shape}")
    if len(enu_points) == 0:
        return np.empty((0,), dtype=np.float32)

    geo = local_enu.to_geodetic(enu_points[:, 0], enu_points[:, 1], 0.0)
    lats = geo[..., 0]
    lons = geo[..., 1]

    pixel_global = lonlat_to_pixel(lons, lats, heightmap.zoom, heightmap.tile_size)
    px = pixel_global[..., 0] - heightmap.tile_origin_x * heightmap.tile_size
    py = pixel_global[..., 1] - heightmap.tile_origin_y * heightmap.tile_size

    return _bilinear(heightmap.elevations, px, py)


def _bilinear(grid: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    px = np.clip(px, 0.0, w - 1.001)
    py = np.clip(py, 0.0, h - 1.001)

    x0 = np.floor(px).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(py).astype(np.int64)
    y1 = y0 + 1

    fx = px - x0
    fy = py - y0

    z00 = grid[y0, x0]
    z10 = grid[y0, x1]
    z01 = grid[y1, x0]
    z11 = grid[y1, x1]

    return (
        z00 * (1 - fx) * (1 - fy)
        + z10 * fx * (1 - fy)
        + z01 * (1 - fx) * fy
        + z11 * fx * fy
    )
