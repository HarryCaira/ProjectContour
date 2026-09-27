"""Heightmap: a stitched DEM tile mosaic at a fixed Web Mercator zoom."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Heightmap:
    """Stitched elevation grid covering a contiguous block of tiles at one zoom.

    elevations: (H, W) array of metres above sea level.
    tile_origin: (x, y) of the NW-most tile in the mosaic. Pixel (0, 0) of the
                 heightmap is the NW corner of that tile.
    """

    elevations: np.ndarray
    zoom: int
    tile_origin_x: int
    tile_origin_y: int
    tile_size: int = 256

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevations.shape

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Geographic bbox of the mosaic: (west, south, east, north)."""
        h, w = self.shape
        n = 2**self.zoom
        west = self.tile_origin_x / n * 360.0 - 180.0
        east = (self.tile_origin_x + w / self.tile_size) / n * 360.0 - 180.0
        north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * self.tile_origin_y / n))))
        south = math.degrees(
            math.atan(math.sinh(math.pi * (1 - 2 * (self.tile_origin_y + h / self.tile_size) / n)))
        )
        return west, south, east, north
