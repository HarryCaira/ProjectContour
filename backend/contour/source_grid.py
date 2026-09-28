"""Native DEM sample locations, projected into the model without resampling."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely

from contour import terrain_refinement
from contour.errors import MeshDetailLimitError
from contour.heightmap import Heightmap
from contour.hex_frame import HexFrame
from contour.tiles import pixel_to_lonlat


def source_grid_points(heightmap: Heightmap, frame: HexFrame, domain: shapely.Geometry,
                       checkpoint: Callable[[], None] | None = None, max_vertices: int | None = None) -> np.ndarray:
    max_vertices = terrain_refinement.MAX_VERTICES if max_vertices is None else max_vertices
    check = checkpoint or (lambda: None)
    rows, columns = heightmap.shape
    local = frame.local_enu()
    chunks = []
    count = 0
    for row in range(0, rows, 32):
        check()
        x, y = np.meshgrid(np.arange(columns) + heightmap.tile_origin_x * heightmap.tile_size,
                           np.arange(row, min(row + 32, rows)) + heightmap.tile_origin_y * heightmap.tile_size)
        geo = pixel_to_lonlat(x.ravel(), y.ravel(), heightmap.zoom, heightmap.tile_size)
        points = local.to_enu(geo[:, 1], geo[:, 0])[:, :2]
        points = points[shapely.contains_xy(domain, points[:, 0], points[:, 1])]
        count += len(points)
        if count > max_vertices:
            raise MeshDetailLimitError('vertices', count, max_vertices,
                                       details={'stage': 'native_source_grid'})
        chunks.append(points)
    return np.concatenate(chunks) if chunks else np.empty((0, 2))
