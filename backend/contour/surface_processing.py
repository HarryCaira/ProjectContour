"""Bounded elevation smoothing and continuous transitions to flat water."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon

from contour.heightmap import Heightmap

SMOOTHING_SIGMA_PIXELS = 2.0
SMOOTHING_MAX_MM = 0.05  # At 1x relief; exaggeration scales this displacement too.


def smooth_heightmap(heightmap: Heightmap, mm_per_m: float, sigma: float = SMOOTHING_SIGMA_PIXELS, max_mm: float = SMOOTHING_MAX_MM) -> Heightmap:
    """Remove fine raster noise without moving any sample over 0.05 print mm.

    A smooth saturation avoids the new ridges that hard clipping the correction
    would introduce. Source tiles remain untouched, and extrema cannot overshoot.
    """
    if not np.isfinite(mm_per_m) or mm_per_m <= 0:
        raise ValueError("Model scale must be finite and positive")
    original = heightmap.elevations.astype(np.float64)
    if not np.all(np.isfinite(original)):
        raise ValueError("Elevation samples must be finite")
    candidate = gaussian_filter(original, sigma=sigma, mode="nearest")
    if sigma == 0 or max_mm == 0:
        return heightmap
    limit = max_mm / mm_per_m
    elevations = original + limit * np.tanh((candidate - original) / limit)
    return replace(heightmap, elevations=elevations)


def shoreline_sampler(
    sample: Callable[[np.ndarray], np.ndarray],
    polygons: list[Polygon],
    levels: list[float],
    minimum_width: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Share the same gently blended surface between terrain and route geometry."""
    if len(polygons) != len(levels):
        raise ValueError("Each water polygon needs a surface level")
    if minimum_width <= 0:
        raise ValueError("Shoreline blend width must be positive")
    widths = []
    for polygon, level in zip(polygons, levels):
        boundary = shapely.segmentize(polygon.boundary, max_segment_length=minimum_width / 4)
        delta = np.abs(sample(shapely.get_coordinates(boundary)) - level)
        # Spread larger adjustments over a wider bank instead of excavating a
        # narrow rim. Median lake levels keep both lowering and raising modest.
        widths.append(max(minimum_width, float(delta.max()) * 4))
    tree = shapely.STRtree([polygon.boundary for polygon in polygons])

    def sample_surface(points: np.ndarray) -> np.ndarray:
        elevations = sample(points)
        if not polygons or not len(points):
            return elevations
        point_geometries = shapely.points(points)
        point_ids, polygon_ids = tree.query(point_geometries, predicate="dwithin", distance=max(widths))
        weight_sum = np.zeros(len(points))
        correction_sum = np.zeros(len(points))
        strongest = np.zeros(len(points))
        boundary_level = np.full(len(points), np.nan)
        for index in np.unique(polygon_ids):
            ids = point_ids[polygon_ids == index]
            distance = shapely.distance(point_geometries[ids], polygons[index].boundary)
            t = np.clip(distance / widths[index], 0, 1)
            weight = 1 - t*t*(3 - 2*t)
            influence = weight / np.maximum(t, 1e-9)**2
            weight_sum[ids] += influence
            correction_sum[ids] += influence * (levels[index] - elevations[ids])
            strongest[ids] = np.maximum(strongest[ids], weight)
            boundary_level[ids[distance <= minimum_width * 1e-6]] = levels[index]
        active = weight_sum > 0
        elevations[active] += strongest[active] * correction_sum[active] / weight_sum[active]
        # Exact contact remains authoritative even when nearby banks overlap.
        boundary = np.isfinite(boundary_level)
        elevations[boundary] = boundary_level[boundary]
        return elevations

    return sample_surface
