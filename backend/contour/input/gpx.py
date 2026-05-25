"""GPX parsing and route normalisation."""
from __future__ import annotations

import io

import gpxpy
import numpy as np

from contour.schema.route import Route


def parse_gpx(data: bytes) -> Route:
    """Parse GPX bytes into a normalised Route.

    Handles real-world quirks:
    - Multi-track / multi-segment files: takes the first non-empty segment;
      falls back to the first non-empty <rte> if no tracks have points.
    - Missing elevation: forward-fills and back-fills from neighbours; if every
      point lacks elevation, returns zeros (DEM-based imputation is a later stage).
    - Empty or malformed input: raises ValueError with an actionable message.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"GPX file is not valid UTF-8: {e}") from e

    try:
        gpx = gpxpy.parse(io.StringIO(text))
    except Exception as e:
        raise ValueError(f"GPX file could not be parsed: {e}") from e

    points, name = _find_first_points(gpx)
    if not points:
        raise ValueError("GPX file contains no track or route points.")

    lats = np.array([p.latitude for p in points], dtype=np.float64)
    lons = np.array([p.longitude for p in points], dtype=np.float64)
    raw_eles = np.array(
        [p.elevation if p.elevation is not None else np.nan for p in points],
        dtype=np.float64,
    )
    eles = _fill_elevations(raw_eles)

    return Route(latitudes=lats, longitudes=lons, elevations=eles, name=name)


def _find_first_points(gpx: gpxpy.gpx.GPX) -> tuple[list, str | None]:
    for track in gpx.tracks:
        for segment in track.segments:
            if segment.points:
                return segment.points, track.name
    for route in gpx.routes:
        if route.points:
            return route.points, route.name
    return [], None


def _fill_elevations(eles: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(eles)):
        return np.zeros_like(eles)
    out = eles.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out
