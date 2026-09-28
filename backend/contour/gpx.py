"""GPX parsing and route normalisation."""
from __future__ import annotations

import io

import gpxpy
import numpy as np

from contour.route import Route


def parse_gpx(data: bytes) -> Route:
    """Parse GPX bytes into a normalised Route.

    Handles real-world quirks:
    - Preserves every non-empty track segment, including recording gaps;
      falls back to all non-empty <rte> elements if no tracks have points.
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

    segments = [(segment.points, track.name) for track in gpx.tracks
                for segment in track.segments if segment.points]
    if not segments:
        segments = [(route.points, route.name) for route in gpx.routes if route.points]
    if not segments:
        raise ValueError("GPX file contains no track or route points.")

    starts = []
    latitudes, longitudes, elevations = [], [], []
    for points, _ in segments:
        starts.append(len(latitudes))
        latitudes.extend(p.latitude for p in points)
        longitudes.extend(p.longitude for p in points)
        # Do not fill missing heights across a recording break.
        elevations.extend(_fill_elevations(np.array([
            p.elevation if p.elevation is not None else np.nan for p in points
        ], dtype=np.float64)))
    return Route(latitudes=np.asarray(latitudes, dtype=np.float64),
                 longitudes=np.asarray(longitudes, dtype=np.float64),
                 elevations=np.asarray(elevations, dtype=np.float64),
                 name=next((name for _, name in segments if name), None),
                 segment_starts=tuple(starts))


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
