"""Normalised GPS route: the input data structure flowing into the pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Route:
    """A normalised GPS route in geodetic coordinates.

    Equality is intentionally not generated — routes are large and comparison isn't useful.
    """

    latitudes: np.ndarray
    longitudes: np.ndarray
    elevations: np.ndarray
    name: str | None = None
    segment_starts: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not (self.latitudes.shape == self.longitudes.shape == self.elevations.shape):
            raise ValueError("Route arrays must have matching shapes")
        if self.latitudes.ndim != 1:
            raise ValueError("Route arrays must be 1-D")

        if (not self.segment_starts or self.segment_starts[0] != 0
                or any(a >= b for a, b in zip(self.segment_starts, self.segment_starts[1:]))
                or any(i < 0 or i >= max(1, self.num_points) for i in self.segment_starts)):
            raise ValueError("Segment starts must begin at zero and increase within the route")

    def segments(self) -> list[Route]:
        """Independent recorded pieces, in file order."""
        return [Route(self.latitudes[a:b], self.longitudes[a:b], self.elevations[a:b], self.name)
                for a, b in zip(self.segment_starts, (*self.segment_starts[1:], self.num_points))]

    @property
    def num_points(self) -> int:
        return int(self.latitudes.shape[0])

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west_lon, south_lat, east_lon, north_lat)."""
        return (
            float(self.longitudes.min()),
            float(self.latitudes.min()),
            float(self.longitudes.max()),
            float(self.latitudes.max()),
        )

    @property
    def centroid(self) -> tuple[float, float]:
        """(lon, lat) of the bbox centroid."""
        west, south, east, north = self.bbox
        return ((west + east) / 2, (south + north) / 2)

    def distance_km(self) -> float:
        """Approximate planar distance in km using the haversine formula (elevation ignored)."""
        if self.num_points < 2:
            return 0.0
        R = 6371.0088
        lat = np.radians(self.latitudes)
        lon = np.radians(self.longitudes)
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        c[np.asarray(self.segment_starts[1:], dtype=int) - 1] = 0
        return float(np.sum(R * c))
