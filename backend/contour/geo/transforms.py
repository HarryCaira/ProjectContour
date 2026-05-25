"""Coordinate frame transforms: geodetic <-> local ENU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymap3d as pm
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LocalENU:
    """Local East-North-Up frame anchored at a geodetic origin (lat, lon, height).

    Used as the working frame for all geometry once a hex frame has been established;
    routes, heightmap pixel positions, and biome polygons all live in the same ENU.
    """

    lat0: float
    lon0: float
    h0: float = 0.0

    def to_enu(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike = 0.0) -> NDArray[np.float64]:
        """Convert geodetic coordinates to ENU.

        Accepts scalars or arrays. Returns an array stacked on the last axis:
        scalar input -> shape (3,); array input of shape (...,) -> shape (..., 3).
        """
        e, n, u = pm.geodetic2enu(lat, lon, h, self.lat0, self.lon0, self.h0)
        return np.stack([np.asarray(e), np.asarray(n), np.asarray(u)], axis=-1)

    def to_geodetic(self, e: ArrayLike, n: ArrayLike, u: ArrayLike = 0.0) -> NDArray[np.float64]:
        """Inverse of to_enu. Returns (lat, lon, h) stacked on the last axis."""
        lat, lon, h = pm.enu2geodetic(e, n, u, self.lat0, self.lon0, self.h0)
        return np.stack([np.asarray(lat), np.asarray(lon), np.asarray(h)], axis=-1)
