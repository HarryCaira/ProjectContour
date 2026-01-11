from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pymap3d as pm


@dataclass(frozen=True)
class LonLatToENU:
    """
    Immutable converter from geodetic coordinates to ENU.
    Elevation is intentionally ignored (DEM provides height).
    """

    lat0: float
    lon0: float
    h0: float

    def lonlat_to_enu(self, lat: ArrayLike, lon: ArrayLike, ele: ArrayLike) -> np.ndarray:
        """
        Convert lat/lon/ele to ENU coordinates.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        ele = np.asarray(ele, dtype=float)

        E, N, U = pm.geodetic2enu(lat, lon, ele, self.lat0, self.lon0, self.h0)
        return np.vstack([E, N, U]).T

    def enu_to_lonlat(self, e: ArrayLike, n: ArrayLike, u: ArrayLike) -> np.ndarray:
        """
        Convert ENU coordinates back to lat/lon/ele.
        Returns an (N, 3) array.
        """
        e = np.asarray(e, dtype=float)
        n = np.asarray(n, dtype=float)
        u = np.asarray(u, dtype=float)

        lat, lon, ele = pm.enu2geodetic(e, n, u, self.lat0, self.lon0, self.h0)
        return np.vstack([lat, lon, ele]).T

    @classmethod
    def new(cls, origin: tuple[float, float, float]) -> LonLatToENU:
        lat0, lon0, h0 = origin
        return cls(lat0=lat0, lon0=lon0, h0=h0)

@dataclass(frozen=True)
class RasterTile:
    zoom: int
    x: int
    y: int

@dataclass(frozen=True)
class LonLatToRasterTile:
    def lonlat_to_tile(self, lon: float, lat: float, zoom: int) -> RasterTile:
        """
        Convert a single lon/lat coordinate to tile coordinates at the given zoom level.
        Uses Web Mercator projection (EPSG:3857).
        
        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            zoom: Zoom level
            
        Returns:
            RasterTile with x, y tile indices
        """
        n = 2**zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        
        lat_rad = np.radians(lat)
        y_tile = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)

        return RasterTile(zoom=zoom, x=x_tile, y=y_tile)