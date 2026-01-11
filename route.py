from __future__ import annotations
from dataclasses import dataclass
import math
from gpx import GPX
import numpy as np
from typing import Tuple
import sys

from parameters import GlobalParameters, ModelResolution
from coordinate_transform import LonLatToENU, LonLatToRasterTile, RasterTile


@dataclass(frozen=True)
class ZoomLevel:
    value: int

    @classmethod
    def _compute_auto_zoom(cls, bbox: BBox_LL, params: GlobalParameters, model_res: ModelResolution, max_tiles: int = 1000) -> int:
        optimum_zoom = 15
        for zoom in range(1, 16):
            meters_per_pixel = (params.EARTH_CIRCUMFERENCE_M * math.cos(math.radians(bbox.central_latitude))) / (256 * (2**zoom))
            tiles = bbox.tiles_to_cover(zoom, transform=LonLatToRasterTile())
            num_tiles = len(tiles)

            if num_tiles > max_tiles:
                break
            if meters_per_pixel <= model_res.meters and num_tiles <= max_tiles:
                optimum_zoom = zoom
        return optimum_zoom

    @classmethod
    def new(cls, params: GlobalParameters, bbox: BBox_LL, manual_zoom: int | None, model_res: ModelResolution) -> ZoomLevel:
        if manual_zoom is not None:
            return cls(value=manual_zoom)

        auto_zoom = cls._compute_auto_zoom(bbox, params, model_res, max_tiles=1000)
        return cls(value=auto_zoom)


@dataclass(frozen=True)
class Grid:
    lat_grid: np.ndarray
    lon_grid: np.ndarray

    @classmethod
    def new(cls, enu_coordinates: np.ndarray, transform: LonLatToENU) -> Grid:
        num_n, num_e = enu_coordinates.shape

        e_flat = enu_coordinates[:, :, 0].ravel()
        n_flat = enu_coordinates[:, :, 1].ravel()
        u_flat = np.zeros_like(e_flat)

        lonlat_coords = transform.enu_to_lonlat(e_flat, n_flat, u_flat)

        lat_grid = lonlat_coords[:, 0].reshape((num_n, num_e))
        lon_grid = lonlat_coords[:, 1].reshape((num_n, num_e))

        return cls(lat_grid=lat_grid, lon_grid=lon_grid)


@dataclass(frozen=True)
class BBox_LL:
    min_longitude: float
    min_latitude: float
    max_longitude: float
    max_latitude: float

    @property
    def longitude_span(self) -> float:
        return self.max_longitude - self.min_longitude

    @property
    def latitude_span(self) -> float:
        return self.max_latitude - self.min_latitude

    @property
    def central_latitude(self) -> float:
        return (self.min_latitude + self.max_latitude) / 2

    def tiles_to_cover(self, zoom: int, transform: LonLatToRasterTile) -> list[RasterTile]:
        """
        Get all tile coordinates (z, x, y) needed to cover this bounding box at zoom level z.

        Args:
            zoom: Zoom level (higher = more detail, max 15)

        Returns:
            List of Tiles
        """
        min_tile = transform.lonlat_to_tile(self.min_longitude, self.min_latitude, zoom)
        max_tile = transform.lonlat_to_tile(self.max_longitude, self.max_latitude, zoom)

        # In Web Mercator, y increases southward, so min_lat → max_y and max_lat → min_y
        min_x, max_x = min(min_tile.x, max_tile.x), max(min_tile.x, max_tile.x)
        min_y, max_y = min(min_tile.y, max_tile.y), max(min_tile.y, max_tile.y)

        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append(RasterTile(zoom, x, y))
        return tiles

    @classmethod
    def new(cls, bounds: Bounds) -> BBox_LL:
        return cls(
            min_longitude=bounds.min_longitude,
            max_longitude=bounds.max_longitude,
            min_latitude=bounds.min_latitude,
            max_latitude=bounds.max_latitude,
        )


@dataclass(frozen=True)
class Bounds:
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


@dataclass(frozen=True)
class Coordinate:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Route_LL:
    """
    Immutable GPX-derived route represented in geographic coordinates (lat/lon).

    Responsibilities:
    - Parse GPX files
    - Store route data
    """

    latitudes: np.ndarray
    longitudes: np.ndarray
    elevations: np.ndarray

    min_longitude: float
    max_longitude: float
    min_latitude: float
    max_latitude: float

    @property
    def origin(self) -> Tuple[float, float, float]:
        return float(self.latitudes[0]), float(self.longitudes[0]), float(self.elevations[0])

    @property
    def centroid(self) -> Coordinate:
        center_lat = (self.min_latitude + self.max_latitude) / 2
        center_lon = (self.min_longitude + self.max_longitude) / 2
        return Coordinate(latitude=center_lat, longitude=center_lon)

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            min_latitude=float(self.latitudes.min()),
            min_longitude=float(self.longitudes.min()),
            max_latitude=float(self.latitudes.max()),
            max_longitude=float(self.longitudes.max()),
        )

    @classmethod
    def new(cls, gpx_file_path: str) -> Route_LL:
        gpx = GPX.from_file(gpx_file_path)
        print(gpx.tracks[0].min_elevation, gpx.tracks[0].max_elevation)

        if not gpx.tracks or not gpx.tracks[0].segments:
            raise ValueError("GPX file has no track/segment data.")

        segment = gpx.tracks[0].segments[0]

        lats = np.array([p.lat for p in segment.points], dtype=float)
        lons = np.array([p.lon for p in segment.points], dtype=float)
        ele = np.array([p.ele for p in segment.points], dtype=float)
        return cls(
            latitudes=lats,
            longitudes=lons,
            elevations=ele,
            min_longitude=float(lons.min()),
            max_longitude=float(lons.max()),
            min_latitude=float(lats.min()),
            max_latitude=float(lats.max()),
        )


@dataclass(frozen=True)
class Route_ENU:
    """
    Immutable route represented in ENU coordinates.
    Responsibilities:
    - Store E/N/U arrays
    """

    e: np.ndarray
    n: np.ndarray
    u: np.ndarray

    @classmethod
    def new(cls, route: Route_LL, transform: LonLatToENU) -> Route_ENU:
        enu_coords = transform.lonlat_to_enu(route.latitudes, route.longitudes, route.elevations)
        return cls(e=enu_coords[:, 0], n=enu_coords[:, 1], u=enu_coords[:, 2])
