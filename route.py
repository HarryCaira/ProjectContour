from __future__ import annotations
from dataclasses import dataclass
import math
from gpx import GPX
import numpy as np
from typing import Tuple

from parameters import GlobalParameters, ModelResolution
from coordinate_transform import LonLatToENU, RasterTile, lonlat_to_tile


@dataclass(frozen=True)
class ZoomLevel:
    value: int

    @classmethod
    def _compute_auto_zoom(cls, bbox: LatLonBBox, params: GlobalParameters, model_res: ModelResolution, max_tiles: int = 1000) -> int:
        optimum_zoom: int | None = None
        for zoom in range(1, 16):
            meters_per_pixel = (params.EARTH_CIRCUMFERENCE_M * math.cos(math.radians(bbox.central_latitude))) / (256 * (2**zoom))
            tiles = bbox.tiles_to_cover(zoom)
            num_tiles = len(tiles)

            if num_tiles > max_tiles:
                break
            optimum_zoom = zoom
            if meters_per_pixel <= model_res.meters:
                return zoom

        if optimum_zoom is None:
            raise ValueError(f"Bounding box too large: even zoom 1 exceeds {max_tiles}-tile budget.")
        return optimum_zoom

    @classmethod
    def new(cls, params: GlobalParameters, bbox: LatLonBBox, manual_zoom: int | None, model_res: ModelResolution) -> ZoomLevel:
        if manual_zoom is not None:
            return cls(value=manual_zoom)

        auto_zoom = cls._compute_auto_zoom(bbox, params, model_res, max_tiles=1000)
        return cls(value=auto_zoom)


@dataclass(frozen=True)
class LatLonBBox:
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

    def tiles_to_cover(self, zoom: int) -> list[RasterTile]:
        """
        Get all tile coordinates (z, x, y) needed to cover this bounding box at zoom level z.
        """
        min_tile = lonlat_to_tile(self.min_longitude, self.min_latitude, zoom)
        max_tile = lonlat_to_tile(self.max_longitude, self.max_latitude, zoom)

        # In Web Mercator, y increases southward, so min_lat → max_y and max_lat → min_y
        min_x, max_x = min(min_tile.x, max_tile.x), max(min_tile.x, max_tile.x)
        min_y, max_y = min(min_tile.y, max_tile.y), max(min_tile.y, max_tile.y)

        tiles = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append(RasterTile(zoom, x, y))
        return tiles


@dataclass(frozen=True)
class LatLonRoute:
    """
    Immutable GPX-derived route represented in geographic coordinates (lat/lon).
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
    def bbox(self) -> LatLonBBox:
        return LatLonBBox(
            min_latitude=float(self.latitudes.min()),
            min_longitude=float(self.longitudes.min()),
            max_latitude=float(self.latitudes.max()),
            max_longitude=float(self.longitudes.max()),
        )

    @classmethod
    def new(cls, gpx_file_path: str) -> LatLonRoute:
        gpx = GPX.from_file(gpx_file_path)

        if not gpx.tracks or not gpx.tracks[0].segments:
            raise ValueError("GPX file has no track/segment data.")

        if len(gpx.tracks) > 1:
            raise ValueError(f"GPX file has {len(gpx.tracks)} tracks; only single-track files are supported.")
        if len(gpx.tracks[0].segments) > 1:
            raise ValueError(f"GPX track has {len(gpx.tracks[0].segments)} segments; only single-segment tracks are supported.")

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
class EnuRoute:
    """
    Immutable route represented in ENU coordinates.
    """

    e: np.ndarray
    n: np.ndarray
    u: np.ndarray

    @classmethod
    def new(cls, route: LatLonRoute, transform: LonLatToENU) -> EnuRoute:
        enu_coords = transform.lonlat_to_enu(route.latitudes, route.longitudes, route.elevations)
        return cls(e=enu_coords[:, 0], n=enu_coords[:, 1], u=enu_coords[:, 2])
