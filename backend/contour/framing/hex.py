"""Hex framing: derive a pointy-top hexagonal region around a route."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from contour.geo.transforms import LocalENU
from contour.schema.route import Route

# Pointy-top hex: first vertex points to +N (90° in mathematical convention).
_FIRST_VERTEX_ANGLE_RAD = math.pi / 2


@dataclass(frozen=True)
class HexFrame:
    """A pointy-top regular hexagon anchored at a geodetic point.

    All downstream geometry is computed in a LocalENU anchored on the hex centre.
    rotation_degrees ∈ [0, 60) rotates the hex around its centre; rotation beyond 60°
    is redundant because of hex symmetry.
    """

    centre_lon: float
    centre_lat: float
    circumradius_m: float
    rotation_degrees: float = 0.0

    @property
    def apothem_m(self) -> float:
        """Perpendicular distance from centre to each edge midpoint."""
        return self.circumradius_m * math.sqrt(3) / 2

    def local_enu(self) -> LocalENU:
        """The LocalENU frame anchored on this hex's centre."""
        return LocalENU(lat0=self.centre_lat, lon0=self.centre_lon, h0=0.0)

    def vertices_enu(self) -> NDArray[np.float64]:
        """6 vertices in ENU (relative to the hex centre), CCW starting from the top vertex."""
        rot_rad = math.radians(self.rotation_degrees)
        angles = _FIRST_VERTEX_ANGLE_RAD + rot_rad + np.arange(6) * (math.pi / 3)
        return np.stack([self.circumradius_m * np.cos(angles), self.circumradius_m * np.sin(angles)], axis=-1)

    def polygon_enu(self) -> Polygon:
        """The hex as a Shapely polygon in ENU coordinates (relative to the hex centre)."""
        return Polygon(self.vertices_enu().tolist())


def hex_frame_for_route(
    route: Route,
    padding_ratio: float = 0.15,
    rotation_degrees: float = 0.0,
) -> HexFrame:
    """Compute the smallest hex containing the route, with the given proportional padding.

    The hex is centred on the route's geographic bbox centroid. Its size is set so the
    inscribed circle (radius = apothem) contains every route point plus the requested
    padding — guaranteeing the route fits regardless of hex rotation.
    """
    centre_lon, centre_lat = route.centroid
    enu = LocalENU(lat0=centre_lat, lon0=centre_lon)

    pts_enu = enu.to_enu(route.latitudes, route.longitudes, 0.0)
    radii = np.hypot(pts_enu[..., 0], pts_enu[..., 1])
    max_radius_m = float(radii.max()) if route.num_points > 0 else 0.0

    apothem_needed = max_radius_m * (1.0 + padding_ratio)
    circumradius_m = apothem_needed * 2.0 / math.sqrt(3)

    return HexFrame(
        centre_lon=centre_lon,
        centre_lat=centre_lat,
        circumradius_m=circumradius_m,
        rotation_degrees=rotation_degrees,
    )
