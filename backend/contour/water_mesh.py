"""Water mesh: extrude water polygons into a recessed closed solid."""
from __future__ import annotations

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

from contour.hex_frame import HexFrame
from contour.heightmap import Heightmap
from contour.sampling import sample_at_enu


def build_water_mesh(
    water_polygons: list[Polygon],
    top_z: float | list[float],
    bottom_z: float | list[float],
) -> trimesh.Trimesh | None:
    """Build a closed water solid by extruding each polygon between two Z levels.

    Returns None if there are no polygons or the height is non-positive.
    """
    if not water_polygons:
        return None
    levels = top_z if isinstance(top_z, list) else [top_z] * len(water_polygons)
    if len(levels) != len(water_polygons):
        raise ValueError("Each water polygon needs a surface level")

    bottoms = bottom_z if isinstance(bottom_z, list) else [bottom_z] * len(water_polygons)
    if len(bottoms) != len(water_polygons):
        raise ValueError("Each water polygon needs a bottom level")

    pieces: list[trimesh.Trimesh] = []
    for poly, level, bottom in zip(water_polygons, levels, bottoms):
        height = level - bottom
        if poly.is_empty or not poly.is_valid or height <= 0:
            continue
        tolerance = max(np.ptp(np.asarray(poly.exterior.coords), axis=0)) * 1e-10
        poly = shapely.remove_repeated_points(poly, tolerance=tolerance)
        mesh = trimesh.creation.extrude_polygon(poly, height=height)
        mesh.apply_translation([0.0, 0.0, bottom])
        pieces.append(mesh)

    if not pieces:
        return None
    return trimesh.util.concatenate(pieces)


def shoreline_water_levels(
    water_polygons: list[Polygon],
    heightmap: Heightmap,
    frame: HexFrame,
    bottom_z: float,
    recess_m: float,
) -> list[float]:
    """Give each connected water polygon a flat surface below its own shoreline.

    Use the lowest sampled bank to avoid water protruding through lower land.
    Ignore artificial shorelines created where the frame cuts through water.
    Clamp the recess to preserve a positive solid above the shared model base.
    """
    if recess_m < 0 or not np.isfinite(recess_m):
        raise ValueError("Water recess must be finite and non-negative")
    hex_polygon = frame.polygon_enu()
    spacing = 2 * frame.circumradius_m / 100
    local_enu = frame.local_enu()
    levels: list[float] = []
    for polygon in water_polygons:
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Water polygon must be valid and non-empty")
        boundary = shapely.segmentize(polygon.boundary, max_segment_length=spacing)
        coordinates = shapely.get_coordinates(boundary)
        # Cut edges are not physical banks. Fall back to the complete boundary
        # for a frame entirely covered by water.
        distances = shapely.distance(shapely.points(coordinates), hex_polygon.boundary)
        shoreline = coordinates[distances > spacing * 1e-6]
        if len(shoreline) == 0:
            shoreline = coordinates
        elevations = sample_at_enu(heightmap, shoreline, local_enu)
        if not np.all(np.isfinite(elevations)):
            raise ValueError("Water shoreline contains invalid elevations")
        bank_z = float(elevations.min())
        if bank_z <= bottom_z:
            raise ValueError("Water shoreline must be above the model base")
        top_z = bank_z - min(recess_m, (bank_z - bottom_z) * 0.5)
        levels.append(top_z)
    return levels


def build_shoreline_water_mesh(
    water_polygons: list[Polygon],
    heightmap: Heightmap,
    frame: HexFrame,
    bottom_z: float,
    recess_m: float,
) -> trimesh.Trimesh | None:
    polygons = [p for p in water_polygons if not p.is_empty and p.is_valid]
    levels = shoreline_water_levels(polygons, heightmap, frame, bottom_z, recess_m)
    return build_water_mesh(polygons, top_z=levels, bottom_z=bottom_z)
