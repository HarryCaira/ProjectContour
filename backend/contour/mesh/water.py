"""Water mesh: extrude water polygons into a recessed closed solid."""
from __future__ import annotations

import trimesh
from shapely.geometry import Polygon


def build_water_mesh(
    water_polygons: list[Polygon],
    top_z: float,
    bottom_z: float,
) -> trimesh.Trimesh | None:
    """Build a closed water solid by extruding each polygon between two Z levels.

    Returns None if there are no polygons or the height is non-positive.
    """
    if not water_polygons:
        return None
    height = top_z - bottom_z
    if height <= 0:
        return None

    pieces: list[trimesh.Trimesh] = []
    for poly in water_polygons:
        if poly.is_empty or not poly.is_valid:
            continue
        mesh = trimesh.creation.extrude_polygon(poly, height=height)
        mesh.apply_translation([0.0, 0.0, bottom_z])
        pieces.append(mesh)

    if not pieces:
        return None
    return trimesh.util.concatenate(pieces)
