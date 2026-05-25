"""Land mesh: heightmap + (hex \\ water) polygon -> watertight 3D solid."""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from contour.framing.hex import HexFrame
from contour.mesh.hex_clip import triangulate_land
from contour.mesh.sampling import sample_at_enu
from contour.schema.heightmap import Heightmap


def build_land_mesh(
    hex_frame: HexFrame,
    heightmap: Heightmap,
    water_polygons: list[Polygon],
    base_z: float,
    grid_points_per_side: int = 100,
) -> trimesh.Trimesh:
    """Build a watertight land solid from the heightmap, clipped to the hex with
    water polygons as holes.

    The mesh consists of three logical surfaces:
    - Top: triangulated land polygon, vertices at sampled heightmap elevations.
    - Bottom: same triangulation, flat at `base_z`, reverse-wound.
    - Walls: vertical strips along every boundary segment (outer hex + water holes),
             from heightmap elevation down to `base_z`.
    """
    hex_poly = hex_frame.polygon_enu()
    local_enu = hex_frame.local_enu()

    tri = triangulate_land(hex_poly, water_polygons, grid_points_per_side=grid_points_per_side)
    vertices_2d = tri.vertices

    elevations = sample_at_enu(heightmap, vertices_2d, local_enu)

    top_vertices = np.column_stack([vertices_2d, elevations]).astype(np.float64)
    bottom_vertices = np.column_stack([vertices_2d, np.full(len(vertices_2d), base_z)]).astype(np.float64)
    n_top = len(top_vertices)

    all_vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)

    top_faces = tri.triangles.astype(np.int64)
    bottom_faces = (tri.triangles[:, ::-1] + n_top).astype(np.int64)

    # Walls: for each boundary segment (i -> j) with the polygon interior on the
    # left, build two outward-facing triangles connecting the top and bottom rims.
    walls: list[list[int]] = []
    for i, j in tri.boundary_segments:
        top_i, top_j = i, j
        bot_i, bot_j = i + n_top, j + n_top
        walls.append([top_i, top_j, bot_j])
        walls.append([top_i, bot_j, bot_i])

    all_faces = np.concatenate(
        [top_faces, bottom_faces, np.array(walls, dtype=np.int64)], axis=0
    )

    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh
