"""Land mesh: heightmap + (hex \\ water) polygon -> watertight 3D solid."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

from contour.hex_frame import HexFrame
from contour.hex_clip import triangulate_land, land_polygon
from contour.terrain_refinement import refine_terrain
from contour import terrain_refinement
from contour.errors import MeshDetailLimitError
from contour.source_grid import source_grid_points
from contour.sampling import sample_at_enu
from contour.heightmap import Heightmap


def build_land_mesh(
    hex_frame: HexFrame,
    heightmap: Heightmap,
    water_polygons: list[Polygon],
    base_z: float,
    grid_points_per_side: int = 100,
    water_levels: list[float] | None = None,
    surface_tolerance_m: float | None = None,
    sample_spacing_m: float | None = None,
    checkpoint: Callable[[], None] | None = None,
    surface_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
    native_source_grid: bool = False,
    max_vertices: int | None = None, max_reference_points: int | None = None, max_passes: int | None = None,
) -> trimesh.Trimesh:
    """Build a watertight land solid from the heightmap, clipped to the hex with
    water polygons as holes.

    The mesh consists of three logical surfaces:
    - Top: triangulated land polygon, vertices at sampled heightmap elevations.
    - Bottom: same triangulation, flat at `base_z`, reverse-wound.
    - Walls: vertical strips along every boundary segment (outer hex + water holes),
             from heightmap elevation down to `base_z`.
    """
    max_vertices = terrain_refinement.MAX_VERTICES if max_vertices is None else max_vertices
    hex_poly = hex_frame.polygon_enu()
    if water_polygons and hex_poly.difference(shapely.union_all(water_polygons)).is_empty:
        return trimesh.Trimesh()
    local_enu = hex_frame.local_enu()

    native_points = None
    if native_source_grid:
        native_points = source_grid_points(heightmap, hex_frame, land_polygon(hex_poly, water_polygons), checkpoint, max_vertices=max_vertices)
    tri = triangulate_land(hex_poly, water_polygons, grid_points_per_side=grid_points_per_side,
                           interior_points=native_points)
    if native_source_grid and len(tri.vertices) > max_vertices:
        raise MeshDetailLimitError("vertices", len(tri.vertices), max_vertices,
                                   details={"stage": "native_source_grid_with_boundary"})
    vertices_2d = tri.vertices

    if water_levels is not None and len(water_levels) != len(water_polygons):
        raise ValueError("Each water polygon needs a surface level")

    shorelines = shapely.STRtree([polygon.boundary for polygon in water_polygons])

    def sample_surface(points_2d: np.ndarray) -> np.ndarray:
        if surface_sampler is not None:
            return surface_sampler(points_2d)
        elevations = sample_at_enu(heightmap, points_2d, local_enu)
        if water_levels is not None:
            points = shapely.points(points_2d)
            tolerance = max(hex_frame.circumradius_m * 1e-8, 1e-8)
            blend_width = 4 * sample_spacing_m if sample_spacing_m is not None else tolerance
            point_ids, shoreline_ids = shorelines.query(points, predicate="dwithin", distance=blend_width)
            # Most terrain samples are nowhere near a shoreline. Only measure
            # the nearby candidates, rather than every sample against every lake.
            for index in np.unique(shoreline_ids):
                ids = point_ids[shoreline_ids == index]
                distance = shapely.distance(points[ids], water_polygons[index].boundary)
                level = water_levels[index]
                if sample_spacing_m is not None:
                    blend = np.clip(distance / blend_width, 0, 1)
                    elevations[ids] = level * (1 - blend) + elevations[ids] * blend
                elevations[ids[distance <= tolerance]] = level
        return elevations

    if surface_tolerance_m is not None and not native_source_grid:
        if sample_spacing_m is None:
            raise ValueError("Adaptive terrain requires a reference spacing")
        tri = refine_terrain(tri, sample_surface, land_polygon(hex_poly, water_polygons),
                             sample_spacing_m, surface_tolerance_m, checkpoint=checkpoint, max_vertices=max_vertices,
                             max_reference_points=max_reference_points, max_passes=max_passes)
        vertices_2d = tri.vertices
    elevations = sample_surface(vertices_2d)

    top_vertices = np.column_stack([vertices_2d, elevations]).astype(np.float64)
    bottom_vertices = np.column_stack([vertices_2d, np.full(len(vertices_2d), base_z)]).astype(np.float64)
    n_top = len(top_vertices)

    all_vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)

    top_faces = tri.triangles.astype(np.int64).copy()
    xy = vertices_2d[top_faces]
    ab, ac = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    clockwise = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0] < 0
    top_faces[clockwise] = top_faces[clockwise, ::-1]
    bottom_faces = top_faces[:, ::-1] + n_top

    # Recover directed boundary edges from the CCW top faces. Triangle's segment
    # list does not promise winding; repairing every face afterwards was costly.
    edges = np.concatenate([top_faces[:, [0, 1]], top_faces[:, [1, 2]], top_faces[:, [2, 0]]])
    codes = np.min(edges, axis=1) * n_top + np.max(edges, axis=1)
    _, first, counts = np.unique(codes, return_index=True, return_counts=True)
    boundary = edges[first[counts == 1]]
    i, j = boundary.T
    walls = np.concatenate([
        np.column_stack((i, i + n_top, j + n_top)),
        np.column_stack((i, j + n_top, j)),
    ])
    all_faces = np.concatenate([top_faces, bottom_faces, walls])
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Terrain triangulation did not produce a closed, consistently wound solid")
    if mesh.volume < 0:
        mesh.invert()
    return mesh
