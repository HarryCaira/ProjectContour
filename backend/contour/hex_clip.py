"""Hex polygon clipping + constrained Delaunay triangulation for the land polygon."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
import triangle as tr
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import orient, unary_union


@dataclass(frozen=True)
class LandTriangulation:
    """Result of triangulating the land polygon (hex minus water).

    vertices: (N, 2) array of (E, N) coordinates.
    triangles: (M, 3) array of vertex indices.
    boundary_segments: list of (i, j) index pairs that lie on a wall boundary —
                       either the hex outer edge or a water hole edge. Used to
                       build side walls for the 3D mesh.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    boundary_segments: list[tuple[int, int]]


def land_polygon(hex_polygon: Polygon, water_polygons: list[Polygon]) -> Polygon | MultiPolygon:
    """Compute (hex \\ water_union) with OGC orientation (CCW exterior, CW holes)."""
    if water_polygons:
        water_union = unary_union(water_polygons)
        result = hex_polygon.difference(water_union)
    else:
        result = hex_polygon

    if result.geom_type == "Polygon":
        return orient(result, sign=1.0)
    if result.geom_type == "MultiPolygon":
        return MultiPolygon([orient(p, sign=1.0) for p in result.geoms])
    raise ValueError(f"Unexpected geometry type from land polygon: {result.geom_type}")


def triangulate_land(
    hex_polygon: Polygon,
    water_polygons: list[Polygon],
    grid_points_per_side: int = 100,
    interior_points: np.ndarray | None = None,
) -> LandTriangulation:
    """Constrained Delaunay triangulation of (hex \\ water) with an interior grid.

    The `grid_points_per_side` parameter controls heightmap-sampling density: a
    regular grid is generated across the hex bbox and points falling inside the
    land polygon are added as Steiner vertices, giving the mesh real terrain detail.
    """
    if grid_points_per_side < 1:
        raise ValueError("grid_points_per_side must be positive")
    land = land_polygon(hex_polygon, water_polygons)
    minx, miny, maxx, maxy = hex_polygon.bounds
    spacing = max(maxx - minx, maxy - miny) / grid_points_per_side
    # Clean cut vertices before splitting disconnected land into components.
    # Recursive calls no longer carry the water polygons, so cleaning only
    # those calls with water leaves near-zero island edges in the triangulation.
    if water_polygons:
        land = shapely.set_precision(land, grid_size=spacing * 1e-8)
    if isinstance(land, MultiPolygon):
        pieces = [triangulate_land(poly, [], grid_points_per_side, interior_points) for poly in land.geoms]
        vertices, triangles, segments = [], [], []
        offset = 0
        for piece in pieces:
            vertices.append(piece.vertices)
            triangles.append(piece.triangles + offset)
            segments.extend((i + offset, j + offset) for i, j in piece.boundary_segments)
            offset += len(piece.vertices)
        return LandTriangulation(np.concatenate(vertices), np.concatenate(triangles), segments)
    land = orient(land, sign=1.0)
    # Sample cut edges as densely as the interior. Corner-only elevation
    # samples interpolate straight across valleys and create raised fins.
    land = shapely.segmentize(land, max_segment_length=spacing)
    polygons = [land] if land.geom_type == "Polygon" else list(land.geoms)

    boundary_vertices: list[tuple[float, float]] = []
    boundary_segments: list[tuple[int, int]] = []
    hole_points: list[tuple[float, float]] = []

    for poly in polygons:
        # Exterior ring (CCW by orient)
        exterior_start = len(boundary_vertices)
        ext_coords = list(poly.exterior.coords)[:-1]  # last point is a duplicate of the first
        boundary_vertices.extend(ext_coords)
        n_ext = len(ext_coords)
        for i in range(n_ext):
            j = (i + 1) % n_ext
            boundary_segments.append((exterior_start + i, exterior_start + j))

        # Interior rings (CW by orient) — water holes
        for interior in poly.interiors:
            interior_start = len(boundary_vertices)
            int_coords = list(interior.coords)[:-1]
            boundary_vertices.extend(int_coords)
            n_int = len(int_coords)
            for i in range(n_int):
                j = (i + 1) % n_int
                boundary_segments.append((interior_start + i, interior_start + j))

            hole_polygon = Polygon(int_coords)
            rep = hole_polygon.representative_point()
            hole_points.append((rep.x, rep.y))

    # Interior grid points for terrain detail. We sample over the hex bbox, then
    # keep only those strictly inside the land polygon (with a small inset so
    # they don't fight the boundary).
    if interior_points is None:
        interior_grid = _interior_grid_points(hex_polygon, land, grid_points_per_side)
    else:
        # Native raster samples replace the coarse seed grid. Retain every
        # interior sample except numerically coincident boundary vertices.
        inset_land = land.buffer(-spacing * 1e-8)
        interior_grid = interior_points[shapely.contains_xy(inset_land, interior_points[:, 0], interior_points[:, 1])]

    all_vertices = np.concatenate((np.asarray(boundary_vertices, dtype=np.float64), interior_grid))

    triangulation_input: dict = {
        "vertices": all_vertices,
        "segments": np.array(boundary_segments, dtype=np.int32),
    }
    if hole_points:
        triangulation_input["holes"] = np.array(hole_points, dtype=np.float64)

    result = tr.triangulate(triangulation_input, "p")

    return LandTriangulation(
        vertices=np.asarray(result["vertices"], dtype=np.float64),
        triangles=np.asarray(result["triangles"], dtype=np.int64),
        boundary_segments=[tuple(map(int, segment)) for segment in result["segments"]],
    )


def _interior_grid_points(
    hex_polygon: Polygon, land: Polygon | MultiPolygon, n_per_side: int
) -> np.ndarray:
    minx, miny, maxx, maxy = hex_polygon.bounds
    step = max(maxx - minx, maxy - miny) / n_per_side
    inset = step * 0.5  # avoid placing grid points exactly on boundary edges

    xs = np.arange(minx + inset, maxx - inset + 1e-9, step)
    ys = np.arange(miny + inset, maxy - inset + 1e-9, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Shrink the land polygon by `inset` so interior grid points stay well clear
    # of the boundary segments (otherwise CDT can produce sliver triangles).
    shrunken = land.buffer(-inset)
    if shrunken.is_empty:
        return np.empty((0, 2))

    mask = shapely.contains_xy(shrunken, points[:, 0], points[:, 1])
    return points[mask]
