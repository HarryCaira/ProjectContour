"""Route mesh: GPX line -> ribbon extruded above the land surface."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from contour.hex_frame import HexFrame
from contour.sampling import sample_at_enu
from contour.heightmap import Heightmap
from contour.route import Route
from contour.route_detail import resample_route, simplify_route
from contour.production import PREVIEW_MAX_EXAGGERATION


def build_route_mesh(
    route: Route,
    hex_frame: HexFrame,
    heightmap: Heightmap,
    width_m: float,
    height_above_terrain_m: float,
    max_segments: int = 500,
    max_points: int = 200_000,
    vertical_exaggeration: float = 1.0,
    elevation_origin_m: float = 0.0,
    surface_tolerance_m: float | None = None,
    sample_spacing_m: float | None = None,
    surface_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
) -> trimesh.Trimesh | None:
    """Build a watertight ribbon that follows the route, sampled onto the terrain.

    The ribbon has a square cross-section of width × height_above_terrain, rests
    on the sampled terrain, and is closed at both ends with vertical caps.

    Returns None if the route has fewer than 2 points or all points fall outside
    the hex (the route is purely outside the model bounds).
    """
    if route.num_points < 2:
        return None

    if len(route.segment_starts) > 1:
        segments = [segment for segment in route.segments() if segment.num_points >= 2]
        if surface_tolerance_m is not None:
            if sample_spacing_m is None or sample_spacing_m <= 0 or surface_tolerance_m <= 0:
                raise ValueError("Route detail requires positive spacing and tolerance")
            # The budget covers the whole file, not each recording independently.
            count = 0
            for segment in segments:
                xy = hex_frame.local_enu().to_enu(segment.latitudes, segment.longitudes, 0.0)[:, :2]
                count += int(np.maximum(1, np.ceil(np.linalg.norm(np.diff(xy, axis=0), axis=1) / sample_spacing_m)).sum())
            if count > max_points:
                raise ValueError("Route exceeds the production mesh budget")
        pieces = [build_route_mesh(segment, hex_frame, heightmap, width_m, height_above_terrain_m,
                  max_segments=max_segments, max_points=max_points,
                  vertical_exaggeration=vertical_exaggeration, elevation_origin_m=elevation_origin_m,
                  surface_tolerance_m=surface_tolerance_m, sample_spacing_m=sample_spacing_m,
                  surface_sampler=surface_sampler) for segment in segments]
        pieces = [piece for piece in pieces if piece is not None]
        if not pieces:
            return None
        combined = trimesh.util.concatenate(pieces)
        # Preserve the custom GLB attribute used for instant route resizing.
        combined.vertex_attributes["_route_offset"] = np.concatenate([
            piece.vertex_attributes["_route_offset"] for piece in pieces])
        return combined

    local_enu = hex_frame.local_enu()
    enu_pts = local_enu.to_enu(route.latitudes, route.longitudes, 0.0)
    pts_2d = enu_pts[:, :2]

    # Downsample to keep the mesh tractable.
    if surface_tolerance_m is not None:
        if sample_spacing_m is None or sample_spacing_m <= 0 or surface_tolerance_m <= 0:
            raise ValueError("Route detail requires positive spacing and tolerance")
        # Dense GPX recordings should not spend the entire mesh budget on
        # redundant input points. Bound horizontal error below surface tolerance.
        pts_2d = pts_2d[simplify_route(pts_2d, min(surface_tolerance_m, sample_spacing_m * .1))]
        pts_2d = resample_route(pts_2d, sample_spacing_m, max_points=max_points)
    elif len(pts_2d) > max_segments:
        idx = np.linspace(0, len(pts_2d) - 1, max_segments + 1).astype(int)
        pts_2d = pts_2d[idx]

    n = len(pts_2d)
    if n < 2:
        return None

    terrain_z = surface_sampler(pts_2d) if surface_sampler is not None else sample_at_enu(heightmap, pts_2d, local_enu)
    if surface_tolerance_m is not None:
        metric = np.column_stack((pts_2d, terrain_z * max(vertical_exaggeration, PREVIEW_MAX_EXAGGERATION)))
        keep = simplify_route(metric, surface_tolerance_m)
        pts_2d, terrain_z = pts_2d[keep], terrain_z[keep]
        n = len(pts_2d)
        if n < 2:
            return None
    terrain_z = elevation_origin_m + (terrain_z - elevation_origin_m) * vertical_exaggeration

    # Tangent at each point (central differences in the interior, one-sided at ends).
    tangents = np.zeros_like(pts_2d)
    tangents[0] = pts_2d[1] - pts_2d[0]
    tangents[-1] = pts_2d[-1] - pts_2d[-2]
    tangents[1:-1] = pts_2d[2:] - pts_2d[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    tangents = tangents / norms

    # Perpendicular in the ground plane (CCW rotation of tangent).
    perps = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    half_w = width_m / 2.0
    bottom_z = terrain_z
    top_z = terrain_z + height_above_terrain_m

    # Four vertices per cross-section: top-left, top-right, bottom-left, bottom-right.
    tl = np.column_stack([pts_2d + perps * half_w, top_z])
    tr = np.column_stack([pts_2d - perps * half_w, top_z])
    bl = np.column_stack([pts_2d + perps * half_w, bottom_z])
    br = np.column_stack([pts_2d - perps * half_w, bottom_z])

    vertices = np.empty((4 * n, 3), dtype=np.float64)
    vertices[0::4] = tl
    vertices[1::4] = tr
    vertices[2::4] = bl
    vertices[3::4] = br

    faces: list[list[int]] = []
    for i in range(n - 1):
        a = i * 4  # current section base index
        b = (i + 1) * 4  # next section base index
        # Vertex offsets: +0 TL, +1 TR, +2 BL, +3 BR
        # Top face (looking down: normal +Z)
        faces.append([a + 0, b + 0, b + 1])
        faces.append([a + 0, b + 1, a + 1])
        # Bottom face (looking up: normal -Z)
        faces.append([a + 2, b + 1 + 2, b + 0 + 2])  # BL_a, BR_b, BL_b
        faces.append([a + 2, a + 3, b + 3])  # BL_a, BR_a, BR_b
        # Left side (outward in +perp direction)
        faces.append([a + 0, a + 2, b + 2])
        faces.append([a + 0, b + 2, b + 0])
        # Right side (outward in -perp direction)
        faces.append([a + 1, b + 1, b + 3])
        faces.append([a + 1, b + 3, a + 3])

    # End caps
    # Start cap (looking back along route, i.e. normal in -tangent direction)
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])
    # End cap (last section, normal in +tangent direction)
    last = (n - 1) * 4
    faces.append([last, last + 1, last + 3])
    faces.append([last, last + 3, last + 2])

    # Per-vertex displacement from the route's terrain centreline. The viewer
    # can change width/height without rebuilding terrain or downloading a GLB.
    centres = np.repeat(np.column_stack((pts_2d, terrain_z)), 4, axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=True,
                           vertex_attributes={"_route_offset": vertices - centres})
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh
