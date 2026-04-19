from __future__ import annotations
import logging
import numpy as np
import trimesh

log = logging.getLogger(__name__)


def create_terrain_mesh(
    terrain_e: np.ndarray,
    terrain_n: np.ndarray,
    terrain_u: np.ndarray,
    base_height: float = 10.0,
) -> trimesh.Trimesh:
    """
    Create a watertight 3D mesh from terrain grids (E/N/U), with a flat base
    `base_height` below the minimum elevation.
    """
    height, width = terrain_u.shape

    # Create vertices for the top surface
    vertices = []
    for i in range(height):
        for j in range(width):
            vertices.append([terrain_e[i, j], terrain_n[i, j], terrain_u[i, j]])

    vertices = np.array(vertices)

    # Create triangular faces for the top surface using Delaunay-like grid
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Two triangles per grid square
            idx = i * width + j
            # Triangle 1
            faces.append([idx, idx + width, idx + 1])
            # Triangle 2
            faces.append([idx + 1, idx + width, idx + width + 1])

    faces = np.array(faces)

    # Create the top surface mesh
    top_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Find the boundary of the terrain
    min_u = terrain_u.min()
    base_z = min_u - base_height

    # Create base vertices (project top boundary vertices down)
    base_vertices = vertices.copy()
    base_vertices[:, 2] = base_z

    # Create bottom face (flip face normals)
    bottom_faces = faces[:, ::-1].copy()

    # Offset bottom face indices
    bottom_faces += len(vertices)

    # Create side walls by connecting edges
    side_faces = []

    # Connect edges around the perimeter
    # Top edge (j=width-1)
    for i in range(height - 1):
        top_idx1 = i * width + (width - 1)
        top_idx2 = (i + 1) * width + (width - 1)
        bot_idx1 = top_idx1 + len(vertices)
        bot_idx2 = top_idx2 + len(vertices)
        side_faces.append([top_idx1, bot_idx1, top_idx2])
        side_faces.append([top_idx2, bot_idx1, bot_idx2])

    # Right edge (i=height-1)
    for j in range(width - 1):
        top_idx1 = (height - 1) * width + j
        top_idx2 = (height - 1) * width + (j + 1)
        bot_idx1 = top_idx1 + len(vertices)
        bot_idx2 = top_idx2 + len(vertices)
        side_faces.append([top_idx1, top_idx2, bot_idx1])
        side_faces.append([top_idx2, bot_idx2, bot_idx1])

    # Bottom edge (j=0)
    for i in range(height - 1):
        top_idx1 = i * width
        top_idx2 = (i + 1) * width
        bot_idx1 = top_idx1 + len(vertices)
        bot_idx2 = top_idx2 + len(vertices)
        side_faces.append([top_idx1, top_idx2, bot_idx1])
        side_faces.append([top_idx2, bot_idx2, bot_idx1])

    # Left edge (i=0)
    for j in range(width - 1):
        top_idx1 = j
        top_idx2 = j + 1
        bot_idx1 = top_idx1 + len(vertices)
        bot_idx2 = top_idx2 + len(vertices)
        side_faces.append([top_idx1, bot_idx1, top_idx2])
        side_faces.append([top_idx2, bot_idx1, bot_idx2])

    # Combine all vertices and faces
    all_vertices = np.vstack([vertices, base_vertices])
    all_faces = np.vstack([faces, bottom_faces, np.array(side_faces)])

    # Create the complete mesh
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)

    log.info("Terrain mesh: %d vertices, %d faces (watertight=%s, manifold=%s)", len(mesh.vertices), len(mesh.faces), mesh.is_watertight, mesh.is_winding_consistent)

    return mesh


def scale_mesh_for_printing(mesh: trimesh.Trimesh, target_size_mm: float = 100.0) -> trimesh.Trimesh:
    """
    Scale mesh to target print size.

    Args:
        mesh: Input mesh (in meters)
        target_size_mm: Desired size of longest dimension in mm

    Returns:
        Scaled mesh
    """
    # Get current size in meters
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    max_size_m = max(size)

    # Calculate scale factor (meters to mm)
    scale_factor = target_size_mm / (max_size_m * 1000)

    log.info("Scaling from %.1fm to %smm (scale: %.6f)", max_size_m, target_size_mm, scale_factor)

    # Scale the mesh
    mesh.apply_scale(scale_factor)

    return mesh


def create_route_ribbon_mesh(
    route_e: np.ndarray,
    route_n: np.ndarray,
    route_u: np.ndarray,
    width: float = 4.0,
    height: float = 1.0,
    thickness: float = 0.5,
) -> trimesh.Trimesh:
    """
    Create a solid ribbon mesh following the route path.

    Args:
        route_e: East coordinates of route points
        route_n: North coordinates of route points
        route_u: Elevation (Up) of route points
        width: Width of the ribbon
        height: Height above the route points
        thickness: Thickness of the ribbon (for manifold)

    Returns:
        trimesh.Trimesh: Solid ribbon mesh following the route
    """
    # Downsample for reasonable mesh size
    step = max(1, len(route_e) // 500)
    route_e = route_e[::step]
    route_n = route_n[::step]
    route_u = route_u[::step]

    n_points = len(route_e)

    if n_points < 2:
        raise ValueError("Route must have at least 2 points")

    vertices = []
    faces = []

    # Create top and bottom surfaces
    for i in range(n_points):
        pos = np.array([route_e[i], route_n[i], route_u[i] + height])

        # Calculate perpendicular direction using average of adjacent segments
        if i == 0:
            forward = np.array([route_e[1] - route_e[0], route_n[1] - route_n[0], 0])
        elif i == n_points - 1:
            forward = np.array([route_e[i] - route_e[i - 1], route_n[i] - route_n[i - 1], 0])
        else:
            forward = np.array([route_e[i + 1] - route_e[i - 1], route_n[i + 1] - route_n[i - 1], 0])

        forward_len = np.linalg.norm(forward)
        if forward_len > 0:
            forward = forward / forward_len
        else:
            forward = np.array([1, 0, 0])

        # Perpendicular is 90° rotation in XY plane
        perpendicular = np.array([-forward[1], forward[0], 0])

        # Create 4 vertices per position: top-left, top-right, bottom-left, bottom-right
        top_left = pos - perpendicular * (width / 2)
        top_right = pos + perpendicular * (width / 2)
        bottom_left = top_left - np.array([0, 0, thickness])
        bottom_right = top_right - np.array([0, 0, thickness])

        vertices.extend([top_left, top_right, bottom_left, bottom_right])

        # Create faces connecting to previous segment
        if i > 0:
            prev_base = (i - 1) * 4
            curr_base = i * 4

            # Top surface (2 triangles)
            faces.append([prev_base, curr_base, prev_base + 1])
            faces.append([prev_base + 1, curr_base, curr_base + 1])

            # Bottom surface (2 triangles, reversed winding)
            faces.append([prev_base + 2, prev_base + 3, curr_base + 2])
            faces.append([prev_base + 3, curr_base + 3, curr_base + 2])

            # Left edge (2 triangles)
            faces.append([prev_base, prev_base + 2, curr_base])
            faces.append([prev_base + 2, curr_base + 2, curr_base])

            # Right edge (2 triangles)
            faces.append([prev_base + 1, curr_base + 1, prev_base + 3])
            faces.append([prev_base + 3, curr_base + 1, curr_base + 3])

    # Add end caps
    # Start cap (4 triangles forming a rectangle)
    faces.append([0, 2, 1])
    faces.append([1, 2, 3])

    # End cap (4 triangles forming a rectangle)
    last_base = (n_points - 1) * 4
    faces.append([last_base, last_base + 1, last_base + 2])
    faces.append([last_base + 1, last_base + 3, last_base + 2])

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create mesh with processing enabled
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.merge_vertices()

    log.info("Route ribbon mesh: %d vertices, %d faces (watertight=%s, manifold=%s)", len(mesh.vertices), len(mesh.faces), mesh.is_watertight, mesh.is_winding_consistent)

    return mesh
