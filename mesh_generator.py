from __future__ import annotations
import logging
import numpy as np
import trimesh
from terrain_data import EnuTerrain
from route import EnuRoute
from mesh_builder import create_terrain_mesh, scale_mesh_for_printing, create_route_ribbon_mesh

log = logging.getLogger(__name__)


def build_terrain_mesh(
    terrain: EnuTerrain,
    target_size_mm: float,
    base_height: float = 50.0,
    max_resolution: int = 500,
) -> trimesh.Trimesh:
    """
    Build a terrain mesh (downsampled, watertight) scaled to the target print size in mm.
    """
    height, width = terrain.shape
    mesh_step = max(1, max(height, width) // max_resolution)

    e = terrain.e_grid[::mesh_step, ::mesh_step]
    n = terrain.n_grid[::mesh_step, ::mesh_step]
    u = terrain.u_grid[::mesh_step, ::mesh_step]

    log.info("Building terrain mesh: %d×%d heightmap → %s grid (step=%d)", height, width, e.shape, mesh_step)

    mesh = create_terrain_mesh(e, n, u, base_height=base_height)
    return scale_mesh_for_printing(mesh, target_size_mm=target_size_mm)


def build_route_mesh(
    route_enu: EnuRoute,
    terrain: EnuTerrain,
    target_size_mm: float,
    base_height: float = 50.0,
    route_height_ratio: float = 0.2,
    width: float = 20.0,
    thickness: float = 20.0,
) -> trimesh.Trimesh:
    """
    Build a route ribbon mesh, elevated above the terrain, scaled to match the terrain's print size.
    """
    route_terrain_elevation = terrain.sample_at_points(route_enu.e, route_enu.n)
    route_height = base_height * route_height_ratio

    log.info("Building route mesh: height=%.2fm (%.1f%% of %sm base)", route_height, route_height_ratio * 100, base_height)

    mesh = create_route_ribbon_mesh(
        route_enu.e,
        route_enu.n,
        route_terrain_elevation,
        width=width,
        height=route_height,
        thickness=thickness,
    )

    terrain_size_m = np.array(
        [
            terrain.e_grid.max() - terrain.e_grid.min(),
            terrain.n_grid.max() - terrain.n_grid.min(),
            terrain.u_grid.max() - terrain.u_grid.min(),
        ]
    )
    scale_factor = target_size_mm / (terrain_size_m.max() * 1000)
    mesh.apply_scale(scale_factor)
    return mesh
