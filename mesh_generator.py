from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from terrain_data import Terrain_ENU
from route import Route_ENU
from mesh_builder import create_terrain_mesh, scale_mesh_for_printing, create_route_ribbon_mesh


@dataclass(frozen=True)
class MeshGenerator:
    """
    Responsible for generating 3D meshes from terrain and route data.

    Responsibilities:
    - Create terrain mesh with base
    - Create route ribbon mesh
    - Scale meshes for printing
    - Export meshes to files
    """

    terrain: Terrain_ENU
    route_enu: Route_ENU
    route_terrain_elevation: np.ndarray

    def create_terrain_mesh(self, base_height: float = 50.0, max_resolution: int = 10_000):
        """
        Build 3D terrain mesh with base.

        Args:
            base_height: Height of base in meters
            max_resolution: Maximum dimension for mesh grid

        Returns:
            Trimesh object
        """
        print("\n" + "=" * 50)
        print("Building 3D terrain mesh...")

        # Downsample for reasonable file size
        height, width = self.terrain.shape
        mesh_step = max(1, max(height, width) // max_resolution)

        terrain_e_mesh = self.terrain.e_grid[::mesh_step, ::mesh_step]
        terrain_n_mesh = self.terrain.n_grid[::mesh_step, ::mesh_step]
        terrain_u_mesh = self.terrain.u_grid[::mesh_step, ::mesh_step]

        print(f"Original heightmap: {height}×{width}")
        print(f"Mesh grid: {terrain_e_mesh.shape} (step={mesh_step})")
        print(f"Mesh will have ~{terrain_e_mesh.shape[0] * terrain_e_mesh.shape[1] * 2} faces")

        mesh = create_terrain_mesh(terrain_e_mesh, terrain_n_mesh, terrain_u_mesh, base_height=base_height)

        print(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Is watertight: {mesh.is_watertight}")
        print(f"Is manifold: {mesh.is_winding_consistent}")

        return mesh

    def create_route_mesh(self, width: float = 20.0, height_ratio: float = 0.2, thickness: float = 20.0, base_height: float = 50.0):
        """
        Build 3D route ribbon mesh.

        Args:
            width: Width of route ribbon in meters
            height_ratio: Height above terrain as a ratio of base_height (default: 0.02 = 2% of base)
            thickness: Thickness of ribbon in meters
            base_height: Base height of the terrain model in meters
 
        Returns:
            Trimesh object
        """
        print("\nCreating route mesh...")
        
        # Calculate absolute height from ratio
        height = base_height * height_ratio

        mesh = create_route_ribbon_mesh(
            self.route_enu.e,
            self.route_enu.n,
            self.route_terrain_elevation,
            width=width,
            height=height,
            thickness=thickness,
        )

        return mesh

    def export_meshes(self, target_size_mm: float = 100.0, base_height: float = 50.0, route_height_ratio: float = 0.2):
        """
        Create, scale, and export both terrain and route meshes.

        Args:
            target_size_mm: Target model size in millimeters
            base_height: Height of the terrain base in meters (default: 50.0)
            route_height_ratio: Route height as proportion of base height (default: 0.02 = 2%)
        """
        # Create terrain mesh
        terrain_mesh = self.create_terrain_mesh(base_height=base_height, max_resolution=500)
        terrain_mesh = scale_mesh_for_printing(terrain_mesh, target_size_mm=target_size_mm)

        # Export terrain
        terrain_output = "terrain_model.stl"
        terrain_mesh.export(terrain_output)
        print(f"\n✓ Exported terrain mesh to: {terrain_output}")
        print(f"  Vertices: {len(terrain_mesh.vertices)}")
        print(f"  Faces: {len(terrain_mesh.faces)}")
        print(f"  Volume: {terrain_mesh.volume:.2f} mm³")

        # Create route mesh with height proportional to base height
        route_mesh = self.create_route_mesh(
            width=20.0, 
            height_ratio=route_height_ratio, 
            thickness=20.0,
            base_height=base_height
        )
        print(f"  Route height: {base_height * route_height_ratio:.2f}m ({route_height_ratio*100:.1f}% of {base_height}m base)")

        # Calculate scale factor from terrain mesh
        terrain_bounds_m = np.array(
            [
                [self.terrain.e_grid.min(), self.terrain.n_grid.min(), self.terrain.u_grid.min()],
                [self.terrain.e_grid.max(), self.terrain.n_grid.max(), self.terrain.u_grid.max()],
            ]
        )
        terrain_size_m = terrain_bounds_m[1] - terrain_bounds_m[0]
        terrain_max_m = max(terrain_size_m)
        scale_factor = target_size_mm / (terrain_max_m * 1000)

        route_mesh.apply_scale(scale_factor)

        # Export route
        route_output = "route_model.stl"
        route_mesh.export(route_output)
        print(f"\n✓ Exported route mesh to: {route_output}")
        print(f"  Vertices: {len(route_mesh.vertices)}")
        print(f"  Faces: {len(route_mesh.faces)}")
        print(f"  Volume: {route_mesh.volume:.2f} mm³")

    @classmethod
    def new(cls, terrain: Terrain_ENU, route_enu: Route_ENU) -> MeshGenerator:
        """
        Create MeshGenerator with terrain elevation sampled at route points.

        Args:
            terrain: TerrainData object
            route_enu: Route in ENU coordinates

        Returns:
            MeshGenerator instance
        """
        route_terrain_u = terrain.sample_at_points(route_enu.e, route_enu.n)
        return cls(terrain=terrain, route_enu=route_enu, route_terrain_elevation=route_terrain_u)
