from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from coordinate_transform import LonLatToENU


@dataclass(frozen=True)
class Terrain_ENU:
    """
    Represents terrain elevation data in ENU coordinates.

    Responsibilities:
    - Convert heightmap from tile coordinates to ENU
    - Store terrain grids (E, N, U)
    - Provide terrain bounds information
    """

    e_grid: np.ndarray
    n_grid: np.ndarray
    u_grid: np.ndarray

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Returns ((e_min, e_max), (n_min, n_max), (u_min, u_max))"""
        return (
            (float(self.e_grid.min()), float(self.e_grid.max())),
            (float(self.n_grid.min()), float(self.n_grid.max())),
            (float(self.u_grid.min()), float(self.u_grid.max())),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Returns (height, width) of terrain grids"""
        return self.e_grid.shape

    def sample_at_points(self, e_points: np.ndarray, n_points: np.ndarray) -> np.ndarray:
        """
        Sample terrain elevation at given ENU points using nearest neighbor.

        Args:
            e_points: East coordinates to sample
            n_points: North coordinates to sample

        Returns:
            Array of elevation values at the given points
        """
        terrain_points = np.column_stack([self.e_grid.ravel(), self.n_grid.ravel()])
        tree = cKDTree(terrain_points)

        query_points = np.column_stack([e_points, n_points])
        _, indices = tree.query(query_points)

        return self.u_grid.ravel()[indices]

    def downsample(self, target_size: int = 500) -> Terrain_ENU:
        """
        Downsample terrain for visualization or faster processing.

        Args:
            target_size: Target maximum dimension

        Returns:
            Downsampled TerrainData
        """
        height, width = self.shape
        step = max(1, max(height, width) // target_size)

        return Terrain_ENU(e_grid=self.e_grid[::step, ::step], n_grid=self.n_grid[::step, ::step], u_grid=self.u_grid[::step, ::step])

    @classmethod
    def new(cls, heightmap: np.ndarray, tiles: list, zoom: int, transform: LonLatToENU) -> Terrain_ENU:
        """
        Convert heightmap from tile coordinates to ENU coordinates.

        Args:
            heightmap: 2D elevation array
            tiles: List of tiles used to create heightmap
            zoom: Zoom level of tiles
            transform: Coordinate transform for geodetic to ENU conversion

        Returns:
            TerrainData with ENU coordinate grids
        """
        print("Converting heightmap to ENU coordinates...")
        height, width = heightmap.shape

        # Get tile bounds
        tile_xs = [tile.x for tile in tiles]
        tile_ys = [tile.y for tile in tiles]
        tile_x_min = min(tile_xs)
        tile_y_min = min(tile_ys)

        # Create meshgrid for all pixel coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Convert pixels to tile coordinates (vectorized)
        tile_x_grid = tile_x_min + cols / 256.0
        tile_y_grid = tile_y_min + rows / 256.0

        # Convert tile coordinates to lat/lon (vectorized)
        n = 2**zoom
        lon_grid = tile_x_grid / n * 360.0 - 180.0
        lat_rad_grid = np.arctan(np.sinh(np.pi * (1 - 2 * tile_y_grid / n)))
        lat_grid = np.degrees(lat_rad_grid)

        # Flatten for batch conversion to ENU
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        ele_flat = heightmap.ravel()

        # Batch convert to ENU
        enu_coords = transform.lonlat_to_enu(lat_flat, lon_flat, ele_flat)

        # Reshape back to 2D grids
        e_grid = enu_coords[:, 0].reshape(height, width)
        n_grid = enu_coords[:, 1].reshape(height, width)
        u_grid = heightmap.copy()

        print("ENU conversion complete!")
        e_bounds, n_bounds, _ = cls(e_grid, n_grid, u_grid).bounds
        print(f"Terrain ENU bounds: E=[{e_bounds[0]:.1f}, {e_bounds[1]:.1f}], N=[{n_bounds[0]:.1f}, {n_bounds[1]:.1f}]")

        return cls(e_grid=e_grid, n_grid=n_grid, u_grid=u_grid)
