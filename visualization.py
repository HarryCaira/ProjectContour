from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from terrain_data import Terrain_ENU
from route import Route_ENU


@dataclass(frozen=True)
class VisualizationBuilder:
    """
    Responsible for creating interactive 3D visualizations.

    Responsibilities:
    - Create Plotly 3D surface plots
    - Add route overlays
    - Configure visualization settings
    """

    terrain: Terrain_ENU
    route_enu: Route_ENU
    route_terrain_elevation: np.ndarray

    def create_3d_visualization(self, downsample_size: int = 500) -> go.Figure:
        """
        Create interactive 3D surface plot with route overlay.

        Args:
            downsample_size: Target size for downsampling terrain

        Returns:
            Plotly Figure object
        """
        # Downsample for performance
        terrain_down = self.terrain.downsample(target_size=downsample_size)

        # Calculate aspect ratio
        e_bounds, n_bounds, u_bounds = terrain_down.bounds
        e_range = e_bounds[1] - e_bounds[0]
        n_range = n_bounds[1] - n_bounds[0]
        u_range = u_bounds[1] - u_bounds[0]

        max_range = max(e_range, n_range, u_range)
        aspect_x = e_range / max_range
        aspect_y = n_range / max_range
        aspect_z = u_range / max_range

        # Create figure
        fig = go.Figure()

        # Add terrain surface
        fig.add_trace(
            go.Surface(
                x=terrain_down.e_grid,
                y=terrain_down.n_grid,
                z=terrain_down.u_grid,
                colorscale="Earth",
                colorbar=dict(title="Elevation (m)"),
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.8, fresnel=0.2),
                contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))),
                showscale=True,
            )
        )

        # Add route as 3D line
        fig.add_trace(
            go.Scatter3d(
                x=self.route_enu.e,
                y=self.route_enu.n,
                z=self.route_terrain_elevation + 1,  # Slightly above terrain
                mode="lines+markers",
                line=dict(color="red", width=6),
                marker=dict(size=3, color="red"),
                name="GPX Route",
                showlegend=True,
            )
        )

        # Configure layout
        height, width = self.terrain.shape
        fig.update_layout(
            title=f"Terrain with GPX Route (ENU coordinates, {height}×{width} pixels)",
            scene=dict(
                xaxis_title="East (m)",
                yaxis_title="North (m)",
                zaxis_title="Up (m)",
                aspectmode="manual",
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=1200,
            height=800,
        )

        return fig

    def show(self, downsample_size: int = 500) -> None:
        """Create and display 3D visualization."""
        fig = self.create_3d_visualization(downsample_size=downsample_size)
        fig.show()

    @classmethod
    def new(cls, terrain: Terrain_ENU, route_enu: EnuRoute, route_terrain_elevation: np.ndarray) -> VisualizationBuilder:
        """Create VisualizationBuilder from terrain and route data."""
        return cls(terrain=terrain, route_enu=route_enu, route_terrain_elevation=route_terrain_elevation)
