"""Pipeline orchestration: Settings + Route -> MeshKit."""
from __future__ import annotations

from dataclasses import dataclass

from contour.data.biomes import fetch_water_polygons
from contour.data.terrain import fetch_heightmap
from contour.framing.hex import hex_frame_for_route
from contour.http.cache import TileCache
from contour.http.client import HttpClient
from contour.mesh.plinth import build_plinth_mesh
from contour.mesh.route import build_route_mesh
from contour.mesh.terrain import build_land_mesh
from contour.mesh.water import build_water_mesh
from contour.schema.kit import MeshKit
from contour.schema.route import Route
from contour.schema.settings import Settings
from contour.styles.base import NeutralScene, Style
from contour.styles.monochrome_biome import MonochromeBiome


@dataclass
class PipelineDependencies:
    """External resources the pipeline needs to run.

    Held outside Settings because they are environment- and process-scoped
    (sockets, disk paths, secrets) rather than per-model state.
    """

    http_client: HttpClient
    tile_cache: TileCache
    mapbox_token: str


def build_kit(settings: Settings, route: Route, deps: PipelineDependencies) -> MeshKit:
    """Run the full pipeline and return a styled MeshKit.

    Stages: framing -> terrain fetch -> biomes fetch -> Z planning ->
    neutral meshes -> style application.
    """
    hex_frame = hex_frame_for_route(
        route,
        padding_ratio=settings.framing.padding_ratio,
        rotation_degrees=settings.framing.rotation_degrees,
    )

    heightmap = fetch_heightmap(
        hex_frame, settings.physical, deps.http_client, deps.tile_cache, deps.mapbox_token
    )

    water_polygons = []
    if settings.biomes.water.enabled:
        water_polygons = fetch_water_polygons(
            hex_frame, deps.http_client, deps.tile_cache, deps.mapbox_token
        )

    # Z planning — everything is in metres at this point.
    elev_min = float(heightmap.elevations.min())
    elev_max = float(heightmap.elevations.max())
    elev_range = max(elev_max - elev_min, 1.0)
    model_world_diameter_m = 2 * hex_frame.circumradius_m
    base_thickness_m = max(0.05 * elev_range, 0.005 * model_world_diameter_m)
    land_base_z = elev_min - base_thickness_m
    plinth_height_m = 0.05 * model_world_diameter_m
    water_top_z = elev_min - settings.biomes.water.depth_fraction * elev_range

    land = build_land_mesh(hex_frame, heightmap, water_polygons, base_z=land_base_z)

    water = None
    if water_polygons:
        water = build_water_mesh(water_polygons, top_z=water_top_z, bottom_z=land_base_z)

    route_mesh = None
    if settings.route.enabled:
        mm_per_m = settings.physical.size_mm / model_world_diameter_m
        width_m = settings.route.width_mm / mm_per_m
        height_m = settings.route.height_above_terrain_mm / mm_per_m
        route_mesh = build_route_mesh(
            route, hex_frame, heightmap, width_m=width_m, height_above_terrain_m=height_m
        )

    plinth = None
    if settings.plinth.enabled:
        plinth = build_plinth_mesh(hex_frame, height_m=plinth_height_m, top_z=land_base_z)

    scene = NeutralScene(land=land, water=water, route=route_mesh, plinth=plinth)
    return _resolve_style(settings.style.name).apply(scene, settings)


def _resolve_style(name: str) -> Style:
    if name == "monochrome-biome":
        return MonochromeBiome()
    raise ValueError(f"Unknown style: {name}")
