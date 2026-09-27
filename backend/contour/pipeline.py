"""Pipeline orchestration: Settings + Route -> MeshKit."""
from __future__ import annotations

import math

from dataclasses import dataclass
from collections.abc import Callable

import trimesh

from contour.biome_data import fetch_water_polygons
from contour.terrain_data import fetch_heightmap, EARTH_CIRCUMFERENCE_M
from contour.hex_frame import hex_frame_for_route
from contour.tile_cache import TileCache
from contour.http_client import HttpClient
from contour.plinth_mesh import build_plinth_mesh
from contour.route_mesh import build_route_mesh
from contour.terrain_mesh import build_land_mesh
from contour.terrain_cache import cached_terrain
from contour.water_mesh import build_water_mesh, shoreline_water_levels
from contour.kit import MeshKit
from contour.route import Route
from contour.settings import Settings
from contour.style import NeutralScene, Style
from contour.monochrome_biome import MonochromeBiome
from contour.production import TERRAIN_SAMPLE_MM, SURFACE_TOLERANCE_MM, PREVIEW_MAX_EXAGGERATION


WATER_THICKNESS_MM = 0.6


@dataclass
class PipelineDependencies:
    """External resources the pipeline needs to run.

    Held outside Settings because they are environment- and process-scoped
    (sockets, disk paths, secrets) rather than per-model state.
    """

    http_client: HttpClient
    tile_cache: TileCache
    mapbox_token: str


def build_kit(
    settings: Settings, route: Route, deps: PipelineDependencies,
    on_progress: Callable[[str], None] | None = None,
) -> MeshKit:
    """Run the full pipeline and return a styled MeshKit.

    Stages: framing -> terrain fetch -> biomes fetch -> Z planning ->
    neutral meshes -> style application.
    """
    report = on_progress or (lambda stage: None)
    report("frame")
    hex_frame = hex_frame_for_route(
        route,
        padding_ratio=settings.framing.padding_ratio,
        rotation_degrees=settings.framing.rotation_degrees,
    )

    report("terrain")
    heightmap = fetch_heightmap(
        hex_frame, settings.physical, deps.http_client, deps.tile_cache, deps.mapbox_token
    )

    report("water")
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
    minx, miny, maxx, maxy = hex_frame.polygon_enu().bounds
    mm_per_m = settings.physical.size_mm / max(maxx - minx, maxy - miny)
    exaggeration = settings.terrain.vertical_exaggeration
    water_levels = shoreline_water_levels(
        water_polygons, heightmap, hex_frame, bottom_z=land_base_z, recess_m=0,
    )

    # Keep the coloured insert a fixed physical thickness after exaggeration.
    # On very thin models leave at least half the available base as terrain.
    water_bottoms = [
        level - min(WATER_THICKNESS_MM / (mm_per_m * exaggeration), (level - land_base_z) * 0.5)
        for level in water_levels
    ]

    native_spacing_m = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(hex_frame.centre_lat)) / (heightmap.tile_size * 2**heightmap.zoom)

    report("land")
    # Route width/height, colours, and plinth settings never alter this surface.
    # A mesh validated at a larger print size is also sufficient for a smaller one.
    land = cached_terrain(
        deps.tile_cache.root / "_derived" / "terrain_meshes",
        identity={
            "source": settings.source.sha256,
            "framing": settings.framing.model_dump(),
            "water": settings.biomes.water.enabled,
            "zoom": heightmap.zoom,
            "exaggeration": max(exaggeration, PREVIEW_MAX_EXAGGERATION),
            "tolerance": SURFACE_TOLERANCE_MM,
            "sampling": TERRAIN_SAMPLE_MM,
        },
        quality_size_mm=settings.physical.size_mm,
        checkpoint=lambda: report("land"),
        build=lambda: build_land_mesh(
            hex_frame, heightmap, water_polygons, base_z=land_base_z, water_levels=water_levels,
            grid_points_per_side=24,
            surface_tolerance_m=SURFACE_TOLERANCE_MM / (mm_per_m * max(exaggeration, PREVIEW_MAX_EXAGGERATION)),
            sample_spacing_m=max(TERRAIN_SAMPLE_MM / mm_per_m, native_spacing_m / 2),
            checkpoint=lambda: report("land"),
        ),
    )

    report("details")
    water = None
    if water_polygons:
        water = build_water_mesh(water_polygons, top_z=water_levels, bottom_z=water_bottoms)
        foundation = build_water_mesh(water_polygons, top_z=water_bottoms, bottom_z=land_base_z)
        if foundation is not None:
            # Unite the under-water foundation with the terrain, removing shared
            # interior walls so the land exports as a single printable solid.
            land = foundation if land.is_empty else trimesh.boolean.union([land, foundation], engine="manifold")

    route_mesh = None
    if settings.route.enabled:
        width_m = settings.route.width_mm / mm_per_m
        height_m = settings.route.height_above_terrain_mm / mm_per_m
        route_mesh = build_route_mesh(
            route, hex_frame, heightmap, width_m=width_m, height_above_terrain_m=height_m,
            vertical_exaggeration=exaggeration, elevation_origin_m=land_base_z,
            surface_tolerance_m=SURFACE_TOLERANCE_MM / mm_per_m,
            sample_spacing_m=max(TERRAIN_SAMPLE_MM / mm_per_m, native_spacing_m / 2),
        )

    plinth = None
    if settings.plinth.enabled:
        plinth = build_plinth_mesh(hex_frame, height_m=plinth_height_m, top_z=land_base_z)

    # Exaggerate relief around the top of the fixed plinth. Route height was
    # added after exaggerating its base, so it remains a physical dimension.
    for mesh in (land, water):
        if mesh is not None:
            mesh.vertices[:, 2] = land_base_z + (mesh.vertices[:, 2] - land_base_z) * exaggeration
    origin_z = land_base_z - (plinth_height_m if plinth is not None else 0)
    for mesh in (land, water, route_mesh, plinth):
        if mesh is not None:
            mesh.apply_translation([0, 0, -origin_z])
            mesh.apply_scale(mm_per_m)
            if "_route_offset" in mesh.vertex_attributes:
                mesh.vertex_attributes["_route_offset"] *= mm_per_m

    scene = NeutralScene(land=land, water=water, route=route_mesh, plinth=plinth)
    return _resolve_style(settings.style.name).apply(scene, settings)


def _resolve_style(name: str) -> Style:
    if name == "monochrome-biome":
        return MonochromeBiome()
    raise ValueError(f"Unknown style: {name}")
