"""Pipeline orchestration: Settings + Route -> MeshKit."""
from __future__ import annotations

import math

from dataclasses import dataclass
from collections.abc import Callable

import trimesh
import shapely
from shapely.affinity import scale as scale_polygon

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
from contour.kit import MeshKit, KitPart, Material
from contour.landcover import fetch_landcover_regions
from contour.landcover_mesh import split_surface_material
from contour.snow import snow_region
from contour.infrastructure import fetch_infrastructure, build_buildings
from contour.route import Route
from contour.settings import Settings, DetailSettings
from contour.style import NeutralScene, Style
from contour.monochrome_biome import MonochromeBiome
from contour.sampling import sample_at_enu
from contour.surface_processing import smooth_heightmap, shoreline_sampler
from contour.production import PREVIEW_MAX_EXAGGERATION


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
    printable_landcover: bool = False,
) -> MeshKit:
    """Run the full pipeline and return a styled MeshKit.

    Stages: framing -> terrain fetch -> biomes fetch -> Z planning ->
    neutral meshes -> style application.
    """
    source_detail = settings.terrain.maximum_source_detail
    detail = settings.terrain.detail or DetailSettings(force_source_zoom=source_detail, smoothing_max_mm=0 if source_detail else 0.05)
    report = on_progress or (lambda stage: None)
    report("frame")
    hex_frame = hex_frame_for_route(
        route,
        padding_ratio=settings.framing.padding_ratio,
    )

    report("terrain")
    heightmap = fetch_heightmap(
        hex_frame, settings.physical, deps.http_client, deps.tile_cache, deps.mapbox_token,
        maximum_source_detail=detail.force_source_zoom, sample_mm=detail.sample_mm,
        max_zoom=detail.max_zoom, max_tiles=detail.max_tiles,
    )

    report("water")
    water_polygons = []
    if settings.biomes.water.enabled:
        water_polygons = fetch_water_polygons(
            hex_frame, deps.http_client, deps.tile_cache, deps.mapbox_token, zoom=detail.water_zoom
        )

    bounds = hex_frame.polygon_enu().bounds
    mm_per_m = settings.physical.size_mm / max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if detail.smoothing_sigma > 0 and detail.smoothing_max_mm > 0:
        heightmap = smooth_heightmap(heightmap, mm_per_m, sigma=detail.smoothing_sigma, max_mm=detail.smoothing_max_mm)

    # Z planning — everything is in metres at this point.
    elev_min = float(heightmap.elevations.min())
    elev_max = float(heightmap.elevations.max())
    elev_range = max(elev_max - elev_min, 1.0)
    model_world_diameter_m = 2 * hex_frame.circumradius_m
    base_thickness_m = max(0.05 * elev_range, 0.005 * model_world_diameter_m)
    land_base_z = elev_min - base_thickness_m
    plinth_height_m = 0.05 * model_world_diameter_m
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

    source_detail = settings.terrain.maximum_source_detail
    surface_tolerance = (None if source_detail else
                         detail.tolerance_mm / (mm_per_m * max(exaggeration, PREVIEW_MAX_EXAGGERATION)))
    sample_spacing = native_spacing_m / 2 if source_detail else max(detail.sample_mm / mm_per_m, native_spacing_m / 2)

    surface = shoreline_sampler(
        lambda points: sample_at_enu(heightmap, points, hex_frame.local_enu()),
        water_polygons, water_levels, minimum_width=detail.shoreline_pixels * native_spacing_m,
    )

    report("land")
    # Route width/height, colours, and plinth settings never alter this surface.
    # Smoothing is tied to physical size, which is part of the cache identity.
    land = cached_terrain(
        deps.tile_cache.root / "_derived" / "terrain_meshes",
        identity={
            "source": settings.source.sha256,
            "framing": settings.framing.model_dump(),
            "water": settings.biomes.water.enabled,
            "zoom": heightmap.zoom,
            "exaggeration": max(exaggeration, PREVIEW_MAX_EXAGGERATION),
            "tolerance": surface_tolerance,
            "maximum_source_detail": source_detail,
            "sampling": detail.sample_mm,
            "detail": detail.model_dump(),
            "surface_size_mm": settings.physical.size_mm,
        },
        quality_size_mm=settings.physical.size_mm,
        checkpoint=lambda: report("land"),
        build=lambda: build_land_mesh(
            hex_frame, heightmap, water_polygons, base_z=land_base_z, water_levels=water_levels,
            grid_points_per_side=math.ceil(model_world_diameter_m / native_spacing_m) if source_detail else 24,
            native_source_grid=source_detail,
            max_vertices=detail.max_vertices, max_reference_points=detail.max_reference_points, max_passes=detail.max_passes,
            surface_tolerance_m=surface_tolerance,
            sample_spacing_m=sample_spacing,
            checkpoint=lambda: report("land"),
            surface_sampler=surface,
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
            surface_tolerance_m=detail.source_route_tolerance_m if source_detail else detail.route_tolerance_mm / mm_per_m,
            max_points=detail.max_route_points,
            sample_spacing_m=sample_spacing,
            surface_sampler=surface,
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
    kit = _resolve_style(settings.style.name).apply(scene, settings)
    infrastructure_coverage = shapely.Polygon()
    if settings.biomes.roads.enabled or settings.biomes.buildings.enabled:
        roads, candidates, bridges = fetch_infrastructure(hex_frame, deps.http_client, deps.tile_cache, deps.mapbox_token, mm_per_m)
        water_region_mm = scale_polygon(shapely.union_all(water_polygons), mm_per_m, mm_per_m, origin=(0, 0))
        building_coverage = shapely.Polygon()
        if settings.biomes.buildings.enabled:
            exclusion = water_region_mm
            if route_mesh is not None:
                projected = shapely.polygons(route_mesh.triangles[:, :, :2])
                exclusion = shapely.union_all([exclusion, shapely.union_all(projected[shapely.area(projected) > 1e-10]).buffer(.15)])
            buildings, building_coverage = build_buildings(land, candidates, exclusion)
            if buildings is not None:
                kit.parts.append(KitPart(name='buildings', mesh=buildings, material=Material(colour=settings.style.colours.buildings)))
        infrastructure_coverage = building_coverage
        if settings.biomes.roads.enabled:
            # Ground roads stop at water; only explicitly mapped bridges cross it.
            roads = shapely.union_all([roads.difference(water_region_mm), bridges]).difference(building_coverage)
            land, road_part = split_surface_material(land, roads.difference(water_region_mm), preserve_network=True)
            kit.part('land').mesh = land
            road_parts = [road_part] if road_part is not None else []
            if water is not None and not bridges.is_empty:
                water, crossing = split_surface_material(
                    water, bridges.difference(building_coverage), preserve_network=True)
                kit.part('water').mesh = water
                if crossing is not None:
                    road_parts.append(crossing)
            if road_parts:
                road_mesh = (trimesh.boolean.union(road_parts, engine='manifold')
                             if len(road_parts) > 1 else road_parts[0])
                kit.parts.append(KitPart(name='roads', mesh=road_mesh, material=Material(colour=settings.style.colours.roads)))
            infrastructure_coverage = shapely.union_all([infrastructure_coverage, roads])
    snow_coverage = shapely.Polygon()
    if printable_landcover and settings.biomes.snow.enabled:
        snow_coverage = snow_region(land, settings.biomes.snow.snowline, exaggeration).difference(infrastructure_coverage)
        land, snow = split_surface_material(land, snow_coverage)
        kit.part('land').mesh = land
        if snow is not None:
            kit.parts.append(KitPart(name='snow', mesh=snow, material=Material(colour=settings.style.colours.snow)))
    if printable_landcover and (settings.biomes.woodland.enabled or settings.biomes.rock.enabled):
        regions = fetch_landcover_regions(hex_frame, deps.http_client, deps.tile_cache, deps.mapbox_token)
        water_region = shapely.union_all(water_polygons)
        remaining = land
        for name, source, enabled, colour in (
            ('rock', 'rock', settings.biomes.rock.enabled, settings.style.colours.rock),
            ('woodland', 'wood', settings.biomes.woodland.enabled, settings.style.colours.woodland),
        ):
            if not enabled:
                continue
            coverage = regions[source].difference(water_region)
            region = scale_polygon(coverage, xfact=mm_per_m, yfact=mm_per_m, origin=(0, 0))
            region = region.difference(shapely.union_all([snow_coverage, infrastructure_coverage]))
            remaining, insert = split_surface_material(remaining, region)
            if insert is not None:
                kit.parts.append(KitPart(name=name, mesh=insert, material=Material(colour=colour)))
        kit.part('land').mesh = remaining
    return kit


def _resolve_style(name: str) -> Style:
    if name == "monochrome-biome":
        return MonochromeBiome()
    raise ValueError(f"Unknown style: {name}")
