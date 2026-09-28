"""Settings schema. Single source of truth for a model. Versioned, serialisable."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    type: Literal["gpx"] = "gpx"
    id: str
    sha256: str


class Framing(BaseModel):
    shape: Literal["hex"] = "hex"
    padding_ratio: float = Field(0.15, ge=0.0, le=1.0, alias="paddingRatio")

    model_config = ConfigDict(populate_by_name=True)


class Physical(BaseModel):
    size_mm: float = Field(100.0, gt=0, alias="sizeMm")
    resolution_mm: float = Field(0.2, gt=0, alias="resolutionMm")

    model_config = ConfigDict(populate_by_name=True)


class ModelColours(BaseModel):
    terrain: str = Field("#7a8060", pattern=r"^#[0-9a-fA-F]{6}$")
    water: str = Field("#6a8aa0", pattern=r"^#[0-9a-fA-F]{6}$")
    route: str = Field("#c44545", pattern=r"^#[0-9a-fA-F]{6}$")


class StyleRef(BaseModel):
    name: Literal["monochrome-biome"] = "monochrome-biome"
    colours: ModelColours = Field(default_factory=ModelColours)


class DetailSettings(BaseModel):
    force_source_zoom: bool = Field(False, alias="forceSourceZoom")
    sample_mm: float = Field(0.1, ge=0.01, le=2, alias="sampleMm")
    tolerance_mm: float = Field(0.0125, ge=0.001, le=0.5, alias="toleranceMm")
    max_zoom: int = Field(15, ge=1, le=15, alias="maxZoom")
    water_zoom: int = Field(14, ge=1, le=16, alias="waterZoom")
    smoothing_sigma: float = Field(2.0, ge=0, le=8, alias="smoothingSigma")
    smoothing_max_mm: float = Field(0.05, ge=0, le=1, alias="smoothingMaxMm")
    shoreline_pixels: float = Field(8.0, ge=0.1, le=32, alias="shorelinePixels")
    route_tolerance_mm: float = Field(0.0125, ge=0.001, le=0.5, alias="routeToleranceMm")
    source_route_tolerance_m: float = Field(0.025, ge=0.001, le=5, alias="sourceRouteToleranceM")
    max_vertices: int = Field(1200000, ge=1000, le=5000000, alias="maxVertices")
    max_reference_points: int = Field(8000000, ge=1000, le=32000000, alias="maxReferencePoints")
    max_passes: int = Field(16, ge=1, le=64, alias="maxPasses")
    max_tiles: int = Field(1024, ge=1, le=4096, alias="maxTiles")
    max_route_points: int = Field(200000, ge=1000, le=1000000, alias="maxRoutePoints")

    model_config = ConfigDict(populate_by_name=True)


class TerrainSettings(BaseModel):
    detail: DetailSettings | None = None
    maximum_source_detail: bool = Field(False, alias="maximumSourceDetail")
    vertical_exaggeration: float = Field(1.5, gt=0, alias="verticalExaggeration")

    model_config = ConfigDict(populate_by_name=True)


class WaterBiome(BaseModel):
    enabled: bool = True
    depth_fraction: float = Field(0.07, ge=0.0, le=0.5, alias="depthFraction")

    model_config = ConfigDict(populate_by_name=True)


class Biomes(BaseModel):
    water: WaterBiome = Field(default_factory=WaterBiome)


class RouteSettings(BaseModel):
    enabled: bool = True
    width_mm: float = Field(1.0, gt=0, alias="widthMm")
    height_above_terrain_mm: float = Field(1.0, ge=0, alias="heightAboveTerrainMm")

    model_config = ConfigDict(populate_by_name=True)


class Plinth(BaseModel):
    enabled: bool = True
    style: Literal["default"] = "default"


class Settings(BaseModel):
    """A complete description of a model. The renderer accepts this and produces a MeshKit."""

    schema_version: Literal[1] = Field(1, alias="schemaVersion")
    source: Source
    framing: Framing = Field(default_factory=Framing)
    physical: Physical = Field(default_factory=Physical)
    style: StyleRef = Field(default_factory=StyleRef)
    terrain: TerrainSettings = Field(default_factory=TerrainSettings)
    biomes: Biomes = Field(default_factory=Biomes)
    route: RouteSettings = Field(default_factory=RouteSettings)
    plinth: Plinth = Field(default_factory=Plinth)

    model_config = ConfigDict(populate_by_name=True)
