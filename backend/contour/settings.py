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
    rotation_degrees: float = Field(0.0, ge=0.0, lt=60.0, alias="rotationDegrees")

    model_config = ConfigDict(populate_by_name=True)


class Physical(BaseModel):
    size_mm: float = Field(150.0, gt=0, alias="sizeMm")
    resolution_mm: float = Field(0.2, gt=0, alias="resolutionMm")

    model_config = ConfigDict(populate_by_name=True)


class StyleRef(BaseModel):
    name: Literal["monochrome-biome"] = "monochrome-biome"


class TerrainSettings(BaseModel):
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
    width_mm: float = Field(2.0, gt=0, alias="widthMm")
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
