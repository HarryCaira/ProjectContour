"""End-to-end pipeline tests with mocked Mapbox HTTP."""
from __future__ import annotations

import io

import mapbox_vector_tile
import numpy as np
import pytest
import responses
from PIL import Image
from shapely.geometry import Polygon

from contour.http.cache import TileCache
from contour.http.client import HttpClient
from contour.pipeline import PipelineDependencies, build_kit
from contour.schema.route import Route
from contour.schema.settings import Settings


def _make_png(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _add_terrain_mock(rgb_value: tuple[int, int, int] = (0, 0, 100)) -> None:
    png = _make_png(np.full((256, 256, 3), rgb_value, dtype=np.uint8))
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.terrain-rgb/.*"),
        body=png,
        status=200,
    )


def _add_biomes_mock(water_polygons: list[Polygon] | None = None) -> None:
    layer = {"name": "water", "features": []}
    if water_polygons:
        layer["features"] = [{"geometry": p, "properties": {}} for p in water_polygons]
    encoded = mapbox_vector_tile.encode([layer])
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.mapbox-streets-v8/.*"),
        body=encoded,
        status=200,
    )


def _route() -> Route:
    return Route(
        latitudes=np.array([0.0, 0.0005, 0.001]),
        longitudes=np.array([0.0, 0.0005, 0.001]),
        elevations=np.array([0.0, 0.0, 0.0]),
    )


def _settings(**overrides) -> Settings:
    base = {
        "schemaVersion": 1,
        "source": {"type": "gpx", "id": "x", "sha256": "a" * 64},
        "physical": {"sizeMm": 150, "resolutionMm": 2.0},
    }
    base.update(overrides)
    return Settings.model_validate(base)


def _deps(tmp_path) -> PipelineDependencies:
    return PipelineDependencies(
        http_client=HttpClient(backoff_factor=0.0),
        tile_cache=TileCache(tmp_path),
        mapbox_token="test-token",
    )


@responses.activate
def test_build_kit_produces_land_plinth_route_without_water(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock(water_polygons=None)

    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    names = {p.name for p in kit.parts}
    assert "land" in names
    assert "plinth" in names
    assert "route" in names
    assert "water" not in names


@responses.activate
def test_build_kit_includes_water_when_present(tmp_path):
    _add_terrain_mock()
    # Polygon covering essentially the whole tile so it overlaps the hex
    # regardless of where the hex falls within the tile grid.
    _add_biomes_mock(
        water_polygons=[
            Polygon([(50, 50), (4046, 50), (4046, 4046), (50, 4046)])
        ]
    )

    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    names = {p.name for p in kit.parts}
    assert "water" in names


@responses.activate
def test_build_kit_respects_disabled_route(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    settings = _settings(route={"enabled": False, "widthMm": 2.0, "heightAboveTerrainMm": 1.0})

    kit = build_kit(settings, _route(), _deps(tmp_path))
    assert "route" not in {p.name for p in kit.parts}


@responses.activate
def test_build_kit_respects_disabled_plinth(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    settings = _settings(plinth={"enabled": False, "style": "default"})

    kit = build_kit(settings, _route(), _deps(tmp_path))
    assert "plinth" not in {p.name for p in kit.parts}


@responses.activate
def test_build_kit_meshes_are_watertight(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    for part in kit.parts:
        assert part.mesh.is_watertight, f"{part.name} mesh is not watertight"


@responses.activate
def test_build_kit_unknown_style_raises(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    # We can't pass a non-enum value through Pydantic, so we build the Settings
    # then mutate via model_copy(update=...) bypassing the literal type at runtime.
    settings = _settings()
    object.__setattr__(settings.style, "name", "non-existent-style")
    with pytest.raises(ValueError, match="Unknown style"):
        build_kit(settings, _route(), _deps(tmp_path))
