"""Tests for the MonochromeBiome style strategy."""
from __future__ import annotations

import io
import json
import zipfile

import trimesh

from contour.stl_export import to_stl_zip

from contour.settings import Settings
from contour.style import NeutralScene
from contour.monochrome_biome import MonochromeBiome


def _settings() -> Settings:
    return Settings.model_validate(
        {"schemaVersion": 1, "source": {"type": "gpx", "id": "x", "sha256": "a" * 64}}
    )


def _box() -> trimesh.Trimesh:
    return trimesh.creation.box()


def test_apply_with_all_parts_produces_kit_with_four_parts():
    scene = NeutralScene(land=_box(), water=_box(), route=_box(), plinth=_box())
    kit = MonochromeBiome().apply(scene, _settings())
    names = [p.name for p in kit.parts]
    assert names == ["land", "water", "route", "plinth"]


def test_apply_with_only_land_omits_others():
    scene = NeutralScene(land=_box())
    kit = MonochromeBiome().apply(scene, _settings())
    assert [p.name for p in kit.parts] == ["land"]


def test_apply_with_no_route_omits_route():
    scene = NeutralScene(land=_box(), water=_box(), plinth=_box())
    kit = MonochromeBiome().apply(scene, _settings())
    assert {p.name for p in kit.parts} == {"land", "water", "plinth"}


def test_materials_use_expected_colours():
    scene = NeutralScene(land=_box(), water=_box(), route=_box(), plinth=_box())
    kit = MonochromeBiome().apply(scene, _settings())
    by_name = {p.name: p for p in kit.parts}
    assert by_name["land"].material.colour == MonochromeBiome.LAND.colour
    assert by_name["water"].material.colour == MonochromeBiome.WATER.colour
    assert by_name["route"].material.colour == MonochromeBiome.ROUTE.colour
    assert by_name["plinth"].material.colour == MonochromeBiome.PLINTH.colour


def test_custom_palette_preserves_defaults_and_is_exported():
    settings = _settings()
    settings.style.colours.terrain = "#123456"
    settings.style.colours.water = "#abcdef"
    settings.style.colours.route = "#fedcba"
    scene = NeutralScene(land=_box(), water=_box(), route=_box(), plinth=_box())
    kit = MonochromeBiome().apply(scene, settings)
    with zipfile.ZipFile(io.BytesIO(to_stl_zip(kit))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    colours = {p["name"]: p["material"]["colour"] for p in manifest["parts"]}
    assert colours == {"land": "#123456", "water": "#abcdef", "route": "#fedcba", "plinth": "#2a2a2a"}
    assert MonochromeBiome().apply(scene, _settings()).part("land").material.colour == "#7a8060"
