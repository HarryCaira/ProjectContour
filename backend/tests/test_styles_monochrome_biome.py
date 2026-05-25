"""Tests for the MonochromeBiome style strategy."""
from __future__ import annotations

import trimesh

from contour.schema.settings import Settings
from contour.styles.base import NeutralScene
from contour.styles.monochrome_biome import MonochromeBiome


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
