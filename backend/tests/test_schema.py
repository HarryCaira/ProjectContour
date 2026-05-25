"""Tests for the Settings schema: validation, defaults, alias handling."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from contour.schema.settings import Settings

VALID_SOURCE = {"type": "gpx", "id": "abc123", "sha256": "deadbeef" * 8}


def test_minimal_settings_uses_defaults():
    s = Settings.model_validate({"schemaVersion": 1, "source": VALID_SOURCE})
    assert s.framing.shape == "hex"
    assert s.framing.padding_ratio == 0.15
    assert s.physical.size_mm == 150.0
    assert s.physical.resolution_mm == 0.2
    assert s.style.name == "monochrome-biome"
    assert s.terrain.vertical_exaggeration == 1.5
    assert s.biomes.water.enabled is True
    assert s.biomes.water.depth_fraction == 0.07
    assert s.route.enabled is True
    assert s.plinth.enabled is True


def test_settings_accepts_camelcase_aliases():
    payload = {
        "schemaVersion": 1,
        "source": VALID_SOURCE,
        "physical": {"sizeMm": 200, "resolutionMm": 0.1},
        "terrain": {"verticalExaggeration": 2.0},
    }
    s = Settings.model_validate(payload)
    assert s.physical.size_mm == 200
    assert s.physical.resolution_mm == 0.1
    assert s.terrain.vertical_exaggeration == 2.0


def test_settings_rejects_unknown_schema_version():
    payload = {"schemaVersion": 2, "source": VALID_SOURCE}
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_settings_rejects_non_positive_size():
    payload = {"schemaVersion": 1, "source": VALID_SOURCE, "physical": {"sizeMm": -1, "resolutionMm": 0.2}}
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_settings_rejects_out_of_range_rotation():
    payload = {"schemaVersion": 1, "source": VALID_SOURCE, "framing": {"rotationDegrees": 60.0}}
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_settings_round_trip_json():
    s = Settings.model_validate({"schemaVersion": 1, "source": VALID_SOURCE})
    payload = s.model_dump(by_alias=True)
    restored = Settings.model_validate(payload)
    assert restored == s
