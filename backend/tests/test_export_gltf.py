"""Tests for glTF (.glb) export."""
from __future__ import annotations

import io

import pytest
import trimesh

from contour.export.gltf import _hex_to_rgba, to_glb
from contour.schema.kit import KitPart, Material, MeshKit


def _kit(*part_specs: tuple[str, str]) -> MeshKit:
    return MeshKit(
        parts=[
            KitPart(name=name, mesh=trimesh.creation.box(), material=Material(colour=colour))
            for name, colour in part_specs
        ]
    )


def test_glb_starts_with_magic():
    kit = _kit(("land", "#abcdef"))
    glb = to_glb(kit)
    assert glb.startswith(b"glTF")


def test_glb_round_trips_via_trimesh():
    kit = _kit(("land", "#abcdef"), ("water", "#123456"), ("route", "#ff0000"))
    glb = to_glb(kit)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == 3


def test_glb_contains_geometry_names():
    kit = _kit(("land", "#abcdef"), ("plinth", "#222222"))
    glb = to_glb(kit)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    names = set(scene.geometry.keys())
    # trimesh assigns names from the source nodes; check that land and plinth appear
    assert any("land" in n for n in names)
    assert any("plinth" in n for n in names)


def test_hex_to_rgba_known_values():
    assert _hex_to_rgba("#000000") == [0.0, 0.0, 0.0, 1.0]
    assert _hex_to_rgba("#ffffff") == [1.0, 1.0, 1.0, 1.0]
    assert _hex_to_rgba("#ff8040")[0] == pytest.approx(1.0)
    assert _hex_to_rgba("#ff8040")[1] == pytest.approx(128 / 255)
    assert _hex_to_rgba("#ff8040")[2] == pytest.approx(64 / 255)


def test_hex_to_rgba_rejects_bad_input():
    with pytest.raises(ValueError):
        _hex_to_rgba("not-a-colour")
