"""Tests for STL kit (zip) export."""
from __future__ import annotations

import io
import json
import zipfile

import trimesh

from contour.export.stl import to_stl_zip
from contour.schema.kit import KitPart, Material, MeshKit


def _kit(*part_specs: tuple[str, str]) -> MeshKit:
    return MeshKit(
        parts=[
            KitPart(name=name, mesh=trimesh.creation.box(), material=Material(colour=colour))
            for name, colour in part_specs
        ]
    )


def test_zip_starts_with_pk_magic():
    z = to_stl_zip(_kit(("land", "#abcdef")))
    assert z[:2] == b"PK"


def test_zip_contains_expected_files():
    z = to_stl_zip(_kit(("land", "#abcdef"), ("water", "#112233"), ("plinth", "#222222")))
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        names = set(zf.namelist())
    assert names == {"land.stl", "water.stl", "plinth.stl", "manifest.json"}


def test_manifest_has_expected_shape():
    z = to_stl_zip(_kit(("land", "#abcdef"), ("water", "#112233")))
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert {p["name"] for p in manifest["parts"]} == {"land", "water"}
    land = next(p for p in manifest["parts"] if p["name"] == "land")
    assert land["material"]["colour"] == "#abcdef"
    assert "vertices" in land["stats"]
    assert "faces" in land["stats"]
    assert land["stats"]["volume_m3"] > 0


def test_stl_files_are_loadable_as_meshes():
    z = to_stl_zip(_kit(("land", "#abcdef")))
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        stl = zf.read("land.stl")
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight  # box is watertight
