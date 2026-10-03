"""Tests for STL kit (zip) export."""
from __future__ import annotations

import io
import json
import zipfile

import trimesh

from contour.stl_export import to_stl_zip
from contour.kit import KitPart, Material, MeshKit


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
    assert manifest["units"] == "mm"
    assert {p["name"] for p in manifest["parts"]} == {"land", "water"}
    land = next(p for p in manifest["parts"] if p["name"] == "land")
    assert land["material"]["colour"] == "#abcdef"
    assert "vertices" in land["stats"]
    assert "faces" in land["stats"]
    assert land["stats"]["volume_mm3"] > 0


def test_stl_files_are_loadable_as_meshes():
    z = to_stl_zip(_kit(("land", "#abcdef")))
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        stl = zf.read("land.stl")
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight  # box is watertight


def test_export_rejects_open_part_with_specific_diagnostics():
    import pytest
    from contour.errors import ContourError
    kit = _kit(("land", "#abcdef"))
    kit.parts[0].mesh.update_faces([True] * 11 + [False])
    with pytest.raises(ContourError) as error:
        to_stl_zip(kit)
    assert error.value.status_code == 422
    assert error.value.details["part"] == "land"
    assert error.value.details["boundary_edges"] == 3


def test_export_rejects_geometry_that_collapses_only_at_stl_precision():
    import numpy as np
    import pytest
    from contour.errors import ContourError
    kit = _kit(("rock", "#abcdef"))
    mesh = trimesh.creation.box(extents=[1e-7, 1, 1])
    mesh.apply_translation([50, 0, 0])
    assert mesh.is_volume
    assert len(np.unique(mesh.vertices[:, 0])) == 2
    kit.parts[0].mesh = mesh
    with pytest.raises(ContourError) as error:
        to_stl_zip(kit)
    assert error.value.details["part"] == "rock"
    assert error.value.details["degenerate_faces"] > 0


def test_export_separates_touching_shells_without_losing_detail():
    import numpy as np
    import pytest
    kit = _kit(("woodland", "#abcdef"))
    a = trimesh.creation.box()
    b = a.copy()
    b.apply_translation([1, 1, 0])
    original = trimesh.util.concatenate([a, b])
    kit.parts[0].mesh = original
    assert original.is_volume
    with zipfile.ZipFile(io.BytesIO(to_stl_zip(kit))) as archive:
        repaired = trimesh.load(io.BytesIO(archive.read('woodland.stl')), file_type='stl')
    assert repaired.is_volume
    assert np.all(np.bincount(repaired.edges_unique_inverse) == 2)
    assert repaired.nondegenerate_faces(height=1e-8).all()
    assert repaired.volume == pytest.approx(original.volume, rel=1e-4)
    assert len(repaired.faces) == len(original.faces)
    from scipy.spatial import cKDTree
    assert cKDTree(original.vertices).query(repaired.vertices)[0].max() < .002


def test_export_separates_near_zero_coordinates_that_weld_but_are_not_equal():
    import numpy as np
    left=trimesh.creation.box();left.apply_translation([-.5,-.5,0])
    right=trimesh.creation.box();right.apply_translation([.5+2e-9,.5+2e-9,0])
    original=trimesh.util.concatenate([left,right])
    # Distinct even in float32; both round to the reader's 1e-8 welding grid.
    assert np.float32(2e-9) != np.float32(0)
    kit=MeshKit([KitPart('land',original,Material('#abcdef'))])
    with zipfile.ZipFile(io.BytesIO(to_stl_zip(kit))) as archive:
        repaired=trimesh.load(io.BytesIO(archive.read('land.stl')),file_type='stl')
    assert repaired.is_volume
    assert np.all(np.bincount(repaired.edges_unique_inverse)==2)
    assert repaired.nondegenerate_faces(height=1e-8).all()


def test_zero_area_cleanup_removes_only_whole_closed_shells():
    import numpy as np
    from contour.stl_repair import _remove_zero_area_shells
    collapsed=trimesh.Trimesh(vertices=[[5,0,0],[5,0,1],[5,0,2],[5,0,3]],
        faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]],process=False)
    box=trimesh.creation.box()
    combined=trimesh.util.concatenate([box,collapsed])
    before=combined.vertices.copy()
    _remove_zero_area_shells(combined)
    assert len(combined.faces)==len(box.faces)
    assert combined.is_volume
    np.testing.assert_array_equal(combined.vertices,before)
    # A collapsed face belonging to an open patch must never be removed.
    collapsed.update_faces([True,True,True,False])
    _remove_zero_area_shells(collapsed)
    assert len(collapsed.faces)==3
