import numpy as np
import trimesh
from contour.kit import MeshKit, KitPart, Material
import contour.preview_mesh as preview


def test_large_display_reduction_keeps_print_source_and_route_attributes(monkeypatch):
    monkeypatch.setattr(preview, 'PREVIEW_TRIGGER_FACES', 0)
    land=trimesh.creation.icosphere(subdivisions=7,radius=10)
    vertices,faces=land.vertices.copy(),land.faces.copy()
    route=trimesh.creation.box()
    route.vertex_attributes['_route_offset']=np.ones((len(route.vertices),3))
    kit=MeshKit([KitPart('land',land,Material('#112233')),KitPart('route',route,Material('#445566'))])
    result=preview.display_kit(kit)
    reduced=result.part('land').mesh
    assert reduced.is_volume
    assert len(reduced.faces)<len(land.faces)//2
    assert np.max(np.abs(np.linalg.norm(reduced.vertices,axis=1)-10)) < .015
    np.testing.assert_array_equal(land.vertices,vertices)
    np.testing.assert_array_equal(land.faces,faces)
    assert result.part('route').mesh is route
    assert result.part('land').material is kit.part('land').material


def test_small_display_is_unchanged():
    kit=MeshKit([KitPart('land',trimesh.creation.box(),Material('#112233'))])
    assert preview.display_kit(kit) is kit
