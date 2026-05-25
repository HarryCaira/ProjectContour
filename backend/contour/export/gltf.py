"""glTF (.glb) serialisation of a MeshKit for the frontend viewer."""
from __future__ import annotations

import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from contour.schema.kit import Material, MeshKit


def to_glb(kit: MeshKit) -> bytes:
    """Serialise a MeshKit to binary glTF (.glb) bytes.

    Each KitPart becomes a geometry in the scene with its colour applied as a
    PBR material. The frontend loads the glb via the standard glTF loader and
    can apply whatever final rendering treatment the chosen style needs.
    """
    scene = trimesh.Scene()
    for part in kit.parts:
        mesh = part.mesh.copy()
        mesh.visual = TextureVisuals(material=_pbr_material(part.material, part.name))
        scene.add_geometry(mesh, node_name=part.name, geom_name=part.name)
    return scene.export(file_type="glb")


def _pbr_material(material: Material, name: str) -> PBRMaterial:
    return PBRMaterial(
        name=name,
        baseColorFactor=_hex_to_rgba(material.colour),
        roughnessFactor=material.roughness,
        metallicFactor=material.metalness,
    )


def _hex_to_rgba(hex_str: str) -> list[float]:
    """Convert '#aabbcc' to [r, g, b, 1.0] in 0..1."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex colour, got {hex_str!r}")
    return [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, 1.0]
