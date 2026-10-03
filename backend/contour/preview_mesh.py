"""Bounded-error display meshes; never used for printable exports."""
import numpy as np
import manifold3d
import trimesh
from contour.kit import MeshKit, KitPart

PREVIEW_TRIGGER_FACES = 1_000_000
PREVIEW_TOLERANCE_MM = 0.01


def display_kit(kit: MeshKit) -> MeshKit:
    if sum(len(part.mesh.faces) for part in kit.parts) <= PREVIEW_TRIGGER_FACES:
        return kit
    parts = []
    for part in kit.parts:
        mesh = part.mesh
        # Preserve route/building attributes for interactive height/width edits.
        if part.name in {'land', 'water', 'roads'} and len(mesh.faces) > 100_000:
            solid = manifold3d.Manifold(manifold3d.Mesh64(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.uint64)))
            if solid.status() == manifold3d.Error.NoError:
                reduced = solid.simplify(PREVIEW_TOLERANCE_MM).to_mesh64()
                if 0 < len(reduced.tri_verts) < len(mesh.faces):
                    candidate = trimesh.Trimesh(vertices=np.asarray(reduced.vert_properties)[:, :3],
                                                faces=np.asarray(reduced.tri_verts), process=False)
                    if candidate.is_volume:
                        mesh = candidate
        parts.append(KitPart(name=part.name, mesh=mesh, material=part.material,
                             exportable_as_separate_part=part.exportable_as_separate_part))
    return MeshKit(parts=parts)
