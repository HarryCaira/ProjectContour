"""Validated STL kit serialization."""
from __future__ import annotations
import io
import json
import zipfile
import numpy as np
import trimesh
from contour.errors import ContourError
from contour.kit import MeshKit
from contour.stl_repair import repair_candidates


def to_stl_zip(kit: MeshKit) -> bytes:
    stats = {}
    stls = {part.name: _validated_stl(part.name, part.mesh, stats=stats) for part in kit.parts}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in stls.items():
            zf.writestr(f'{name}.stl', data)
        zf.writestr('manifest.json', json.dumps(_manifest(kit, stats), indent=2))
    return buf.getvalue()


def _manifest(kit: MeshKit, stats: dict | None = None) -> dict:
    return {'units': 'mm', 'attribution': '© OpenStreetMap contributors — https://www.openstreetmap.org/copyright', 'parts': [
        {'name': p.name, 'material': {'colour': p.material.colour,
                                    'roughness': p.material.roughness, 'metalness': p.material.metalness},
         'stats': (stats or {}).get(p.name, {'vertices': int(len(p.mesh.vertices)), 'faces': int(len(p.mesh.faces)),
                   'volume_mm3': float(p.mesh.volume)})} for p in kit.parts]}


def _validated_stl(name: str, mesh: trimesh.Trimesh, *, repair: bool = True, stats: dict | None = None) -> bytes:
    """Validate the float32 triangle soup a slicer receives, after any repair."""
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        raise ContourError('invalid_export_mesh', f'Cannot export {name}: mesh is empty or has non-finite coordinates.',
                           422, {'part': name})
    data = mesh.export(file_type='stl')
    decoded = trimesh.load(io.BytesIO(data), file_type='stl', process=True)
    counts = np.bincount(decoded.edges_unique_inverse)
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    degenerate = int(np.count_nonzero(~decoded.nondegenerate_faces(height=1e-8)))
    duplicates = int(np.count_nonzero(~decoded.unique_faces()))
    if boundary or nonmanifold or degenerate or duplicates or not decoded.is_volume:
        if repair:
            for candidate in repair_candidates(mesh):
                try:
                    return _validated_stl(name, candidate, repair=False, stats=stats)
                except ContourError:
                    continue
        details = {'part': name, 'boundary_edges': boundary, 'non_manifold_edges': nonmanifold,
                   'degenerate_faces': degenerate, 'duplicate_faces': duplicates,
                   'winding_consistent': bool(decoded.is_winding_consistent)}
        message = (f'Cannot export {name}: the STL has {boundary:,} open edges, '
                   f'{nonmanifold:,} non-manifold edges, {degenerate:,} collapsed triangles and '
                   f'{duplicates:,} duplicate triangles. ')
        if not decoded.is_volume:
            message += 'Solid validation failed. '
        raise ContourError('invalid_export_mesh', message + 'The download was blocked because this part needs a mesh-generation repair.', 422, details)
    if stats is not None:
        stats[name] = {'vertices': int(len(decoded.vertices)), 'faces': int(len(decoded.faces)), 'volume_mm3': float(decoded.volume)}
    return data
