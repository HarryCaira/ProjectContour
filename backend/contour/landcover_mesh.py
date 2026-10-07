"""Partition a finished terrain solid into shallow printable material regions."""
from __future__ import annotations

import shapely
import numpy as np
import manifold3d as md
from shapely.geometry.polygon import orient
import trimesh
from shapely.geometry import Polygon

from contour.biome_data import _flatten_polygons

THICKNESS_MM = 0.6
MIN_FEATURE_MM = 0.4
MIN_FOUNDATION_MM = 0.3
# One nanometre in print space: far below printable detail, but sufficient to
# collapse near-collinear clipping remnants before solid triangulation.
SURFACE_PRECISION_MM = 1e-6


def printable_coverage(region: shapely.Geometry, min_feature_mm: float = MIN_FEATURE_MM) -> list[Polygon]:
    """Remove sub-print-width strips and tiny islands in physical millimetres."""
    region = shapely.make_valid(region).simplify(0.025, preserve_topology=True)
    region = region.buffer(-min_feature_mm / 2).buffer(min_feature_mm / 2).intersection(region)
    return [p for p in _flatten_polygons(region) if p.area >= min_feature_mm ** 2]


def split_surface_material(land: trimesh.Trimesh, region: shapely.Geometry, *, min_feature_mm: float = MIN_FEATURE_MM, preserve_network: bool = False) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None]:
    """Use the exact terrain surface, with a fixed vertical 0.6 mm insert depth.

    Boolean partitioning preserves terrain detail and complementary interfaces.
    A 0.3 mm floor protects the terrain foundation on unusually shallow models.
    Inputs are already exaggerated and scaled to final print millimetres.
    """
    # Road widths are established from centreline buffers. Erosion or outline
    # simplification here would sever otherwise printable bends and junctions.
    if preserve_network:
        region = shapely.set_precision(shapely.make_valid(region), SURFACE_PRECISION_MM)
    polygons = (_flatten_polygons(region) if preserve_network
                else printable_coverage(region, min_feature_mm))
    floor = float(land.bounds[0, 2]) + MIN_FOUNDATION_MM
    ceiling = float(land.bounds[1, 2]) + THICKNESS_MM
    if not polygons or floor >= float(land.bounds[1, 2]):
        return land, None
    masks = []
    for polygon in polygons:
        # Boolean clipping can leave numerical dust that has no resolvable
        # surface (e.g. a 6e-17 mm² triangle at a building boundary).
        # This is an area-only precision guard, not road-width erosion.
        if polygon.area <= 1e-12:
            continue
        polygon = orient(polygon, sign=1.0)
        rings = [np.asarray(ring.coords)[:-1, :2] for ring in (polygon.exterior, *polygon.interiors)]
        solid = md.CrossSection(rings).extrude(ceiling - floor).translate((0, 0, floor))
        if solid.status() != md.Error.NoError:
            raise ValueError('Surface mask construction failed')
        if solid.is_empty():
            continue
        data = solid.to_mesh64()
        prism = trimesh.Trimesh(vertices=np.asarray(data.vert_properties)[:, :3],
                                faces=np.asarray(data.tri_verts), process=False)
        masks.append(prism)
    if not masks:
        return land, None
    mask = trimesh.util.concatenate(masks)
    lowered = land.copy()
    lowered.apply_translation([0, 0, -THICKNESS_MM])
    covered = trimesh.boolean.intersection([land, mask], engine='manifold')
    if covered.is_empty:
        return land, None
    insert = trimesh.boolean.difference([covered, lowered], engine='manifold')
    if insert.is_empty:
        return land, None
    remainder = trimesh.boolean.difference([land, insert], engine='manifold')
    for mesh in (remainder, insert):
        if not mesh.is_volume:
            raise ValueError('Land-cover partition did not produce a closed positive solid')
    return remainder, insert
