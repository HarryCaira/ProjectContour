"""Partition a finished terrain solid into shallow printable material regions."""
from __future__ import annotations

import shapely
import trimesh
from shapely.geometry import Polygon

from contour.biome_data import _flatten_polygons

THICKNESS_MM = 0.6
MIN_FEATURE_MM = 0.4
MIN_FOUNDATION_MM = 0.3


def printable_coverage(region: shapely.Geometry, min_feature_mm: float = MIN_FEATURE_MM) -> list[Polygon]:
    """Remove sub-print-width strips and tiny islands in physical millimetres."""
    region = shapely.make_valid(region).simplify(0.025, preserve_topology=True)
    region = region.buffer(-min_feature_mm / 2).buffer(min_feature_mm / 2).intersection(region)
    return [p for p in _flatten_polygons(region) if p.area >= min_feature_mm ** 2]


def split_surface_material(land: trimesh.Trimesh, region: shapely.Geometry, *, min_feature_mm: float = MIN_FEATURE_MM) -> tuple[trimesh.Trimesh, trimesh.Trimesh | None]:
    """Use the exact terrain surface, with a fixed vertical 0.6 mm insert depth.

    Boolean partitioning preserves terrain detail and complementary interfaces.
    A 0.3 mm floor protects the terrain foundation on unusually shallow models.
    Inputs are already exaggerated and scaled to final print millimetres.
    """
    polygons = printable_coverage(region, min_feature_mm)
    floor = float(land.bounds[0, 2]) + MIN_FOUNDATION_MM
    ceiling = float(land.bounds[1, 2]) + THICKNESS_MM
    if not polygons or floor >= float(land.bounds[1, 2]):
        return land, None
    masks = []
    for polygon in polygons:
        prism = trimesh.creation.extrude_polygon(polygon, ceiling - floor, engine='triangle')
        prism.apply_translation([0, 0, floor])
        masks.append(prism)
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
