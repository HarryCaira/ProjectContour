"""Conforming terrain refinement against a densely sampled reference surface."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely
import triangle

from contour.hex_clip import LandTriangulation
from contour.errors import MeshDetailLimitError

MAX_VERTICES = 600_000
MAX_REFERENCE_POINTS = 8_000_000


def refine_terrain(
    initial: LandTriangulation,
    sample: Callable[[np.ndarray], np.ndarray],
    domain: shapely.Geometry,
    sample_spacing: float,
    tolerance: float,
    checkpoint: Callable[[], None] | None = None,
) -> LandTriangulation:
    """Refine only triangles exceeding a vertical error in world metres.

    Validate against a regular reference grid plus edge/interior probes. The
    grid prevents a narrow peak from hiding between the coarse mesh vertices.
    This controls approximation of the source surface, not DEM accuracy.
    """
    if sample_spacing <= 0 or tolerance <= 0:
        raise ValueError("Terrain spacing and tolerance must be positive")
    check = checkpoint or (lambda: None)
    check()
    minx, miny, maxx, maxy = domain.bounds
    xs = np.arange(minx + sample_spacing / 2, maxx, sample_spacing)
    ys = np.arange(miny + sample_spacing / 2, maxy, sample_spacing)
    if len(xs) * len(ys) > MAX_REFERENCE_POINTS:
        raise MeshDetailLimitError()
    reference = []
    # Chunk to avoid creating several full-size temporary coordinate grids.
    for row in range(0, len(ys), 32):
        check()
        gx, gy = np.meshgrid(xs, ys[row:row + 32])
        points = np.column_stack((gx.ravel(), gy.ravel()))
        points = points[shapely.contains_xy(domain, points[:, 0], points[:, 1])]
        reference.append(np.column_stack((points, sample(points))))
    reference = np.concatenate(reference) if reference else np.empty((0, 3))

    tri = initial
    for _ in range(16):
        check()
        if len(tri.vertices) > MAX_VERTICES:
            break
        xyz = np.column_stack((tri.vertices, sample(tri.vertices)))
        faces = xyz[tri.triangles]
        errors = np.zeros(len(faces))
        # Mid-edges and interior probes also check cut boundaries and triangles
        # smaller than the reference grid, with no extra network requests.
        for weights in ((.5, .5, 0), (0, .5, .5), (.5, 0, .5), (1/3, 1/3, 1/3),
                        (.6, .2, .2), (.2, .6, .2), (.2, .2, .6)):
            probes = np.einsum('ijk,j->ik', faces, weights)
            errors = np.maximum(errors, np.abs(sample(probes[:, :2]) - probes[:, 2]))
        tree = shapely.STRtree(shapely.polygons(faces[:, :, :2]))
        for offset in range(0, len(reference), 32_768):
            points = reference[offset:offset + 32_768]
            pi, ti = tree.query(shapely.points(points[:, :2]), predicate="covered_by")
            if not len(pi):
                continue
            a, b, c = faces[ti, 0], faces[ti, 1], faces[ti, 2]
            ab, ac, ap = b - a, c - a, points[pi] - a
            determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            u = (ap[:, 0] * ac[:, 1] - ap[:, 1] * ac[:, 0]) / determinant
            v = (ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / determinant
            predicted = a[:, 2] + u * ab[:, 2] + v * ac[:, 2]
            np.maximum.at(errors, ti, np.abs(points[pi, 2] - predicted))
        needs_refinement = errors > tolerance
        if not needs_refinement.any():
            return tri
        edges1 = faces[:, 1, :2] - faces[:, 0, :2]
        edges2 = faces[:, 2, :2] - faces[:, 0, :2]
        areas = np.abs(edges1[:, 0] * edges2[:, 1] - edges1[:, 1] * edges2[:, 0]) / 2
        result = triangle.triangulate({
            "vertices": tri.vertices,
            "triangles": tri.triangles,
            "segments": np.asarray(tri.boundary_segments),
            "triangle_max_area": np.where(needs_refinement, areas / 4, -1).reshape(-1, 1),
        }, f"rpaS{MAX_VERTICES - len(tri.vertices)}")
        if len(result["vertices"]) == len(tri.vertices):
            break
        tri = LandTriangulation(result["vertices"], result["triangles"],
                                [tuple(map(int, segment)) for segment in result["segments"]])
    # Never silently label a budget-limited mesh as production quality.
    raise MeshDetailLimitError()
