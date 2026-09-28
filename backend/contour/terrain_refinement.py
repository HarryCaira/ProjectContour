"""Conforming terrain refinement against a densely sampled reference surface."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely
import triangle

from contour.hex_clip import LandTriangulation
from contour.errors import MeshDetailLimitError

# Temporarily raised for source-detail comparison.
MAX_VERTICES = 1_200_000
MAX_REFERENCE_POINTS = 8_000_000
MAX_REFINEMENT_PASSES = 16


def refine_terrain(
    initial: LandTriangulation,
    sample: Callable[[np.ndarray], np.ndarray],
    domain: shapely.Geometry,
    sample_spacing: float,
    tolerance: float,
    checkpoint: Callable[[], None] | None = None,
    max_vertices: int | None = None, max_reference_points: int | None = None, max_passes: int | None = None,
) -> LandTriangulation:
    """Refine only triangles exceeding a vertical error in world metres.

    Validate against a regular reference grid plus edge/interior probes. The
    grid prevents a narrow peak from hiding between the coarse mesh vertices.
    This controls approximation of the source surface, not DEM accuracy.
    """
    max_vertices = MAX_VERTICES if max_vertices is None else max_vertices
    max_reference_points = MAX_REFERENCE_POINTS if max_reference_points is None else max_reference_points
    max_passes = MAX_REFINEMENT_PASSES if max_passes is None else max_passes
    if sample_spacing <= 0 or tolerance <= 0:
        raise ValueError("Terrain spacing and tolerance must be positive")
    check = checkpoint or (lambda: None)
    check()
    minx, miny, maxx, maxy = domain.bounds
    xs = np.arange(minx + sample_spacing / 2, maxx, sample_spacing)
    ys = np.arange(miny + sample_spacing / 2, maxy, sample_spacing)
    if len(xs) * len(ys) > max_reference_points:
        raise MeshDetailLimitError("reference_points", len(xs) * len(ys), max_reference_points)
    reference = []
    # Chunk to avoid creating several full-size temporary coordinate grids.
    for row in range(0, len(ys), 32):
        check()
        gx, gy = np.meshgrid(xs, ys[row:row + 32])
        points = np.column_stack((gx.ravel(), gy.ravel()))
        points = points[shapely.contains_xy(domain, points[:, 0], points[:, 1])]
        reference.append(np.column_stack((points, sample(points))))
    reference = np.concatenate(reference) if reference else np.empty((0, 3))

    # Index the fixed reference points once, rather than rebuilding a triangle
    # index and locating every reference point on every refinement pass.
    reference_tree = shapely.STRtree(shapely.points(reference[:, :2]))
    previous = None
    tri = initial
    # Validate the result of the last permitted pass before declaring failure.
    for passes in range(max_passes + 1):
        check()
        if len(tri.vertices) > max_vertices:
            raise MeshDetailLimitError("vertices", len(tri.vertices), max_vertices, details={"passes": passes})
        xyz = np.column_stack((tri.vertices, sample(tri.vertices)))
        faces = xyz[tri.triangles]
        errors = np.full(len(faces), np.nan)
        if previous is not None:
            old_vertices, old_triangles, old_errors = previous
            # Triangle normally retains vertex IDs. Fall back to full checks if
            # a triangulator ever moves/reorders them; never reuse by proximity.
            if (len(tri.vertices) >= len(old_vertices)
                    and np.array_equal(tri.vertices[:len(old_vertices)], old_vertices)):
                _, old_ids, new_ids = np.intersect1d(
                    _face_keys(old_triangles), _face_keys(tri.triangles),
                    assume_unique=True, return_indices=True)
                errors[new_ids] = old_errors[old_ids]
        changed = np.isnan(errors)
        if changed.any():
            errors[changed] = _sampled_errors(faces[changed], sample, reference, reference_tree, check)
        if not (errors > tolerance).any() and previous is not None:
            # Independent full validation before accepting a cached-error result.
            errors = _sampled_errors(faces, sample, reference, reference_tree, check)
        previous = (tri.vertices.copy(), tri.triangles.copy(), errors.copy())
        needs_refinement = errors > tolerance
        if not needs_refinement.any():
            return tri
        diagnostics = {
            "passes": passes,
            "vertices": len(tri.vertices),
            "triangles_over_tolerance": int(needs_refinement.sum()),
            "max_error_m": float(errors.max()),
            "tolerance_m": tolerance,
        }
        if len(tri.vertices) >= max_vertices:
            raise MeshDetailLimitError("vertices", len(tri.vertices), max_vertices, details=diagnostics)
        if passes == max_passes:
            raise MeshDetailLimitError("iterations", passes, max_passes, details=diagnostics)
        # Spend each batch on the largest vertical errors. Restrict area
        # refinement to that batch instead of giving Triangle every failing
        # face and letting its internal ordering exhaust the vertex allowance.
        candidates = np.flatnonzero(needs_refinement)
        remaining = max_vertices - len(tri.vertices)
        batch_size = max(1, min(len(candidates), remaining // 8))
        selected = candidates[np.argsort(-errors[candidates], kind="stable")[:batch_size]]
        refine_mask = np.zeros(len(faces), dtype=bool)
        refine_mask[selected] = True
        edges1 = faces[:, 1, :2] - faces[:, 0, :2]
        edges2 = faces[:, 2, :2] - faces[:, 0, :2]
        areas = np.abs(edges1[:, 0] * edges2[:, 1] - edges1[:, 1] * edges2[:, 0]) / 2
        result = triangle.triangulate({
            "vertices": tri.vertices,
            "triangles": tri.triangles,
            "segments": np.asarray(tri.boundary_segments),
            "triangle_max_area": np.where(refine_mask, areas / 4, -1).reshape(-1, 1),
        }, f"rpaS{max_vertices - len(tri.vertices)}")
        if len(result["vertices"]) == len(tri.vertices):
            raise MeshDetailLimitError("stalled", len(tri.vertices), details=diagnostics)
        tri = LandTriangulation(result["vertices"], result["triangles"],
                                [tuple(map(int, segment)) for segment in result["segments"]])


def _face_keys(triangles: np.ndarray) -> np.ndarray:
    """Exact oriented vertex triples, not rounded positions or face ordinals."""
    rows = np.ascontiguousarray(triangles, dtype=np.int64)
    return rows.view(np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<i8')])).ravel()


def _sampled_errors(
    faces: np.ndarray, sample: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray, reference_tree: shapely.STRtree, check: Callable[[], None],
) -> np.ndarray:
    """Original seven probes and all covered grid samples, without relaxing error."""
    errors = np.zeros(len(faces))
    # Mid-edges and interior probes also check cut boundaries and triangles
    # smaller than the reference grid, with no extra network requests.
    for weights in ((.5, .5, 0), (0, .5, .5), (.5, 0, .5), (1/3, 1/3, 1/3),
                    (.6, .2, .2), (.2, .6, .2), (.2, .2, .6)):
        check()
        probes = np.einsum('ijk,j->ik', faces, weights)
        errors = np.maximum(errors, np.abs(sample(probes[:, :2]) - probes[:, 2]))
    for offset in range(0, len(faces), 8192):
        check()
        polygons = shapely.polygons(faces[offset:offset + 8192, :, :2])
        ti, pi = reference_tree.query(polygons, predicate="covers")
        ti = ti + offset
        if not len(pi):
            continue
        a, b, c = faces[ti, 0], faces[ti, 1], faces[ti, 2]
        ab, ac, ap = b - a, c - a, reference[pi] - a
        determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        u = (ap[:, 0] * ac[:, 1] - ap[:, 1] * ac[:, 0]) / determinant
        v = (ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / determinant
        predicted = a[:, 2] + u * ab[:, 2] + v * ac[:, 2]
        np.maximum.at(errors, ti, np.abs(reference[pi, 2] - predicted))
    return errors
