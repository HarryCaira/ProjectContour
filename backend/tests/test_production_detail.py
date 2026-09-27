"""Physical-detail regression tests independent of remote terrain providers."""
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from contour.hex_clip import triangulate_land
from contour.terrain_refinement import refine_terrain
from contour.route_detail import resample_route, simplify_route
from contour.terrain_data import select_zoom
from contour.hex_frame import HexFrame
from contour.settings import Physical


def _error_at_points(tri, sample, points):
    faces = np.column_stack((tri.vertices, sample(tri.vertices)))[tri.triangles]
    tree = shapely.STRtree(shapely.polygons(faces[:, :, :2]))
    pi, ti = tree.query(shapely.points(points), predicate="covered_by")
    assert len(np.unique(pi)) == len(points)
    a, b, c = faces[ti, 0], faces[ti, 1], faces[ti, 2]
    ab, ac = b - a, c - a
    delta = points[pi] - a[:, :2]
    determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    u = (delta[:, 0] * ac[:, 1] - delta[:, 1] * ac[:, 0]) / determinant
    v = (ab[:, 0] * delta[:, 1] - ab[:, 1] * delta[:, 0]) / determinant
    return np.max(np.abs(sample(points[pi]) - (a[:, 2] + u * ab[:, 2] + v * ac[:, 2])))


def test_refinement_adds_detail_and_meets_error_on_independent_samples():
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=6)
    def sample(p):
        return 2 * np.sin(p[:, 0]) * np.cos(p[:, 1] / 2)
    refined = refine_terrain(initial, sample, domain, .1, .01)
    assert len(refined.vertices) > len(initial.vertices)
    points = np.random.default_rng(17).uniform(.01, 9.99, (10_000, 2))
    assert _error_at_points(refined, sample, points) < .012


def test_flat_surface_keeps_coarse_mesh():
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=6)
    sample = lambda p: p[:, 0] * .4 + p[:, 1] * .2
    refined = refine_terrain(initial, sample, domain, .1, .01)
    assert len(refined.vertices) == len(initial.vertices)


def test_reference_grid_catches_peak_between_coarse_vertices():
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=4)
    sample = lambda p: np.exp(-np.sum((p - [3.3, 4.3]) ** 2, axis=1) / .08)
    refined = refine_terrain(initial, sample, domain, .05, .01)
    assert sample(refined.vertices).max() > .98
    assert len(refined.vertices) > len(initial.vertices)


def test_route_sampling_preserves_bends_but_simplifies_straights():
    points = np.array([[0., 0], [4, 0], [4, 4], [8, 4]])
    dense = resample_route(points, .1)
    assert np.linalg.norm(np.diff(dense, axis=0), axis=1).max() <= .100001
    xyz = np.column_stack((dense, np.zeros(len(dense))))
    kept = simplify_route(xyz, .025)
    assert np.allclose(dense[kept], points)


def test_route_vertical_error_preserves_a_hill():
    x = np.linspace(0, 10, 1001)
    points = np.column_stack((x, np.zeros_like(x), np.sin(x)))
    kept = simplify_route(points, .005)
    assert 4 < len(kept) < 200
    reconstructed = np.interp(x, x[kept], points[kept, 2])
    assert np.max(np.abs(reconstructed - points[:, 2])) < .008


def test_zoom_is_fixed_by_production_profile_and_capped_at_native_detail():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=200)
    assert select_zoom(frame, Physical(size_mm=150, resolution_mm=.01)) == 15
    assert select_zoom(frame, Physical(size_mm=150, resolution_mm=5)) == 15


def test_refinement_never_silently_returns_underresolved_mesh(monkeypatch):
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=4)
    monkeypatch.setattr("contour.terrain_refinement.MAX_VERTICES", 1)
    with pytest.raises(ValueError, match="mesh budget"):
        refine_terrain(initial, lambda p: p[:, 0] ** 2, domain, .1, .01)


def test_obsolete_build_can_cancel_during_reference_sampling():
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=4)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("superseded")
    with pytest.raises(InterruptedError, match="superseded"):
        refine_terrain(initial, lambda p: p[:, 0], domain, .1, .01, checkpoint=cancel)
    assert calls == 3
