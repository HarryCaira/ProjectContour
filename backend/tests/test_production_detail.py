"""Physical-detail regression tests independent of remote terrain providers."""
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from contour.errors import MeshDetailLimitError
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
    with pytest.raises(ValueError, match="vertex limit"):
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


def test__select_zoom__source_detail_uses_native_zoom_independent_of_print_size() -> None:
    frame = HexFrame(0, 0, 5000)
    physical = Physical(size_mm=50)
    assert select_zoom(frame, physical) < 15
    assert select_zoom(frame, physical, maximum_source_detail=True) == 15


def test__select_zoom__source_detail_rejects_tile_budget_instead_of_downsampling() -> None:
    with pytest.raises(ValueError, match='tile budget'):
        select_zoom(HexFrame(0, 0, 100_000), Physical(), maximum_source_detail=True)


@pytest.mark.parametrize('reason', ['vertices', 'reference_points', 'iterations', 'stalled'])
def test__refine_terrain__reports_specific_stop_reason(monkeypatch, reason: str) -> None:
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=4)
    if reason == 'vertices':
        monkeypatch.setattr('contour.terrain_refinement.MAX_VERTICES', len(initial.vertices))
    elif reason == 'reference_points':
        monkeypatch.setattr('contour.terrain_refinement.MAX_REFERENCE_POINTS', 10)
    elif reason == 'iterations':
        monkeypatch.setattr('contour.terrain_refinement.MAX_REFINEMENT_PASSES', 0)
    else:
        monkeypatch.setattr('contour.terrain_refinement.triangle.triangulate', lambda data, opts: data)
    with pytest.raises(MeshDetailLimitError) as caught:
        refine_terrain(initial, lambda p: p[:, 0]**2, domain, 1, .01)
    details = caught.value.details
    assert details['reason'] == reason
    assert details['count'] == (100 if reason == 'reference_points' else 0 if reason == 'iterations' else len(initial.vertices))
    assert 'smaller physical size' not in caught.value.message
    if reason != 'reference_points':
        assert details['max_error_m'] > details['tolerance_m']
        assert details['triangles_over_tolerance'] > 0


def test__refine_terrain__accepts_accuracy_reached_on_final_allowed_pass(monkeypatch) -> None:
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=2)
    dense = triangulate_land(domain, [], grid_points_per_side=40)
    monkeypatch.setattr('contour.terrain_refinement.MAX_REFINEMENT_PASSES', 1)
    monkeypatch.setattr('contour.terrain_refinement.triangle.triangulate', lambda data, opts: {
        'vertices': dense.vertices, 'triangles': dense.triangles, 'segments': dense.boundary_segments,
    })
    result = refine_terrain(initial, lambda p: p[:, 0]**2, domain, .5, .1)
    assert len(result.vertices) == len(dense.vertices)


def test__refine_terrain__prioritises_largest_height_errors_with_small_budget(monkeypatch) -> None:
    domain = shapely.union_all([box(0, 0, 1, 1), box(10, 0, 11, 1)])
    initial = triangulate_land(domain, [], grid_points_per_side=1)
    monkeypatch.setattr('contour.terrain_refinement.MAX_VERTICES', len(initial.vertices) + 8)
    def inspect_batch(data, options):
        selected = np.flatnonzero(data['triangle_max_area'].ravel() > 0)
        assert len(selected) == 1
        centre = data['vertices'][data['triangles'][selected]].mean(axis=1)
        assert centre[0, 0] > 5, 'The highest-error region must get the next batch'
        raise InterruptedError('inspected')
    monkeypatch.setattr('contour.terrain_refinement.triangle.triangulate', inspect_batch)
    def surface(p):
        return (p[:, 0] % 10)**2 * np.where(p[:, 0] > 5, 100, 1)
    with pytest.raises(InterruptedError, match='inspected'):
        refine_terrain(initial, surface, domain, .1, .01)


def test_custom_budgets_are_per_request():
    domain = box(0, 0, 10, 10)
    initial = triangulate_land(domain, [], grid_points_per_side=6)
    sample = lambda points: np.zeros(len(points))
    with pytest.raises(MeshDetailLimitError):
        refine_terrain(initial, sample, domain, .1, .01, max_reference_points=10)
    assert len(refine_terrain(initial, sample, domain, .1, .01).vertices) == len(initial.vertices)
    points = np.array([[0., 0], [10., 0]])
    with pytest.raises(ValueError, match='budget'):
        resample_route(points, .1, max_points=10)
    assert len(resample_route(points, .1)) == 101


def test_indexed_error_checks_match_brute_force_including_shared_edges():
    from contour.terrain_refinement import _sampled_errors
    domain = box(0, 0, 4, 4).difference(box(1, 1, 2, 2))
    tri = triangulate_land(domain, [], grid_points_per_side=6)
    sample = lambda p: np.sin(p[:, 0]) * np.cos(p[:, 1])
    faces = np.column_stack((tri.vertices, sample(tri.vertices)))[tri.triangles]
    # Include vertices, shared-edge midpoints, and points on the hole boundary.
    xy = np.concatenate([tri.vertices, faces[:, :2, :2].mean(axis=1),
                         np.random.default_rng(23).uniform(0, 4, (500, 2))])
    reference = np.column_stack((xy, sample(xy)))
    geometries = shapely.points(xy)
    actual = _sampled_errors(faces, sample, reference, shapely.STRtree(geometries), lambda: None)
    expected = []
    for face in faces:
        probes = np.array([np.array(w) @ face for w in
            ((.5, .5, 0), (0, .5, .5), (.5, 0, .5), (1/3, 1/3, 1/3),
             (.6, .2, .2), (.2, .6, .2), (.2, .2, .6))])
        error = np.max(np.abs(sample(probes[:, :2]) - probes[:, 2]))
        points = reference[shapely.covers(shapely.Polygon(face[:, :2]), geometries)]
        if len(points):
            coefficients = np.linalg.solve(np.column_stack((face[:, :2], np.ones(3))), face[:, 2])
            predicted = np.column_stack((points[:, :2], np.ones(len(points)))) @ coefficients
            error = max(error, np.max(np.abs(points[:, 2] - predicted)))
        expected.append(error)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_incremental_refinement_reuses_checks_but_finishes_with_full_validation(monkeypatch):
    import contour.terrain_refinement as refinement
    counts = []
    original = refinement._sampled_errors
    def inspect(faces, *args):
        counts.append(len(faces))
        return original(faces, *args)
    monkeypatch.setattr(refinement, '_sampled_errors', inspect)
    domain = box(0, 0, 10, 10)
    tri = triangulate_land(domain, [], grid_points_per_side=6)
    sample = lambda p: 2 * np.sin(p[:, 0]) * np.cos(p[:, 1] / 2)
    result = refine_terrain(tri, sample, domain, .2, .02)
    assert counts[-1] == len(result.triangles)
    assert counts[-2] < counts[-1]
    points = np.random.default_rng(11).uniform(.01, 9.99, (10000, 2))
    assert _error_at_points(result, sample, points) < .025
