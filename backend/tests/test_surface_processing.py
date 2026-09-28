"""Bounds and continuity for terrain conditioning."""
import numpy as np
import pytest
from shapely.geometry import box
from contour.heightmap import Heightmap
from contour.surface_processing import smooth_heightmap, shoreline_sampler, SMOOTHING_MAX_MM


def test__smooth_heightmap__reduces_noise_with_bounded_displacement() -> None:
    grid = np.indices((64, 64)).sum(axis=0) % 2 * 2.0 + 100
    heightmap = Heightmap(grid, 15, 0, 0)
    result = smooth_heightmap(heightmap, .1)
    assert np.max(np.abs(result.elevations - grid)) <= SMOOTHING_MAX_MM / .1
    assert np.std(result.elevations) < np.std(grid)
    assert result.elevations.min() >= grid.min()
    assert result.elevations.max() <= grid.max()
    np.testing.assert_array_equal(heightmap.elevations, grid)


def test__smooth_heightmap__preserves_constant_surface() -> None:
    result = smooth_heightmap(Heightmap(np.full((32, 32), 42.0), 15, 0, 0), .1)
    np.testing.assert_allclose(result.elevations, 42)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test__smooth_heightmap__rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match='scale'):
        smooth_heightmap(Heightmap(np.zeros((4, 4)), 15, 0, 0), scale)


def test__shoreline_sampler__flush_contact_and_smooth_transition() -> None:
    sampler = shoreline_sampler(lambda p: np.full(len(p), 12.0), [box(-10, -10, 0, 10)], [10], 20)
    points = np.column_stack((np.array([0, .001, 10, 19.999, 20, 25]), np.zeros(6)))
    values = sampler(points)
    assert values[0] == 10
    assert values[1] - values[0] < 1e-7
    assert values[2] == pytest.approx(11)
    assert values[4] - values[3] < 1e-7
    np.testing.assert_allclose(values[4:], 12)


def test__shoreline_sampler__overlapping_banks_are_order_independent() -> None:
    polygons = [box(-20, -10, -10, 10), box(10, -10, 20, 10)]
    base = lambda p: np.full(len(p), 15.0)
    points = np.array([[-10, 0], [-5, 0], [0, 0], [5, 0], [10, 0]])
    first = shoreline_sampler(base, polygons, [10, 20], 20)(points)
    second = shoreline_sampler(base, polygons[::-1], [20, 10], 20)(points)
    np.testing.assert_allclose(first, second)
    assert first[0] == 10 and first[-1] == 20


def test__shoreline_sampler__empty_water_preserves_surface() -> None:
    sample = lambda p: p[:, 0] + 10
    points = np.array([[0., 0], [1, 0]])
    np.testing.assert_allclose(shoreline_sampler(sample, [], [], 1)(points), sample(points))


def test__shoreline_sampler__overlap_remains_continuous_near_water_edge() -> None:
    polygons = [box(-20, -10, -10, 10), box(10, -10, 20, 10)]
    sampler = shoreline_sampler(lambda p: np.full(len(p), 15.), polygons, [10, 20], 30)
    values = sampler(np.array([[-10., 0], [-10 + .0001, 0]]))
    assert abs(values[0] - values[1]) < 1e-6


@pytest.mark.parametrize('levels,width', [([], 10), ([10], 0)])
def test__shoreline_sampler__rejects_invalid_parameters(levels: list[float], width: float) -> None:
    with pytest.raises(ValueError):
        shoreline_sampler(lambda p: np.ones(len(p)), [box(0, 0, 1, 1)], levels, width)


def test__smooth_heightmap__rejects_nonfinite_elevations() -> None:
    with pytest.raises(ValueError, match='finite'):
        smooth_heightmap(Heightmap(np.full((4,4), np.nan), 15, 0, 0), .1)
