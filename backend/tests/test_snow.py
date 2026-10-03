import numpy as np
import pytest
import trimesh
from contour.snow import snow_region
from contour.landcover_mesh import split_surface_material
from contour.settings import Settings


def hillside():
    mesh = trimesh.creation.box(extents=[20, 20, 2])
    mesh.vertices[:, 2] += mesh.vertices[:, 0]*.5 + 8
    return mesh


def test_snowline_reduces_coverage_and_ignores_exaggeration():
    land = hillside()
    lower = snow_region(land, .4)
    upper = snow_region(land, .8)
    assert 0 < upper.area < lower.area < 400
    assert upper.difference(lower).area < 1e-8
    exaggerated = land.copy()
    exaggerated.vertices[:, 2] *= 3
    assert snow_region(exaggerated, .4, 3).symmetric_difference(lower).area < 1e-8


def test_snow_coverage_is_scale_invariant():
    from shapely.affinity import scale
    land = hillside()
    original = snow_region(land, .72)
    land.apply_scale(1.5)
    assert snow_region(land, .72).symmetric_difference(scale(original, 1.5, 1.5, origin=(0, 0))).area < 1e-8


def test_flat_ground_has_no_artificial_cap():
    assert snow_region(trimesh.creation.box(), .72).is_empty


def test_snow_export_has_shallow_complementary_solids():
    land = hillside()
    base, snow = split_surface_material(land, snow_region(land, .6))
    assert snow is not None and snow.is_volume and base.is_volume
    assert base.volume + snow.volume == pytest.approx(land.volume, rel=1e-5)
    offsets = snow.vertices[:, 2] - (snow.vertices[:, 0]*.5 + 9)
    assert offsets.min() == pytest.approx(-.6, abs=1e-5)
    assert offsets.max() == pytest.approx(0, abs=1e-5)


def test_snow_settings_are_optional_and_validate_range():
    from pydantic import ValidationError
    settings = Settings.model_validate({'source': {'id': 'a', 'sha256': 'b'}})
    assert not settings.biomes.snow.enabled
    assert settings.style.colours.snow == '#f3f4ef'
    with pytest.raises(ValidationError):
        Settings.model_validate({'source': {'id': 'a', 'sha256': 'b'}, 'biomes': {'snow': {'snowline': 2}}})
