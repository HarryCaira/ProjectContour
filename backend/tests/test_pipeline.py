"""End-to-end pipeline tests with mocked Mapbox HTTP."""
from __future__ import annotations

import io
import zipfile

import contour.pipeline as pipeline_module

import trimesh

import mapbox_vector_tile
import numpy as np
import pytest
import responses
from PIL import Image
from shapely.geometry import Point, Polygon, box

from contour.tile_cache import TileCache
from contour.http_client import HttpClient
from contour.stl_export import to_stl_zip
from contour.pipeline import PipelineDependencies, build_kit
from contour.route import Route
from contour.settings import Settings


def _make_png(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _add_terrain_mock(rgb_value: tuple[int, int, int] = (0, 0, 100)) -> None:
    png = _make_png(np.full((256, 256, 3), rgb_value, dtype=np.uint8))
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.terrain-rgb/.*"),
        body=png,
        status=200,
    )


def _add_biomes_mock(water_polygons: list[Polygon] | None = None) -> None:
    layer = {"name": "water", "features": []}
    if water_polygons:
        layer["features"] = [{"geometry": p, "properties": {}} for p in water_polygons]
    encoded = mapbox_vector_tile.encode([layer])
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.mapbox-streets-v8/.*"),
        body=encoded,
        status=200,
    )


def _route() -> Route:
    return Route(
        latitudes=np.array([0.0, 0.0005, 0.001]),
        longitudes=np.array([0.0, 0.0005, 0.001]),
        elevations=np.array([0.0, 0.0, 0.0]),
    )


def _settings(**overrides) -> Settings:
    base = {
        "schemaVersion": 1,
        "source": {"type": "gpx", "id": "x", "sha256": "a" * 64},
        "physical": {"sizeMm": 150, "resolutionMm": 2.0},
    }
    base.update(overrides)
    return Settings.model_validate(base)


def _deps(tmp_path) -> PipelineDependencies:
    return PipelineDependencies(
        http_client=HttpClient(backoff_factor=0.0),
        tile_cache=TileCache(tmp_path),
        mapbox_token="test-token",
    )


@responses.activate
def test_build_kit_produces_land_plinth_route_without_water(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock(water_polygons=None)

    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    names = {p.name for p in kit.parts}
    assert "land" in names
    assert "plinth" in names
    assert "route" in names
    assert "water" not in names


@responses.activate
def test_build_kit_includes_water_when_present(tmp_path):
    _add_terrain_mock()
    # Polygon covering essentially the whole tile so it overlaps the hex
    # regardless of where the hex falls within the tile grid.
    _add_biomes_mock(
        water_polygons=[
            Polygon([(50, 50), (4046, 50), (4046, 4046), (50, 4046)])
        ]
    )

    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    names = {p.name for p in kit.parts}
    assert "water" in names


@responses.activate
def test_build_kit_respects_disabled_route(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    settings = _settings(route={"enabled": False, "widthMm": 2.0, "heightAboveTerrainMm": 1.0})

    kit = build_kit(settings, _route(), _deps(tmp_path))
    assert "route" not in {p.name for p in kit.parts}


@responses.activate
def test_build_kit_respects_disabled_plinth(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    settings = _settings(plinth={"enabled": False, "style": "default"})

    kit = build_kit(settings, _route(), _deps(tmp_path))
    assert "plinth" not in {p.name for p in kit.parts}


@responses.activate
def test_build_kit_meshes_are_watertight(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    kit = build_kit(_settings(), _route(), _deps(tmp_path))
    for part in kit.parts:
        assert part.mesh.is_watertight, f"{part.name} mesh is not watertight"


@responses.activate
def test_build_kit_unknown_style_raises(tmp_path):
    _add_terrain_mock()
    _add_biomes_mock()
    # We can't pass a non-enum value through Pydantic, so we build the Settings
    # then mutate via model_copy(update=...) bypassing the literal type at runtime.
    settings = _settings()
    object.__setattr__(settings.style, "name", "non-existent-style")
    with pytest.raises(ValueError, match="Unknown style"):
        build_kit(settings, _route(), _deps(tmp_path))


@responses.activate
@pytest.mark.parametrize("size_mm", [50, 150, 300])
def test__build_kit__physical_units_and_fixed_route_height(tmp_path, size_mm):
    _add_terrain_mock()
    _add_biomes_mock()
    kits = [build_kit(_settings(
        physical={"sizeMm": size_mm, "resolutionMm": 2},
        terrain={"verticalExaggeration": factor},
    ), _route(), _deps(tmp_path)) for factor in [1, 2]]
    for kit in kits:
        land = kit.part("land").mesh
        assert max(land.extents[:2]) == pytest.approx(size_mm)
        with zipfile.ZipFile(io.BytesIO(to_stl_zip(kit))) as archive:
            exported_land = trimesh.load(io.BytesIO(archive.read("land.stl")), file_type="stl")
            exported_route = trimesh.load(io.BytesIO(archive.read("route.stl")), file_type="stl")
        assert max(exported_land.extents[:2]) == pytest.approx(size_mm, abs=1e-4)
        assert np.allclose(exported_route.bounds, kit.part("route").mesh.bounds, atol=1e-4)
        assert kit.part("plinth").mesh.bounds[0, 2] == pytest.approx(0)
        route = kit.part("route").mesh
        xy, groups = np.unique(route.vertices[:, :2], axis=0, return_inverse=True)
        for index in range(len(xy)):
            zs = route.vertices[groups == index, 2]
            assert np.ptp(zs) == pytest.approx(1.0)
    assert np.allclose(kits[0].part("plinth").mesh.bounds, kits[1].part("plinth").mesh.bounds)
    pivot = kits[0].part("plinth").mesh.bounds[1, 2]
    top1 = kits[0].part("land").mesh.bounds[1, 2]
    top2 = kits[1].part("land").mesh.bounds[1, 2]
    assert top2 - pivot == pytest.approx(2 * (top1 - pivot))


@responses.activate
@pytest.mark.parametrize("size_mm,exaggeration", [(50, 0.5), (150, 1), (300, 5)])
@pytest.mark.parametrize("shape", ["lake", "edge", "island", "full"])
def test_water_is_a_surface_insert_with_solid_terrain_below(tmp_path, monkeypatch, size_mm, exaggeration, shape):
    _add_terrain_mock()
    _add_biomes_mock()
    area = {}

    def water_polygons(frame, *_, **kwargs):
        hexagon = frame.polygon_enu()
        r = frame.circumradius_m
        if shape == "full":
            polygon = hexagon
        elif shape == "edge":
            polygon = hexagon.intersection(box(0, -2 * r, 2 * r, 2 * r))
        else:
            polygon = Point(0, 0).buffer(r / 3)
            if shape == "island":
                polygon = polygon.difference(Point(0, 0).buffer(r / 8))
        area["hex"] = hexagon.area
        area["scale"] = size_mm / max(np.ptp(np.array(hexagon.exterior.coords), axis=0))
        return [polygon]

    monkeypatch.setattr("contour.pipeline.fetch_water_polygons", water_polygons)
    kit = build_kit(_settings(
        physical={"sizeMm": size_mm, "resolutionMm": 2},
        terrain={"verticalExaggeration": exaggeration},
    ), _route(), _deps(tmp_path))
    land = kit.part("land").mesh
    water = kit.part("water").mesh
    assert land.is_watertight and water.is_watertight
    assert land.is_volume and water.is_volume
    available_height = water.bounds[1, 2] - land.bounds[0, 2]
    assert water.extents[2] == pytest.approx(min(0.6, available_height / 2), abs=1e-4)
    assert water.bounds[0, 2] > land.bounds[0, 2]
    # On a flat heightmap, together the two materials must completely fill the
    # hexagonal prism: no through-holes and no overlapping material volumes.
    expected = area["hex"] * area["scale"] ** 2 * available_height
    assert land.volume + water.volume == pytest.approx(expected, rel=2e-4)


@responses.activate
def test__build_kit__source_detail_bypasses_smoothing_and_has_separate_cache(tmp_path, monkeypatch):
    _add_terrain_mock()
    _add_biomes_mock()
    original_smooth = pipeline_module.smooth_heightmap
    original_land = pipeline_module.build_land_mesh
    smoothing_calls = []
    tolerances = []
    def smooth(heightmap, scale, **kwargs):
        smoothing_calls.append(scale)
        return original_smooth(heightmap, scale, **kwargs)
    def land(*args, **kwargs):
        tolerances.append(kwargs['surface_tolerance_m'])
        return original_land(*args, **kwargs)
    monkeypatch.setattr(pipeline_module, 'smooth_heightmap', smooth)
    monkeypatch.setattr(pipeline_module, 'build_land_mesh', land)
    deps = _deps(tmp_path)
    normal = _settings()
    detailed = _settings(terrain={'maximumSourceDetail': True})
    build_kit(normal, _route(), deps)
    build_kit(detailed, _route(), deps)
    build_kit(detailed, _route(), deps)
    assert len(smoothing_calls) == 1
    assert len(tolerances) == 2
    assert tolerances[-1] is None
    assert len(list((tmp_path / '_derived' / 'terrain_meshes').glob('*.npz'))) == 2


@responses.activate
def test_detail_settings_reach_pipeline_and_invalidate_cached_terrain(tmp_path, monkeypatch):
    _add_terrain_mock()
    _add_biomes_mock()
    calls = {}
    for name in ('fetch_heightmap', 'fetch_water_polygons', 'build_land_mesh', 'build_route_mesh', 'smooth_heightmap'):
        original = getattr(pipeline_module, name)
        def capture(*args, _name=name, _original=original, **kwargs):
            calls.setdefault(_name, []).append(kwargs.copy())
            return _original(*args, **kwargs)
        monkeypatch.setattr(pipeline_module, name, capture)
    detail = dict(forceSourceZoom=True, maxZoom=14, waterZoom=13, sampleMm=.2,
                  toleranceMm=.02, smoothingSigma=1, smoothingMaxMm=.01,
                  maxVertices=10000, maxReferencePoints=100000, maxPasses=10,
                  maxTiles=100, maxRoutePoints=5000)
    settings = _settings(terrain={'detail': detail})
    deps = _deps(tmp_path)
    build_kit(settings, _route(), deps)
    build_kit(settings, _route(), deps)
    assert len(calls['build_land_mesh']) == 1
    assert calls['fetch_heightmap'][0]['max_zoom'] == 14
    assert calls['fetch_heightmap'][0]['maximum_source_detail'] is True
    assert calls['fetch_water_polygons'][0]['zoom'] == 13
    assert calls['build_land_mesh'][0]['max_vertices'] == 10000
    assert calls['build_land_mesh'][0]['max_reference_points'] == 100000
    assert calls['build_land_mesh'][0]['max_passes'] == 10
    assert calls['build_route_mesh'][0]['max_points'] == 5000
    assert calls['smooth_heightmap'][0]['sigma'] == 1
    settings.terrain.detail.smoothing_max_mm = .02
    build_kit(settings, _route(), deps)
    assert len(calls['build_land_mesh']) == 2


@responses.activate
def test_woodland_is_partitioned_only_for_export(tmp_path, monkeypatch):
    _add_terrain_mock()
    _add_biomes_mock()
    calls = []
    def coverage(frame, *args):
        calls.append(True)
        return {'wood': box(-20,-20,20,20)}
    monkeypatch.setattr(pipeline_module, 'fetch_landcover_regions', coverage)
    settings = _settings(biomes={'woodland': {'enabled': True}}, style={'colours': {'woodland': '#123456'}})
    preview = build_kit(settings, _route(), _deps(tmp_path))
    assert not calls and preview.part('woodland') is None
    exported = build_kit(settings, _route(), _deps(tmp_path), printable_landcover=True)
    assert calls and exported.part('woodland').mesh.is_volume
    assert exported.part('woodland').material.colour == '#123456'
    assert exported.part('land').mesh.volume + exported.part('woodland').mesh.volume > preview.part('land').mesh.volume


@responses.activate
@pytest.mark.parametrize('wood_enabled', [False, True])
def test_rock_export_is_coloured_and_complements_terrain(tmp_path, monkeypatch, wood_enabled):
    _add_terrain_mock()
    _add_biomes_mock()
    monkeypatch.setattr(pipeline_module, 'fetch_landcover_regions', lambda *args: {
        'rock': box(-35,-20,-5,20), 'wood': box(5,-20,35,20),
    })
    settings = _settings(biomes={'rock': {'enabled': True}, 'woodland': {'enabled': wood_enabled}},
                         style={'colours': {'rock': '#987654', 'woodland': '#123456'}})
    preview = build_kit(settings, _route(), _deps(tmp_path))
    assert preview.part('rock') is None
    exported = build_kit(settings, _route(), _deps(tmp_path), printable_landcover=True)
    assert exported.part('rock').mesh.is_volume
    assert exported.part('rock').material.colour == '#987654'
    assert (exported.part('woodland') is not None) == wood_enabled
    parts = [p.mesh for p in exported.parts if p.name in ('land', 'rock', 'woodland')]
    if wood_enabled:
        assert sum(p.volume for p in parts) > preview.part('land').mesh.volume
    else:
        assert sum(p.volume for p in parts) == pytest.approx(preview.part('land').mesh.volume, rel=1e-5)
    for i, part in enumerate(parts):
        for other in parts[i+1:]:
            overlap = trimesh.boolean.intersection([part, other], engine='manifold')
            assert overlap.is_empty or abs(overlap.volume) < 1e-5
    with zipfile.ZipFile(io.BytesIO(to_stl_zip(exported))) as archive:
        rock = trimesh.load(io.BytesIO(archive.read('rock.stl')), file_type='stl')
        assert rock.is_volume


@responses.activate
def test_infrastructure_is_visible_and_exportable(tmp_path, monkeypatch):
    _add_terrain_mock()
    _add_biomes_mock()
    monkeypatch.setattr(pipeline_module, 'fetch_infrastructure', lambda *args: (
        box(-15,-10,-14,10), [(box(15,-17,17,-15), .8)]))
    settings = _settings(biomes={'roads': {'enabled': True}, 'buildings': {'enabled': True}},
                         style={'colours': {'roads':'#112233','buildings':'#abcdef'}})
    kit = build_kit(settings, _route(), _deps(tmp_path))
    assert kit.part('roads') is not None and kit.part('roads').mesh.is_volume
    assert kit.part('buildings') is not None and kit.part('buildings').mesh.is_volume
    assert kit.part('roads').material.colour == '#112233'
    assert kit.part('buildings').material.colour == '#abcdef'
    from contour.stl_export import to_stl_zip
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(to_stl_zip(kit))) as archive:
        assert 'roads.stl' in archive.namelist()
        assert 'buildings.stl' in archive.namelist()
