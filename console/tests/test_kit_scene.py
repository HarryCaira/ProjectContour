"""Regression checks for viewer transforms using the actual Three.js implementation.

Run with backend/.venv/bin/pytest console/tests from the repository root.
Requires the console's Node dependencies to be installed.
"""
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def console_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("elevation", [-1000, 0, 2500])
@pytest.mark.parametrize("exaggeration", [0.5, 1.5, 5])
def test__normalise_kit__centres_transformed_geometry_and_preserves_base(
    console_dir: Path, elevation: int, exaggeration: float
) -> None:
    script = """
import assert from 'node:assert/strict';
import { Box3, BoxGeometry, Group, Mesh, Vector3 } from 'three';
import { normaliseKit, setKitExaggeration } from './lib/kit-scene.ts';
const [elevation, exaggeration] = process.argv.slice(1).map(Number);
const scene = new Group();
scene.position.set(100, -200, elevation);
const mesh = new Mesh(new BoxGeometry(2000, 1000, 100));
mesh.position.set(50, 30, 50);
scene.add(mesh);
const wrapper = new Group();
const kit = normaliseKit(scene);
setKitExaggeration(kit, exaggeration);
wrapper.add(kit);
const bounds = new Box3().setFromObject(wrapper);
const centre = bounds.getCenter(new Vector3());
const size = bounds.getSize(new Vector3());
assert.ok(Math.abs(centre.x) < 1e-8);
assert.ok(Math.abs(centre.y) < 1e-8);
assert.ok(Math.abs(bounds.min.z) < 1e-8);
assert.ok(Math.abs(size.x - 2) < 1e-8);
assert.ok(Math.abs(size.z - 0.1 * exaggeration) < 1e-8);
"""
    subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script, "--",
         str(elevation), str(exaggeration)],
        cwd=console_dir, check=True, capture_output=True, text=True,
    )


def test__normalise_kit__rejects_empty_geometry(console_dir: Path) -> None:
    script = """
import assert from 'node:assert/strict';
import { Group } from 'three';
import { normaliseKit, setKitExaggeration } from './lib/kit-scene.ts';
assert.throws(() => normaliseKit(new Group()), /no viewable geometry/);
"""
    subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
        cwd=console_dir, check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize("exaggeration", [0.5, 1, 3, 5])
def test__set_kit_exaggeration__keeps_plinth_fixed_and_land_attached(console_dir: Path, exaggeration: float) -> None:
    script = """
import assert from 'node:assert/strict';
import { Box3, BoxGeometry, Group, Mesh } from 'three';
import { normaliseKit, setKitExaggeration } from './lib/kit-scene.ts';
const exaggeration = Number(process.argv[1]);
const scene = new Group();
scene.position.set(25, -30, 1000);
const plinth = new Mesh(new BoxGeometry(200, 200, 20));
plinth.name = 'plinth';
plinth.position.z = -10;
const land = new Mesh(new BoxGeometry(200, 200, 100));
land.name = 'land';
land.position.z = 50;
scene.add(plinth, land);
const root = normaliseKit(scene);
root.updateMatrixWorld(true);
const before = new Box3().setFromObject(plinth);
for (const factor of [exaggeration, 1, exaggeration]) {
  setKitExaggeration(root, factor);
  root.updateMatrixWorld(true);
  const base = new Box3().setFromObject(plinth);
  const terrain = new Box3().setFromObject(land);
  assert.ok(base.min.distanceTo(before.min) < 1e-8);
  assert.ok(base.max.distanceTo(before.max) < 1e-8);
  assert.ok(Math.abs(terrain.min.z - base.max.z) < 1e-8);
  assert.ok(Math.abs(terrain.max.z - terrain.min.z - factor) < 1e-8);
}
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script, str(exaggeration)],
                   cwd=console_dir, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("size_mm", [50, 150, 300])
def test__measure_kit__uses_physical_size_and_relief_but_not_preview_zoom(console_dir: Path, size_mm: int) -> None:
    script = """
import assert from 'node:assert/strict';
import { BoxGeometry, Group, Mesh } from 'three';
import { normaliseKit, setKitExaggeration, measureKit } from './lib/kit-scene.ts';
const size = Number(process.argv[1]);
const scene = new Group();
const plinth = new Mesh(new BoxGeometry(200, 100, 20));
plinth.name = 'plinth';
plinth.position.z = -10;
const land = new Mesh(new BoxGeometry(200, 100, 100));
land.position.z = 50;
scene.add(plinth, land);
const kit = normaliseKit(scene);
const zoom = new Group();
zoom.scale.setScalar(7);
zoom.position.set(40, 20, 10);
zoom.add(kit);
zoom.updateMatrixWorld(true);
setKitExaggeration(kit, 2);
const measured = measureKit(kit, size);
assert.ok(Math.abs(measured.millimetres.x - size) < 1e-8);
assert.ok(Math.abs(measured.millimetres.y - size / 2) < 1e-8);
assert.ok(Math.abs(measured.millimetres.z - size * 1.1) < 1e-8);
assert.ok(Math.abs(measured.bounds.min.z) < 1e-8);
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script, str(size_mm)],
                   cwd=console_dir, check=True, capture_output=True, text=True)


def test__set_kit_exaggeration__route_keeps_physical_width_and_height(console_dir: Path) -> None:
    script = """
import assert from 'node:assert/strict';
import { Box3, BoxGeometry, Group, Mesh } from 'three';
import { normaliseKit, setKitExaggeration } from './lib/kit-scene.ts';
const scene = new Group();
const plinth = new Mesh(new BoxGeometry(150, 100, 5));
plinth.name = 'plinth'; plinth.position.z = -2.5;
const land = new Mesh(new BoxGeometry(150, 100, 60)); land.position.z = 30;
const route = new Mesh(new BoxGeometry(2, 100, 1));
route.name = 'route'; route.position.z = 60.5;
scene.add(plinth, land, route);
const kit = normaliseKit(scene);
for (const factor of [1, 2, 4, 0.5, 1]) {
  setKitExaggeration(kit, factor);
  kit.updateMatrixWorld(true);
  const bounds = new Box3().setFromObject(route);
  const terrain = new Box3().setFromObject(land);
  assert.ok(Math.abs((bounds.max.x - bounds.min.x) * 75 - 2) < 1e-5);
  assert.ok(Math.abs((bounds.max.z - bounds.min.z) * 75 - 1) < 1e-5);
  assert.ok(Math.abs(bounds.min.z - terrain.max.z) < 1e-5);
}
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                   cwd=console_dir, check=True, capture_output=True, text=True)


def test_colours_update_normalised_parts_without_changing_geometry(console_dir: Path) -> None:
    script = """
import assert from 'node:assert/strict';
import { Group, Mesh, MeshStandardMaterial, BoxGeometry } from 'three';
import { normaliseKit, setKitColours } from './lib/kit-scene.ts';
const scene = new Group();
const parts = {};
for (const name of ['land', 'water', 'route', 'plinth']) {
  const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial({color:'#ffffff'}));
  mesh.name = name;
  scene.add(mesh);
  parts[name] = mesh;
}
const kit = normaliseKit(scene);
const geometry = parts.route.geometry;
setKitColours(kit, {terrain:'#123456', water:'#abcdef', route:'#fedcba'});
assert.equal(parts.land.material.color.getHexString(), '123456');
assert.equal(parts.water.material.color.getHexString(), 'abcdef');
assert.equal(parts.route.material.color.getHexString(), 'fedcba');
assert.equal(parts.plinth.material.color.getHexString(), 'ffffff');
assert.equal(parts.route.geometry, geometry);
"""
    subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script],
        cwd=console_dir, check=True, capture_output=True, text=True,
    )


def test_route_edits_are_local_and_keep_physical_dimensions(console_dir: Path) -> None:
    script = """
import assert from 'node:assert/strict';
import { BoxGeometry, Box3, Group, Mesh, Float32BufferAttribute, Vector3 } from 'three';
import { normaliseKit, setKitExaggeration } from './lib/kit-scene.ts';
import { defaultSettings } from './lib/settings.ts';
import { previewSettings } from './lib/preview-settings.ts';
const settings = defaultSettings({type:'gpx', id:'x', sha256:'test'});
const originalKey = JSON.stringify(previewSettings(settings));
settings.route.widthMm = 4;
settings.route.heightAboveTerrainMm = 3;
settings.terrain.verticalExaggeration = 4;
settings.style.colours.terrain = '#123456';
assert.equal(JSON.stringify(previewSettings(settings)), originalKey);
const scene = new Group();
const land = new Mesh(new BoxGeometry(100,80,10));
land.name='land'; land.position.z=5;
const plinth = new Mesh(new BoxGeometry(100,80,2));
plinth.name='plinth'; plinth.position.z=-1;
const route = new Mesh(new BoxGeometry(10,2,1));
route.name='route'; route.position.z=10.5;
const offsets=[];
const p=route.geometry.attributes.position;
for(let i=0;i<p.count;i++) offsets.push(0,p.getY(i),p.getZ(i)+.5);
route.geometry.setAttribute('_route_offset',new Float32BufferAttribute(offsets,3));
scene.add(land,plinth,route);
const kit=normaliseKit(scene);
const landGeometry=land.geometry;
const unchanged=Array.from(land.geometry.attributes.position.array);
for(const size of [100,50,100]) {
  setKitExaggeration(kit,2,3*100/size,2*100/size);
  kit.updateMatrixWorld(true);
  const dims=new Box3().setFromObject(route).getSize(new Vector3()).multiplyScalar(size/2);
  assert.ok(Math.abs(dims.y-6)<1e-5);
  assert.ok(Math.abs(dims.z-2)<1e-5);
}
assert.equal(land.geometry,landGeometry);
assert.deepEqual(Array.from(land.geometry.attributes.position.array),unchanged);
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                   cwd=console_dir, check=True, capture_output=True, text=True)
