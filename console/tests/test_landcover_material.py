"""The coverage overlay is reversible and leaves printable geometry untouched."""
from pathlib import Path
import subprocess


def test_coverage_material_and_cleanup():
    script = '''
import assert from 'node:assert/strict';
import { Color, Group, Mesh, BoxGeometry, MeshStandardMaterial, Texture } from 'three';
import { applyLandCover } from './lib/landcover-material.ts';
const root = new Group();
const land = new Mesh(new BoxGeometry(),new MeshStandardMaterial()); land.name='land'; root.add(land);
const water = new Mesh(new BoxGeometry(),new MeshStandardMaterial()); water.name='water'; root.add(water);
const original = land.material.onBeforeCompile;
const key = land.material.customProgramCacheKey;
const waterOriginal = water.material.onBeforeCompile;
const positions = Array.from(land.geometry.attributes.position.array);
const colours = { woodland: new Color("#112233"), rock: new Color("#abcdef") };
const restore = applyLandCover(root,new Texture(),[-100,-200,100,200],100,colours);
const firstOverlayKey = land.material.customProgramCacheKey();
const shader = {uniforms:{}, vertexShader:'#include <begin_vertex>',fragmentShader:'#include <color_fragment>'};
land.material.onBeforeCompile(shader,{});
assert.deepEqual(shader.uniforms.coverageBounds.value.toArray(),[-25,-50,50,100]);
assert.ok(shader.vertexShader.includes('normal.z'));
assert.ok(shader.fragmentShader.includes('woodColour'));
assert.equal(shader.uniforms.woodColour.value.getHexString(), '112233');
assert.equal(shader.uniforms.rockColour.value.getHexString(), 'abcdef');
colours.woodland.set('#fedcba');
assert.equal(shader.uniforms.woodColour.value.getHexString(), 'fedcba');
assert.equal(water.material.onBeforeCompile,waterOriginal);
restore();
assert.equal(land.material.onBeforeCompile,original);
assert.equal(land.material.customProgramCacheKey,key);
assert.deepEqual(Array.from(land.geometry.attributes.position.array),positions);
const restoreAgain = applyLandCover(root,new Texture(),[-100,-200,100,200],150);
assert.notEqual(land.material.customProgramCacheKey(),firstOverlayKey);
restoreAgain();
'''
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',script],
                   cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
