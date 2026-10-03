from pathlib import Path
import subprocess


def test_snow_is_local_reversible_and_preserves_other_materials():
    script = '''
import assert from 'node:assert/strict';
import { Group, Mesh, BoxGeometry, MeshStandardMaterial, Color } from 'three';
import { applySnow } from './lib/snow-material.ts';
import { defaultSettings } from './lib/settings.ts';
import { previewSettings } from './lib/preview-settings.ts';
const root = new Group();
const land = new Mesh(new BoxGeometry(), new MeshStandardMaterial());land.name='land';root.add(land);
const water = new Mesh(new BoxGeometry(),new MeshStandardMaterial());water.name='water';root.add(water);
const before=land.material.onBeforeCompile, waterBefore=water.material.onBeforeCompile;
const state={enabled:{value:1},line:{value:.72},colour:new Color('#f3f4ef')};
const restore=applySnow(root,state);
const shader={uniforms:{},vertexShader:'#include <begin_vertex>',fragmentShader:'#include <roughnessmap_fragment>'};
land.material.onBeforeCompile(shader,{});
state.line.value=.6;
assert.equal(shader.uniforms.snowLine.value,.6);
assert.equal(water.material.onBeforeCompile,waterBefore);
restore();assert.equal(land.material.onBeforeCompile,before);
const a=defaultSettings({type:'gpx',id:'x',sha256:'y'});
const b=structuredClone(a);b.biomes.snow={enabled:true,snowline:.3};b.style.colours.snow='#ffffff';
assert.deepEqual(previewSettings(a),previewSettings(b));
'''
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',script],
                   cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
