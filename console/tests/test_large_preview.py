from pathlib import Path
import subprocess


def test_large_terrain_stays_indexed_and_releases_resources():
    script='''
import assert from 'node:assert/strict';
import {Group,Mesh,PlaneGeometry,MeshStandardMaterial} from 'three';
import {prepareTerrainShading,disposeKit} from './lib/kit-scene.ts';
const geometry=new PlaneGeometry(10,10,600,600),material=new MeshStandardMaterial();
const land=new Mesh(geometry,material);land.name='land';const root=new Group();root.add(land);
const positions=geometry.attributes.position;
prepareTerrainShading(root);
assert.ok(land.geometry.index);
assert.equal(land.geometry.attributes.position,positions);
let geometries=0,materials=0,bvhs=0;
geometry.addEventListener('dispose',()=>geometries++);
material.addEventListener('dispose',()=>materials++);
geometry.disposeBoundsTree=()=>bvhs++;
disposeKit(root);
assert.equal(geometries,1);assert.equal(materials,1);assert.equal(bvhs,1);
'''
    subprocess.run(['node','--max-old-space-size=256','--experimental-strip-types','--input-type=module','-e',script],
                   cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
