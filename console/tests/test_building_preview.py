from pathlib import Path
import subprocess


def test_building_roof_height_is_independent_of_route_and_exaggeration():
    script = '''
import assert from 'node:assert/strict';
import {Group,Mesh,BoxGeometry,MeshStandardMaterial,Float32BufferAttribute} from 'three';
import {normaliseKit,setKitExaggeration} from './lib/kit-scene.ts';
const scene=new Group();
const land=new Mesh(new BoxGeometry(10,10,2),new MeshStandardMaterial());land.name='land';land.position.z=1;scene.add(land);
const building=new Mesh(new BoxGeometry(1,1,1),new MeshStandardMaterial());building.name='buildings';building.geometry.translate(0,0,2.5);
const p=building.geometry.attributes.position;
const offsets=new Float32Array(p.count*3);
for(let i=0;i<p.count;i++) if(p.getZ(i)>2.5) offsets[i*3+2]=1;
building.geometry.setAttribute('_route_offset',new Float32BufferAttribute(offsets,3));scene.add(building);
const root=normaliseKit(scene);setKitExaggeration(root,2,4,5);
const z=Array.from(building.geometry.attributes.position.array).filter((_,i)=>i%3===2);
assert.equal(Math.max(...z),5);assert.equal(Math.min(...z),4);
'''
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',script],
                   cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
