"use client";

import { useEffect, useMemo, useState } from "react";
import { Group, Mesh, MeshStandardMaterial } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

interface KitMeshProps {
  glb: ArrayBuffer;
  verticalExaggeration: number;
  modelScale: number;
}

/**
 * Loads a glTF kit and renders it with the realtime transforms applied as
 * pure object transforms — no re-meshing, no shader recompilation, no
 * network roundtrip. The kit's natural units are metres; we recentre it
 * so it sits on the origin in XY and the base sits at Z = 0.
 */
export function KitMesh({ glb, verticalExaggeration, modelScale }: KitMeshProps) {
  const [root, setRoot] = useState<Group | null>(null);

  useEffect(() => {
    const loader = new GLTFLoader();
    loader.parse(
      glb,
      "",
      (gltf) => {
        const scene = gltf.scene;
        // Recentre: shift so model bbox centre is at origin in XY and base at z=0.
        scene.updateMatrixWorld(true);
        const box = boundingBox(scene);
        scene.position.set(
          -(box.min.x + box.max.x) / 2,
          -(box.min.y + box.max.y) / 2,
          -box.min.z,
        );
        // Normalise scale so the longest XY dimension fills ~2 world units;
        // the user-controlled `modelScale` then multiplies this.
        const dx = box.max.x - box.min.x;
        const dy = box.max.y - box.min.y;
        const longest = Math.max(dx, dy);
        const normalise = longest > 0 ? 2 / longest : 1;
        scene.scale.setScalar(normalise);
        setRoot(scene as Group);
      },
      (error) => {
        console.error("Failed to parse glb", error);
      },
    );
  }, [glb]);

  const wrapper = useMemo(() => new Group(), []);

  // Apply realtime transforms.
  useEffect(() => {
    if (!root) return;
    root.traverse((obj) => {
      if (obj instanceof Mesh) {
        // Vertical exaggeration is applied via the mesh's local Z scale.
        // Wrapping each mesh keeps the existing recentre-translation honest.
        obj.scale.z = verticalExaggeration;
        const mat = obj.material as MeshStandardMaterial;
        if (mat && "roughness" in mat) {
          mat.envMapIntensity = 0.6;
        }
      }
    });
  }, [root, verticalExaggeration]);

  if (!root) return null;

  return (
    <group scale={modelScale}>
      <primitive object={root} />
    </group>
  );
}

interface Box {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}

function boundingBox(root: Group): Box {
  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };
  root.traverse((obj) => {
    if (obj instanceof Mesh) {
      obj.geometry.computeBoundingBox();
      const bb = obj.geometry.boundingBox;
      if (!bb) return;
      min.x = Math.min(min.x, bb.min.x);
      min.y = Math.min(min.y, bb.min.y);
      min.z = Math.min(min.z, bb.min.z);
      max.x = Math.max(max.x, bb.max.x);
      max.y = Math.max(max.y, bb.max.y);
      max.z = Math.max(max.z, bb.max.z);
    }
  });
  return { min, max };
}
