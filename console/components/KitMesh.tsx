"use client";

import { useEffect, useLayoutEffect, useState } from "react";
import { Group, Mesh, MeshStandardMaterial } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { normaliseKit, setKitExaggeration, measureKit, setKitColours } from "@/lib/kit-scene";

import type { ModelColours } from "@/lib/settings";

import { ModelDimensions } from "./ModelDimensions";

interface KitMeshProps {
  glb: ArrayBuffer;
  colours: ModelColours;
  verticalExaggeration: number;
  modelScale: number;
  routeWidthScale: number;
  routeHeightScale: number;
  physicalSizeMm: number;
  showDimensions: boolean;
  onError: (error: Error | null) => void;
  onReady: (glb: ArrayBuffer) => void;
}

export function KitMesh({ glb, colours, verticalExaggeration, modelScale, routeWidthScale, routeHeightScale, physicalSizeMm, showDimensions, onError, onReady }: KitMeshProps) {
  const [measurements, setMeasurements] = useState<ReturnType<typeof measureKit> | null>(null);
  const [root, setRoot] = useState<Group | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRoot(null);
    onError(null);
    const fail = (error: unknown): void => {
      if (!cancelled) {
        onError(error instanceof Error ? error : new Error("Could not display the model."));
      }
    };
    const loader = new GLTFLoader();
    loader.parse(glb, "", (gltf) => {
      if (cancelled) return;
      try {
        gltf.scene.traverse((obj) => {
          if (obj instanceof Mesh) {
            const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
            for (const material of materials) {
              if (material instanceof MeshStandardMaterial) material.envMapIntensity = 0.6;
            }
          }
        });
        setRoot(normaliseKit(gltf.scene));
      } catch (error) {
        fail(error);
      }
    }, fail);
    return () => { cancelled = true; };
  }, [glb, onError]);

  useLayoutEffect(() => {
    if (root) {
      setKitExaggeration(root, verticalExaggeration, routeWidthScale, routeHeightScale);
      setMeasurements(measureKit(root, physicalSizeMm));
    }
  }, [root, verticalExaggeration, physicalSizeMm, routeWidthScale, routeHeightScale]);

  useLayoutEffect(() => {
    if (root) setKitColours(root, colours);
  }, [root, colours]);

  useLayoutEffect(() => {
    if (root) onReady(glb);
  }, [root, glb, onReady]);

  if (!root) return null;

  return (
    <group scale={modelScale}>
      <primitive object={root} />
      {showDimensions && measurements && <ModelDimensions {...measurements} />}
    </group>
  );
}
