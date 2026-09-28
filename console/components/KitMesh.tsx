"use client";

import { useEffect, useLayoutEffect, useState, useCallback, type RefObject } from "react";
import { Bvh } from "@react-three/drei";
import { Group, Mesh, MeshStandardMaterial, TextureLoader, NearestFilter, NoColorSpace } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { normaliseKit, prepareTerrainShading, setKitExaggeration, measureKit, setKitColours } from "@/lib/kit-scene";

import { applyLandCover } from "@/lib/landcover-material";
import type { LandCover } from "@/lib/api";
import type { ModelColours } from "@/lib/settings";

import { ModelDimensions } from "./ModelDimensions";

interface KitMeshProps {
  glb: ArrayBuffer;
  landCover?: LandCover;
  meshSizeMm: number;
  modelRef: RefObject<Group>;
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

export function KitMesh({ glb, landCover, meshSizeMm, modelRef, colours, verticalExaggeration, modelScale, routeWidthScale, routeHeightScale, physicalSizeMm, showDimensions, onError, onReady }: KitMeshProps) {
  const attachModel = useCallback((object: Group | null) => {
    modelRef.current = object ?? new Group();
  }, [modelRef]);
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
        prepareTerrainShading(gltf.scene);
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

  useEffect(() => {
    if (!root || !landCover) return;
    let cancelled = false;
    let restore: (() => void) | undefined;
    const texture = new TextureLoader().load(landCover.image, loaded => {
      if (cancelled) return;
      loaded.colorSpace = NoColorSpace;
      loaded.minFilter = loaded.magFilter = NearestFilter;
      loaded.generateMipmaps = false;
      restore = applyLandCover(root, loaded, landCover.bounds, meshSizeMm);
    });
    return () => { cancelled = true; restore?.(); texture.dispose(); };
  }, [root, landCover, meshSizeMm]);

  useLayoutEffect(() => {
    if (root) onReady(glb);
  }, [root, glb, onReady]);

  if (!root) return null;

  return (
    <group scale={modelScale}>
      <Bvh firstHitOnly strategy={0} indirect>
        <primitive object={root} ref={attachModel} />
      </Bvh>
      {measurements && <ModelDimensions modelRef={modelRef} {...measurements} visible={showDimensions} />}
    </group>
  );
}
