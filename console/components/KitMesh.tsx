"use client";

import { useEffect, useLayoutEffect, useRef, useState, useCallback, type RefObject } from "react";
import { Bvh } from "@react-three/drei";
import { Color, Group, Mesh, MeshStandardMaterial, TextureLoader, LinearFilter, NoColorSpace } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { normaliseKit, disposeKit, prepareTerrainShading, prepareRoadRendering, setKitExaggeration, measureKit, setKitColours } from "@/lib/kit-scene";

import { applySnow } from "@/lib/snow-material";
import { applyLandCover } from "@/lib/landcover-material";
import type { LandCover } from "@/lib/api";
import type { ModelColours } from "@/lib/settings";

import { ModelDimensions } from "./ModelDimensions";

interface KitMeshProps {
  coverEnabled: { woodland: boolean; rock: boolean };
  snow?: { enabled: boolean; snowline: number };
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

export function KitMesh({ coverEnabled, snow, glb, landCover, meshSizeMm, modelRef, colours, verticalExaggeration, modelScale, routeWidthScale, routeHeightScale, physicalSizeMm, showDimensions, onError, onReady }: KitMeshProps) {
  const coverState = useRef({ woodland: { value: 1 }, rock: { value: 1 } });
  coverState.current.woodland.value = coverEnabled.woodland ? 1 : 0;
  coverState.current.rock.value = coverEnabled.rock ? 1 : 0;
  const snowState = useRef({ enabled: { value: 0 }, line: { value: .72 }, colour: new Color("#f3f4ef") });
  snowState.current.enabled.value = snow?.enabled ? 1 : 0;
  snowState.current.line.value = snow?.snowline ?? .72;
  snowState.current.colour.set(colours.snow ?? "#f3f4ef");
  const woodlandPhysicalScale = useRef({ value: 1 });
  woodlandPhysicalScale.current.value = physicalSizeMm / meshSizeMm;
  const coverColours = useRef({ woodland: new Color("#365d38"), rock: new Color("#96928a") });
  const attachModel = useCallback((object: Group | null) => {
    modelRef.current = object ?? new Group();
  }, [modelRef]);
  const [measurements, setMeasurements] = useState<ReturnType<typeof measureKit> | null>(null);
  const [root, setRoot] = useState<Group | null>(null);

  useEffect(() => {
    let cancelled = false;
    let loadedRoot: Group | undefined;
    setRoot(null);
    onError(null);
    const fail = (error: unknown): void => {
      if (!cancelled) {
        onError(error instanceof Error ? error : new Error("Could not display the model."));
      }
    };
    const loader = new GLTFLoader();
    loader.parse(glb, "", (gltf) => {
      if (cancelled) { disposeKit(gltf.scene); return; }
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
        prepareRoadRendering(gltf.scene);
        loadedRoot = normaliseKit(gltf.scene);
        setRoot(loadedRoot);
      } catch (error) {
        disposeKit(gltf.scene);
        fail(error);
      }
    }, fail);
    return () => { cancelled = true; if (loadedRoot) disposeKit(loadedRoot); };
  }, [glb, onError]);

  useLayoutEffect(() => {
    if (root) {
      setKitExaggeration(root, verticalExaggeration, routeWidthScale, routeHeightScale);
      setMeasurements(measureKit(root, physicalSizeMm));
    }
  }, [root, verticalExaggeration, physicalSizeMm, routeWidthScale, routeHeightScale]);

  useLayoutEffect(() => {
    coverColours.current.woodland.set(colours.woodland ?? "#365d38");
    coverColours.current.rock.set(colours.rock ?? "#96928a");
    if (root) setKitColours(root, colours);
  }, [root, colours]);

  useEffect(() => {
    if (!root) return;
    if (!landCover) return applySnow(root, snowState.current);
    let cancelled = false;
    let restore: (() => void) | undefined;
    const texture = new TextureLoader().load(landCover.image, loaded => {
      if (cancelled) return;
      loaded.colorSpace = NoColorSpace;
      loaded.minFilter = loaded.magFilter = LinearFilter;
      loaded.generateMipmaps = false;
      const restoreCover = applyLandCover(root, loaded, landCover.bounds, meshSizeMm, coverColours.current, woodlandPhysicalScale.current, coverState.current);
      const restoreSnow = applySnow(root, snowState.current);
      restore = () => { restoreSnow(); restoreCover(); };
    });
    return () => { cancelled = true; restore?.(); texture.dispose(); };
  }, [root, landCover, meshSizeMm, applyLandCover]);

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
