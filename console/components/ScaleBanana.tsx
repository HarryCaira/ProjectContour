"use client";

import { useEffect, useMemo } from "react";
import { Html } from "@react-three/drei";
import { createBananaGeometry, VIEW_UNITS_PER_MM } from "@/lib/banana";

export function ScaleBanana({ sizeMm, viewScale }: { sizeMm: number; viewScale: number }) {
  const geometry = useMemo(createBananaGeometry, []);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <group scale={viewScale}>
      <group position={[sizeMm * VIEW_UNITS_PER_MM / 2 + 0.65, 0, 0]}>
        <mesh geometry={geometry} scale={VIEW_UNITS_PER_MM} castShadow receiveShadow>
          <meshStandardMaterial vertexColors roughness={0.8} />
        </mesh>
        <Html position={[0, -1.45, 0]} center style={{ pointerEvents: "none" }}>
          <div className="whitespace-nowrap rounded-md bg-canvas/65 px-2 py-1 text-center text-[11px] text-muted">
            Banana for scale · approx. 18 cm
          </div>
        </Html>
      </group>
    </group>
  );
}
