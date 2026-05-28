"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { useEditorStore } from "@/lib/state/editor-store";
import { useMesh } from "@/lib/api/hooks";
import { KitMesh } from "./KitMesh";

export function Scene() {
  const settings = useEditorStore((s) => s.settings);
  const transforms = useEditorStore((s) => s.transforms);
  const meshQuery = useMesh(settings);

  return (
    <div className="relative h-full w-full bg-canvas">
      <Canvas camera={{ position: [3, 3, 3], fov: 35 }} shadows>
        <color attach="background" args={["#f5f3ee"]} />
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[5, 5, 5]}
          intensity={1.0}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        <Environment preset="apartment" />
        {meshQuery.data ? (
          <KitMesh
            glb={meshQuery.data.glb}
            verticalExaggeration={transforms.verticalExaggeration}
            modelScale={transforms.modelScale}
          />
        ) : null}
        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={1}
          maxDistance={20}
          target={[0, 0, 0.3]}
        />
      </Canvas>

      <StatusOverlay
        loading={meshQuery.isLoading || meshQuery.isFetching}
        error={meshQuery.error as Error | null}
        empty={!settings}
        parts={meshQuery.data?.metadata.parts ?? []}
      />
    </div>
  );
}

interface StatusOverlayProps {
  loading: boolean;
  error: Error | null;
  empty: boolean;
  parts: string[];
}

function StatusOverlay({ loading, error, empty, parts }: StatusOverlayProps) {
  if (empty) {
    return (
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <p className="text-sm text-muted">Upload a GPX to begin.</p>
      </div>
    );
  }
  if (loading) {
    return (
      <div className="absolute top-6 left-6 text-xs text-muted tracking-wider uppercase">
        Building model…
      </div>
    );
  }
  if (error) {
    return (
      <div className="absolute top-6 left-6 max-w-md text-xs text-accentRoute">
        {error.message}
      </div>
    );
  }
  if (parts.length > 0) {
    return (
      <div className="absolute bottom-6 left-6 text-[11px] text-muted tracking-wider uppercase">
        {parts.join(" · ")}
      </div>
    );
  }
  return null;
}
