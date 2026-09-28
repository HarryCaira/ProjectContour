"use client";

import { useState, useRef } from "react";
import { Group } from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { useEditorStore } from "@/lib/editor-store";
import { DEFAULT_COLOURS } from "@/lib/settings";
import { useQuery } from "@tanstack/react-query";
import { fetchLandCover } from "@/lib/api";
import { useMesh } from "@/lib/hooks";
import { ScaleBanana } from "./ScaleBanana";
import { ModelProgress } from "./ModelProgress";
import { KitMesh } from "./KitMesh";

export function Scene() {
  const modelRef = useRef<Group>(new Group());
  const [showDimensions, setShowDimensions] = useState(true);
  const [showBanana, setShowBanana] = useState(true);
  const [readyGlb, setReadyGlb] = useState<ArrayBuffer | null>(null);
  const [viewerError, setViewerError] = useState<Error | null>(null);
  const settings = useEditorStore((s) => s.settings);
  const meshQuery = useMesh(settings);
  const showLandCover = useEditorStore(s => s.showLandCover);
  const coverQuery = useQuery({
    queryKey: ["landcover", settings?.source, settings?.framing],
    queryFn: ({ signal }) => fetchLandCover(settings!, signal),
    enabled: !!settings && showLandCover,
    staleTime: Infinity, retry: false,
  });

  return (
    <div className="relative h-full w-full bg-canvas">
      <Canvas camera={{ position: [5, 5, 5], up: [0, 0, 1], fov: 35 }} shadows>
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
            key={meshQuery.dataUpdatedAt}
            glb={meshQuery.data.glb}
            modelRef={modelRef}
            landCover={showLandCover && meshQuery.matchesCoverage ? coverQuery.data : undefined}
            meshSizeMm={meshQuery.data.physicalSizeMm ?? 100}
            colours={settings?.style.colours ?? DEFAULT_COLOURS}
            onReady={setReadyGlb}
            onError={setViewerError}
            showDimensions={showDimensions}
            physicalSizeMm={settings?.physical.sizeMm ?? 150}
            verticalExaggeration={settings?.terrain.verticalExaggeration ?? 1}
            modelScale={(settings?.physical.sizeMm ?? 150) / 150}
            routeWidthScale={(settings?.route.widthMm ?? 1) * (meshQuery.data.physicalSizeMm ?? 150) / (settings?.physical.sizeMm ?? 150)}
            routeHeightScale={(settings?.route.heightAboveTerrainMm ?? 1) * (meshQuery.data.physicalSizeMm ?? 150) / (settings?.physical.sizeMm ?? 150)}
          />
        ) : null}
        {meshQuery.data && settings && showBanana && (
          <ScaleBanana modelRef={modelRef} sizeMm={settings.physical.sizeMm} viewScale={1} />
        )}
        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          minDistance={1}
          maxDistance={20}
          target={[0, 0, 0.3]}
        />
      </Canvas>

      {settings && (
        <div className="absolute top-5 left-5 right-44 flex flex-col items-start gap-3">
        <div className="flex flex-wrap gap-2">
        <label className="flex items-center gap-2 rounded-lg border border-line bg-canvas/95 px-3 py-2 text-xs text-muted">
          <input type="checkbox" checked={showBanana} onChange={(event) => setShowBanana(event.target.checked)}
            className="accent-accent" />
          Banana for scale
        </label>
        <label className="flex items-center gap-2 rounded-lg border border-line bg-canvas/95 px-3 py-2 text-xs text-muted">
          <input type="checkbox" checked={showDimensions} onChange={(event) => setShowDimensions(event.target.checked)}
            className="accent-accent" />
          Dimensions
        </label>

        </div>
        </div>
      )}
      {settings && (
        <button type="button" onClick={() => { setViewerError(null); void meshQuery.refetch(); }}
          disabled={meshQuery.isFetching}
          className="absolute top-5 right-5 rounded-lg border border-line bg-canvas/95 px-3 py-2 text-xs text-muted disabled:opacity-50">
          Rebuild preview
        </button>
      )}
      {settings && showLandCover && (
        <div className="absolute bottom-12 left-5 max-w-sm rounded-md bg-canvas/90 px-3 py-2 text-xs text-muted">
          {coverQuery.isPending ? "Mapping woodland and rock…" : coverQuery.isError ? "Land cover unavailable. Terrain remains visible." :
            `Mapped woodland ${coverQuery.data.percentages.wood}% · rock ${coverQuery.data.percentages.rock}%`}
        </div>
      )}
      {meshQuery.isFetching && meshQuery.data && (
        <div role="status" className="absolute bottom-6 right-6 rounded-lg bg-canvas/90 px-3 py-2 text-xs text-muted">
          Updating print detail… You can keep editing.
        </div>
      )}
      <StatusOverlay
        loading={(!meshQuery.data && (meshQuery.isFetching || !!settings)) || (!!meshQuery.data && readyGlb !== meshQuery.data.glb)}
        stage={meshQuery.isFetching ? meshQuery.progress?.stage ?? "starting" : "display"}
        onRetry={() => { setViewerError(null); void meshQuery.refetch(); }}
        error={meshQuery.error ?? viewerError}
        empty={!settings}
        parts={meshQuery.data?.metadata.parts ?? []}
      />
    </div>
  );
}

interface StatusOverlayProps {
  loading: boolean;
  stage: string;
  onRetry: () => void;
  error: Error | null;
  empty: boolean;
  parts: string[];
}

function StatusOverlay({ loading, stage, onRetry, error, empty, parts }: StatusOverlayProps) {
  if (empty) {
    return (
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <p className="text-sm text-muted">Upload a GPX to begin.</p>
      </div>
    );
  }
  if (error) {
    return (
      <div role="alert" className="absolute top-6 left-6 max-w-md rounded-xl border border-line bg-canvas p-5 shadow-sm">
        <p className="text-sm text-ink">We couldn't prepare your model.</p>
        <p className="mt-2 text-xs text-muted">{error.message}</p>
        <button onClick={onRetry} className="mt-4 rounded-md border border-line px-4 py-2 text-sm">Try again</button>
      </div>
    );
  }
  if (loading) return <ModelProgress stage={stage} />;
  if (parts.length > 0) {
    return (
      <div className="absolute bottom-6 left-6 text-[11px] text-muted tracking-wider uppercase">
        {parts.join(" · ")}
      </div>
    );
  }
  return null;
}
