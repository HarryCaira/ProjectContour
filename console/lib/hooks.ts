"use client";

import { useEffect, useState } from "react";
import type { MeshProgress } from "@/lib/mesh-stream";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchMesh, uploadGpx, downloadExport, fetchFeatureCoverage } from "@/lib/api";
import type { Settings } from "@/lib/settings";
import { previewSettings } from "@/lib/preview-settings";

/** Upload a GPX file. Caller is responsible for stashing the response into the editor store. */
export function useUploadGpx() {
  return useMutation({
    mutationKey: ["upload"],
    mutationFn: (file: File) => uploadGpx(file),
  });
}

/**
 * Build a mesh kit for the given settings. Cached by a stable settings hash so
 * tweaks within the same topology re-use the result; topology-changing edits
 * yield a new key and trigger a fresh fetch.
 */
export function useMesh(settings: Settings | null) {
  const preview = settings ? previewSettings(settings) : null;
  const physicalFeatures = preview?.biomes.roads?.enabled || preview?.biomes.buildings?.enabled;
  const geometryKey = preview ? JSON.stringify({ ...preview, physical: { ...preview.physical, sizeMm: physicalFeatures ? preview.physical.sizeMm : 0 } }) : null;
  const [detail, setDetail] = useState<{ geometryKey: string | null; size: number }>({ geometryKey: null, size: 0 });
  const size = preview ? Math.max(preview.physical.sizeMm, detail.geometryKey === geometryKey ? detail.size : 0) : 0;
  const desiredKey = preview ? JSON.stringify({ ...preview, physical: { ...preview.physical, sizeMm: size } }) : null;
  const [key, setKey] = useState<string | null>(null);
  useEffect(() => {
    // Geometry edits settle before starting a build. Width/height, colours and
    // relief do not change this key; shrinking reuses the higher-detail mesh.
    const timer = setTimeout(() => {
      setDetail({ geometryKey, size });
      setKey(desiredKey);
    }, 350);
    return () => clearTimeout(timer);
  }, [desiredKey, geometryKey, size]);
  const [progress, setProgress] = useState<(MeshProgress & { key: string | null }) | null>(null);
  const query = useQuery({
    queryKey: ["mesh-interactive-v21", key],
    queryFn: async ({ signal }) => {
      const requested = JSON.parse(key!) as Settings;
      setProgress({ key, stage: "starting" });
      const mesh = await fetchMesh(requested, (update) => {
        if (!signal.aborted) setProgress({ ...update, key });
      }, signal);
      return { ...mesh, coverageKey: JSON.stringify([requested.source, requested.framing]) };
    },
    retry: false,
    placeholderData: keepPreviousData,
    enabled: !!settings && !!key,
    staleTime: Infinity,
    gcTime: 0,
  });
  return { ...query, matchesCoverage: query.data?.coverageKey === JSON.stringify([settings?.source, settings?.framing]), progress: progress?.key === key ? progress : null };
}

export function useExport() {
  return useMutation({
    mutationKey: ["export"],
    mutationFn: (settings: Settings) => downloadExport(settings),
  });
}

export { useQueryClient };

/** Layer switches must not change source availability or hide their own controls. */
export function useFeatureCoverage(settings: Settings | null) {
  return useQuery({
    queryKey: ["feature-coverage-v3", settings?.source, settings?.framing, settings?.physical.sizeMm],
    queryFn: ({ signal }) => fetchFeatureCoverage(settings!, signal),
    enabled: !!settings,
    staleTime: Infinity,
    retry: false,
  });
}
