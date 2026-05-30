"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchMesh, uploadGpx, downloadExport } from "@/lib/api/client";
import type { Settings } from "@/lib/schema/settings";

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
  return useQuery({
    queryKey: ["mesh", settings ? stableHash(settings) : null],
    queryFn: () => fetchMesh(settings as Settings),
    enabled: !!settings,
    staleTime: Infinity,
    gcTime: 1000 * 60 * 30,
  });
}

export function useExport() {
  return useMutation({
    mutationKey: ["export"],
    mutationFn: (settings: Settings) => downloadExport(settings),
  });
}

/** Hash that excludes the realtime-only fields. */
function stableHash(settings: Settings): string {
  // For v1 we hash the whole settings object — the realtime-only fields
  // (verticalExaggeration, modelScale) live OUTSIDE Settings, in the
  // editor store, so settings already only contains topology-affecting
  // values. Keep it simple.
  return JSON.stringify(settings);
}

export { useQueryClient };
