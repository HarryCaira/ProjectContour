import { readMeshStream, type MeshProgress } from "@/lib/mesh-stream";
import type { Settings } from "@/lib/settings";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface UploadResponse {
  id: string;
  sha256: string;
  stats: { points: number; distance_km: number };
}

export interface KitMetadata {
  parts: string[];
  triangles: number[];
}

export interface MeshResult {
  glb: ArrayBuffer;
  physicalSizeMm?: number;
  maximumSourceDetail?: boolean;
  metadata: KitMetadata;
}

export async function uploadGpx(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  let r: Response;
  try {
    r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  } catch {
    throw new Error("Couldn't connect to the upload service. Please try again shortly.");
  }
  if (!r.ok) throw await asError(r);
  return r.json();
}

export async function fetchMesh(
  settings: Settings,
  onProgress: (progress: MeshProgress) => void,
  signal?: AbortSignal,
): Promise<MeshResult> {
  const r = await fetch(`${BASE}/mesh/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(settings),
    signal,
  });
  if (!r.ok) throw await asError(r);
  if (!r.body) throw new Error("Your browser couldn't receive the model. Please try again.");
  return { ...await readMeshStream(r.body, onProgress), physicalSizeMm: settings.physical.sizeMm, maximumSourceDetail: settings.terrain.maximumSourceDetail };
}

export async function downloadExport(settings: Settings): Promise<Blob> {
  const r = await fetch(`${BASE}/export`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!r.ok) throw await asError(r);
  return r.blob();
}

async function asError(r: Response): Promise<Error> {
  let body: string;
  try {
    const json = await r.json();
    body = json.detail ?? json.message ?? JSON.stringify(json);
  } catch {
    body = await r.text();
  }
  return new Error(`${r.status}: ${body}`);
}


export interface LandCover {
  image: string;
  bounds: [number, number, number, number];
  percentages: { wood: number; rock: number };
  zoom: number;
}

export async function fetchLandCover(settings: Settings, signal?: AbortSignal): Promise<LandCover> {
  const response = await fetch(`${BASE}/landcover`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(settings), signal,
  });
  if (!response.ok) throw await asError(response);
  return response.json();
}
