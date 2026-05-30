import type { Settings } from "@/lib/schema/settings";

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
  metadata: KitMetadata;
}

export async function uploadGpx(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!r.ok) throw await asError(r);
  return r.json();
}

export async function fetchMesh(settings: Settings): Promise<MeshResult> {
  const r = await fetch(`${BASE}/mesh`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!r.ok) throw await asError(r);
  return {
    glb: await r.arrayBuffer(),
    metadata: {
      parts: (r.headers.get("x-kit-parts") ?? "").split(",").filter(Boolean),
      triangles: (r.headers.get("x-kit-triangles") ?? "")
        .split(",")
        .filter(Boolean)
        .map((n) => parseInt(n, 10)),
    },
  };
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
