export interface MeshProgress {
  stage: string;
}

interface StreamResult {
  glb: ArrayBuffer;
  metadata: { parts: string[]; triangles: number[] };
}

/** Read newline-delimited events even when network chunks split JSON records. */
export async function readMeshStream(
  body: ReadableStream<Uint8Array>,
  onProgress: (progress: MeshProgress) => void,
): Promise<StreamResult> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let mesh: Uint8Array | undefined;
  let offset = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      if (done && pending.trim()) lines.push(pending);
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "progress") onProgress({ stage: event.stage });
        if (event.type === "mesh_start") {
          if (mesh || !Number.isSafeInteger(event.bytes) || event.bytes <= 0 || event.bytes > 256 * 1024 * 1024) {
            throw new Error("The preview is too large or invalid.");
          }
          mesh = new Uint8Array(event.bytes);
        }
        if (event.type === "mesh_chunk") {
          const chunk = atob(event.data);
          if (!mesh || offset + chunk.length > mesh.length) throw new Error("Invalid preview data.");
          for (let i = 0; i < chunk.length; i++) mesh[offset++] = chunk.charCodeAt(i);
        }
        if (event.type === "result") {
          onProgress({ stage: "display" });
          if (mesh) {
            if (offset !== mesh.length) throw new Error("The preview download was incomplete.");
            return { glb: mesh.buffer as ArrayBuffer, metadata: event.metadata };
          }
          // Support a server still running the earlier stream protocol.
          const binary = atob(event.glb);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          return { glb: bytes.buffer, metadata: event.metadata };
        }
      }
      if (done) throw new Error("The connection ended before your model was ready. Please try again.");
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
