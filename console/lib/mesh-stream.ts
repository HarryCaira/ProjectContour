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
        if (event.type === "result") {
          onProgress({ stage: "display" });
          const bytes = Uint8Array.from(atob(event.glb), (character) => character.charCodeAt(0));
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
