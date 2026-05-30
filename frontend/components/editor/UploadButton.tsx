"use client";

import { useRef } from "react";
import { useUploadGpx } from "@/lib/api/hooks";
import { useEditorStore } from "@/lib/state/editor-store";

export function UploadButton() {
  const inputRef = useRef<HTMLInputElement>(null);
  const upload = useUploadGpx();
  const setSource = useEditorStore((s) => s.setSource);

  const onPick = async (file: File) => {
    const res = await upload.mutateAsync(file);
    setSource({ type: "gpx", id: res.id, sha256: res.sha256 });
  };

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".gpx,application/gpx+xml,application/octet-stream"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={upload.isPending}
        className="px-4 py-2 text-sm tracking-tightish rounded-md border border-line bg-canvas text-ink hover:border-ink transition-colors disabled:opacity-50"
      >
        {upload.isPending ? "Uploading…" : "Upload GPX"}
      </button>
      {upload.isError && (
        <p className="mt-2 text-xs text-accentRoute">{(upload.error as Error).message}</p>
      )}
    </div>
  );
}
