"use client";

import { useEditorStore } from "@/lib/state/editor-store";
import { useExport } from "@/lib/api/hooks";
import { UploadButton } from "./UploadButton";
import { Slider } from "./Slider";

export function EditorPanel() {
  const settings = useEditorStore((s) => s.settings);
  const transforms = useEditorStore((s) => s.transforms);
  const updateSettings = useEditorStore((s) => s.updateSettings);
  const setVerticalExaggeration = useEditorStore((s) => s.setVerticalExaggeration);
  const setModelScale = useEditorStore((s) => s.setModelScale);

  const exportMut = useExport();

  return (
    <aside className="w-[320px] shrink-0 border-l border-line bg-canvas h-full flex flex-col">
      <div className="px-6 py-5 border-b border-line">
        <h1 className="text-base font-medium tracking-tightish">ProjectContour</h1>
        <p className="text-xs text-muted mt-1">Turn a route into a printable landscape.</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
        <section className="space-y-3">
          <h2 className="text-xs uppercase tracking-wider text-muted">Route</h2>
          <UploadButton />
        </section>

        {settings && (
          <>
            <Divider />

            <section className="space-y-4">
              <h2 className="text-xs uppercase tracking-wider text-muted">Real-time</h2>
              <Slider
                label="Model scale"
                value={transforms.modelScale}
                min={0.25}
                max={3}
                step={0.01}
                onChange={setModelScale}
              />
              <Slider
                label="Vertical exaggeration"
                value={transforms.verticalExaggeration}
                min={0.5}
                max={5}
                step={0.05}
                onChange={setVerticalExaggeration}
              />
            </section>

            <Divider />

            <section className="space-y-4">
              <h2 className="text-xs uppercase tracking-wider text-muted">Topology</h2>
              <Slider
                label="Physical size"
                value={settings.physical.sizeMm}
                min={50}
                max={300}
                step={1}
                unit="mm"
                onChange={(sizeMm) =>
                  updateSettings((s) => ({ ...s, physical: { ...s.physical, sizeMm } }))
                }
              />
              <Slider
                label="Print resolution"
                value={settings.physical.resolutionMm}
                min={0.05}
                max={1.0}
                step={0.01}
                unit="mm"
                onChange={(resolutionMm) =>
                  updateSettings((s) => ({ ...s, physical: { ...s.physical, resolutionMm } }))
                }
              />
              <Slider
                label="Frame rotation"
                value={settings.framing.rotationDegrees}
                min={0}
                max={59.99}
                step={1}
                unit="°"
                onChange={(rotationDegrees) =>
                  updateSettings((s) => ({
                    ...s,
                    framing: { ...s.framing, rotationDegrees },
                  }))
                }
              />
            </section>

            <Divider />

            <section className="space-y-3">
              <h2 className="text-xs uppercase tracking-wider text-muted">Export</h2>
              <button
                type="button"
                onClick={() => exportMut.mutateAsync(settings).then(triggerDownload)}
                disabled={exportMut.isPending}
                className="w-full px-4 py-2 text-sm tracking-tightish rounded-md border border-line bg-canvas text-ink hover:border-ink transition-colors disabled:opacity-50"
              >
                {exportMut.isPending ? "Preparing…" : "Download STL kit"}
              </button>
              {exportMut.isError && (
                <p className="text-xs text-accentRoute">{(exportMut.error as Error).message}</p>
              )}
            </section>
          </>
        )}
      </div>
    </aside>
  );
}

function Divider() {
  return <div className="border-t border-line" />;
}

function triggerDownload(blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "contour-kit.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
