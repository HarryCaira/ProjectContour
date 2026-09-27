"use client";

import { useState } from "react";
import { useEditorStore } from "@/lib/editor-store";
import { DEFAULT_COLOURS } from "@/lib/settings";
import { useExport } from "@/lib/hooks";
import { UploadButton } from "./UploadButton";
import { Slider } from "./Slider";

export function EditorPanel() {
  const [scaleUniformly, setScaleUniformly] = useState(false);
  const settings = useEditorStore((s) => s.settings);
  const updateSettings = useEditorStore((s) => s.updateSettings);
  const setVerticalExaggeration = useEditorStore((s) => s.setVerticalExaggeration);

  const exportMut = useExport();

  return (
    <aside className="w-[320px] shrink-0 border-l border-line bg-canvas h-full flex flex-col">
      <div className="px-6 py-5 border-b border-line">
        <h1 className="text-base font-medium tracking-tightish">ProjectContour</h1>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
        <section className="space-y-3">
          <h2 className="text-xs uppercase tracking-wider text-muted">Route</h2>
          <UploadButton />
          {settings && (
            <div className="space-y-4 pt-2">
              <Slider
                label="Route width"
                value={settings.route.widthMm}
                min={0.4}
                max={scaleUniformly ? 5 : 6}
                step={0.1}
                unit="mm"
                onChange={(widthMm) =>
                  updateSettings((s) => ({ ...s, route: { ...s.route, widthMm, ...(scaleUniformly ? { heightAboveTerrainMm: widthMm } : {}) } }))
                }
              />
              <Slider
                label="Height above terrain"
                value={settings.route.heightAboveTerrainMm}
                min={scaleUniformly ? 0.4 : 0.2}
                max={5}
                step={0.1}
                unit="mm"
                onChange={(heightAboveTerrainMm) =>
                  updateSettings((s) => ({ ...s, route: { ...s.route, heightAboveTerrainMm, ...(scaleUniformly ? { widthMm: heightAboveTerrainMm } : {}) } }))
                }
              />
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={scaleUniformly} className="accent-accent"
                  onChange={(event) => {
                    const linked = event.target.checked;
                    setScaleUniformly(linked);
                    if (linked) updateSettings((s) => {
                      const size = Math.min(5, Math.max(0.4, s.route.widthMm));
                      return { ...s, route: { ...s.route, widthMm: size, heightAboveTerrainMm: size } };
                    });
                  }} />
                Uniform scale
              </label>
            </div>
          )}
        </section>

        {settings && (
          <>
            <Divider />

            <section className="space-y-4">
              <h2 className="text-xs uppercase tracking-wider text-muted">Landscape</h2>
              <Slider
                label="Terrain exaggeration"
                value={settings.terrain.verticalExaggeration}
                min={0.5}
                max={5}
                step={0.05}
                onChange={setVerticalExaggeration}
              />
            </section>

            <Divider />

            <section className="space-y-4">
              <h2 className="text-xs uppercase tracking-wider text-muted">Model size</h2>
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
              <div className="flex items-center justify-between">
                <h2 className="text-xs uppercase tracking-wider text-muted">Colours</h2>
                <button type="button" className="text-xs text-muted hover:text-ink"
                  onClick={() => updateSettings((s) => ({ ...s, style: { ...s.style, colours: { ...DEFAULT_COLOURS } } }))}>
                  Reset
                </button>
              </div>
              {(["route", "terrain", "water"] as const).map((part) => {
                const colour = settings.style.colours?.[part] ?? DEFAULT_COLOURS[part];
                return (
                  <label key={part} className="flex items-center justify-between gap-3 text-sm">
                    <span className="capitalize">{part}</span>
                    <span className="flex items-center gap-2">
                      <span className="text-xs font-mono text-muted uppercase">{colour}</span>
                      <input type="color" aria-label={`${part} colour`} value={colour}
                        className="h-8 w-10 cursor-pointer rounded border border-line bg-transparent p-0.5"
                        onChange={(event) => {
                          const value = event.target.value;
                          updateSettings((s) => ({ ...s, style: { ...s.style,
                            colours: { ...DEFAULT_COLOURS, ...s.style.colours, [part]: value },
                          } }));
                        }} />
                    </span>
                  </label>
                );
              })}
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
