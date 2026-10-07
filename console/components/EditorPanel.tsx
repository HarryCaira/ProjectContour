"use client";

import { useState } from "react";
import { useEditorStore } from "@/lib/editor-store";
import { DEFAULT_COLOURS } from "@/lib/settings";
import { useExport, useFeatureCoverage } from "@/lib/hooks";
import { UploadButton } from "./UploadButton";
import { Slider } from "./Slider";

export function EditorPanel() {
  const [scaleUniformly, setScaleUniformly] = useState(false);
  const settings = useEditorStore((s) => s.settings);
  const updateSettings = useEditorStore((s) => s.updateSettings);
  const setVerticalExaggeration = useEditorStore((s) => s.setVerticalExaggeration);

  const exportMut = useExport();
  const coverage = useFeatureCoverage(settings);

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
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" className="accent-accent" checked={settings.biomes.snow?.enabled ?? false}
                  onChange={e => updateSettings(s => ({ ...s, biomes: { ...s.biomes, snow: { snowline: s.biomes.snow?.snowline ?? .72, enabled: e.target.checked } } }))} />
                Snow cap
              </label>
              {settings.biomes.snow?.enabled && <>
                <Slider label="Snowline" value={Math.round(settings.biomes.snow.snowline * 100)} min={0} max={100} step={1} unit="%"
                  onChange={value => updateSettings(s => ({ ...s, biomes: { ...s.biomes, snow: { ...s.biomes.snow, snowline: value / 100 } } }))} />
                <p className="text-xs text-muted">Stylised snow. Snowline is relative to the landscape’s height.</p>
              </> }
              {([['woodland', 'Woodland'], ['rock', 'Rock'], ['roads', 'Roads'], ['buildings', 'Buildings']] as const).filter(([key]) => coverage.data?.[key] === true).map(([key, label]) => (
                <label key={key} className="flex items-center gap-2 text-sm">
                  <input type="checkbox" className="accent-accent" checked={settings.biomes[key]?.enabled ?? false}
                    onChange={e => { const enabled = e.target.checked; updateSettings(s => ({ ...s, biomes: { ...s.biomes, [key]: { enabled } } })); }} />
                  {label}
                </label>
              ))}
              {coverage.isError && <button type="button" className="text-xs text-muted underline"
                onClick={() => void coverage.refetch()}>Retry map feature check</button>}
            </section>

            <Divider />

            <section className="space-y-4">
              <h2 className="text-xs uppercase tracking-wider text-muted">Model</h2>
              <fieldset className="space-y-2">
                <legend className="text-xs uppercase tracking-wider text-muted">Physical size</legend>
                <div className="grid grid-cols-2 gap-2">
                  {([{ label: "Medium", sizeMm: 100 }, { label: "Large", sizeMm: 150 }] as const).map(({ label, sizeMm }) => (
                    <label key={sizeMm} className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm ${settings.physical.sizeMm === sizeMm ? "border-ink text-ink" : "border-line text-muted"}`}>
                      <input type="radio" name="physical-size" value={sizeMm}
                        checked={settings.physical.sizeMm === sizeMm}
                        onChange={() => updateSettings((s) => ({ ...s, physical: { ...s.physical, sizeMm } }))}
                        className="accent-accent" />
                      <span>{label}<span className="block text-xs text-muted">{sizeMm} mm</span></span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </section>

            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-xs uppercase tracking-wider text-muted">Colours</h2>
                <button type="button" className="text-xs text-muted hover:text-ink"
                  onClick={() => updateSettings((s) => ({ ...s, style: { ...s.style, colours: { ...DEFAULT_COLOURS } } }))}>
                  Reset
                </button>
              </div>
              {(["route", "terrain", "water", "woodland", "rock", "snow", "roads", "buildings"] as const).filter((part) => {
                if (part === "terrain") return true;
                if (part === "route") return settings.route.enabled;
                if (part === "snow") return settings.biomes.snow.enabled && settings.biomes.snow.snowline < 1;
                return settings.biomes[part].enabled && coverage.data?.[part] === true;
              }).map((part) => {
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
                onClick={() => exportMut.mutate(settings, { onSuccess: triggerDownload })}
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
