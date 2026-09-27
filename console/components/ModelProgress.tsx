"use client";

import { useEffect, useState } from "react";

const STAGES = [
  { id: "frame", title: "Finding your landscape", detail: "Fitting a hexagonal frame around your route." },
  { id: "terrain", title: "Gathering elevation", detail: "Loading the hills, valleys and peaks around your route." },
  { id: "water", title: "Mapping lakes and water", detail: "Finding water features within your landscape." },
  { id: "land", title: "Shaping the terrain", detail: "Turning elevation data into your 3D landscape." },
  { id: "details", title: "Adding your route", detail: "Building the raised trail, water surfaces and display base." },
  { id: "preview", title: "Finishing the model", detail: "Applying materials and sending the model to your browser." },
  { id: "display", title: "Preparing your preview", detail: "Positioning the model and getting the 3D view ready." },
];

export function ModelProgress({ stage }: { stage: string }) {
  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  const index = STAGES.findIndex((item) => item.id === stage);
  const current = STAGES[index];
  const completed = Math.max(0, index);
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-6">
      <div className="w-full max-w-sm rounded-2xl border border-line bg-canvas/95 p-7 shadow-lg backdrop-blur-sm">
        <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted">
          <span>Creating your landscape</span>
          <span className="tabular-nums">{elapsed}s elapsed</span>
        </div>
        <div aria-live="polite" aria-atomic="true">
          <h2 className="mt-5 text-lg font-medium tracking-tightish text-ink">{current?.title ?? "Starting your model"}</h2>
          <p className="mt-2 min-h-10 text-sm leading-relaxed text-muted">{current?.detail ?? "Connecting to the model builder."}</p>
        </div>
        <div role="progressbar" aria-label="Model preparation" aria-valuemin={0} aria-valuemax={STAGES.length}
          aria-valuenow={completed} aria-valuetext={current ? `Step ${index + 1} of ${STAGES.length}: ${current.title}` : "Connecting"}
          className="mt-6 h-1.5 overflow-hidden rounded-full bg-line">
          <div className="h-full rounded-full bg-accent transition-[width] duration-500 motion-reduce:transition-none"
            style={{ width: `${(completed / STAGES.length) * 100}%` }} />
        </div>
        <div className="mt-3 flex items-center justify-between text-xs text-muted">
          <span>{current ? `Step ${index + 1} of ${STAGES.length}` : "Getting started"}</span>
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse motion-reduce:animate-none" />Working</span>
        </div>
        <p className="mt-5 border-t border-line pt-4 text-xs leading-relaxed text-muted">
          {elapsed >= 30 ? "Still working. Detailed landscapes can take a little longer. You can leave this tab open." : "The first model takes longer while we gather the landscape data."}
        </p>
      </div>
    </div>
  );
}
