"use client";

import { useId, useState } from "react";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
}

export function Slider({ label, value, min, max, step = 0.01, unit, onChange }: SliderProps) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);

  const commit = (text: string) => {
    setDraft(null);
    if (!text.trim()) return;
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) return;
    const next = Math.min(max, Math.max(min, parsed));
    if (next !== value) onChange(next);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-xs uppercase tracking-wider text-muted">{label}</label>
        <span className="flex items-baseline gap-1 text-sm tabular-nums text-ink">
          <input
            type="number"
            aria-label={`${label} value`}
            title={`Enter a value from ${min} to ${max}. Press Enter to apply or Escape to cancel.`}
            min={min}
            max={max}
            step={step}
            value={draft ?? String(value)}
            onFocus={(event) => {
              setDraft(String(value));
              event.target.select();
            }}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={(event) => commit(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") {
                event.currentTarget.value = String(value);
                event.currentTarget.blur();
              }
            }}
            className="w-20 rounded border border-transparent bg-transparent px-1 py-0.5 text-right outline-none hover:border-line focus:border-ink focus:bg-canvas [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          />
          {unit && <span>{unit}</span>}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}
