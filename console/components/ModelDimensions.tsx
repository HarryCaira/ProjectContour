"use client";

import { useState, type RefObject } from "react";
import type { Group } from "three";
import { Html, Line } from "@react-three/drei";
import type { Box3, Vector3 } from "three";

type Point = [number, number, number];

function Dimension({ start, end, tick, label, labelAt, modelRef }: {
  start: Point; end: Point; tick: Point; label: string; labelAt: Point; modelRef: RefObject<Group>;
}) {
  const [occluded, setOccluded] = useState(false);
  const centre: Point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2];
  const local = (point: Point): Point => [point[0] - centre[0], point[1] - centre[1], point[2] - centre[2]];
  return (
    <group>
      <group position={centre}>
        <Line points={[local(start), local(end)]} color="#575e49" lineWidth={1} transparent depthWrite={false} depthTest={false} renderOrder={10} />
        {[start, end].map((point, i) => (
          <Line key={i} points={[
            local([point[0] - tick[0], point[1] - tick[1], point[2] - tick[2]]),
            local([point[0] + tick[0], point[1] + tick[1], point[2] + tick[2]]),
          ]} color="#575e49" lineWidth={1} transparent depthWrite={false} depthTest={false} renderOrder={10} />
        ))}
      </group>
      <Html position={labelAt} center occlude={[modelRef]} onOcclude={setOccluded} style={{ pointerEvents: "none" }} zIndexRange={[20, 0]}>
        <div style={{ opacity: occluded ? 0.2 : 1 }} className="transition-opacity duration-200 motion-reduce:transition-none">
        <span style={{ display: "inline-block" }}
          className="whitespace-nowrap rounded-md border border-line bg-canvas/65 px-2 py-1 text-[11px] tabular-nums text-ink shadow-sm">
          {label}
        </span>
        </div>
      </Html>
    </group>
  );
}

export function ModelDimensions({ bounds, millimetres, visible, modelRef }: { bounds: Box3; millimetres: Vector3; visible: boolean; modelRef: RefObject<Group> }) {
  if (!visible) return null;
  const { min, max } = bounds;
  const gap = 0.18;
  const text = (axis: string, value: number): string => `${axis} ${value.toFixed(1)} mm`;
  return (
    <group>
      <Dimension modelRef={modelRef} start={[min.x, min.y - gap, min.z]} end={[max.x, min.y - gap, min.z]}
        tick={[0, 0.06, 0]} label={text("Width", millimetres.x)}
        labelAt={[(min.x + max.x) / 2, min.y - gap, min.z]} />
      <Dimension modelRef={modelRef} start={[min.x - gap, min.y, min.z]} end={[min.x - gap, max.y, min.z]}
        tick={[0.06, 0, 0]} label={text("Depth", millimetres.y)}
        labelAt={[min.x - gap, (min.y + max.y) / 2, min.z]} />
      <Dimension modelRef={modelRef} start={[max.x + gap, max.y + gap, min.z]} end={[max.x + gap, max.y + gap, max.z]}
        tick={[0.06, 0, 0]} label={text("Height", millimetres.z)}
        labelAt={[max.x + gap, max.y + gap, (min.z + max.z) / 2]} />
    </group>
  );
}
