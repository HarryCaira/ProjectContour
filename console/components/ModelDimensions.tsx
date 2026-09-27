"use client";

import { Html, Line } from "@react-three/drei";
import type { Box3, Vector3 } from "three";

type Point = [number, number, number];

function Dimension({ start, end, tick, label, labelAt }: {
  start: Point; end: Point; tick: Point; label: string; labelAt: Point;
}) {
  const ends = [start, end];
  return (
    <group>
      <Line points={[start, end]} color="#575e49" lineWidth={1} depthTest={false} renderOrder={10} />
      {ends.map((point, i) => (
        <Line key={i} points={[
          [point[0] - tick[0], point[1] - tick[1], point[2] - tick[2]],
          [point[0] + tick[0], point[1] + tick[1], point[2] + tick[2]],
        ]} color="#575e49" lineWidth={1} depthTest={false} renderOrder={10} />
      ))}
      <Html position={labelAt} center style={{ pointerEvents: "none" }} zIndexRange={[20, 0]}>
        <span className="whitespace-nowrap rounded-md border border-line bg-canvas/65 px-2 py-1 text-[11px] tabular-nums text-ink shadow-sm">
          {label}
        </span>
      </Html>
    </group>
  );
}

export function ModelDimensions({ bounds, millimetres }: { bounds: Box3; millimetres: Vector3 }) {
  const { min, max } = bounds;
  const gap = 0.18;
  const text = (axis: string, value: number): string => `${axis} ${value.toFixed(1)} mm`;
  return (
    <group>
      <Dimension start={[min.x, min.y - gap, min.z]} end={[max.x, min.y - gap, min.z]}
        tick={[0, 0.06, 0]} label={text("Width", millimetres.x)}
        labelAt={[(min.x + max.x) / 2, min.y - gap, min.z]} />
      <Dimension start={[min.x - gap, min.y, min.z]} end={[min.x - gap, max.y, min.z]}
        tick={[0.06, 0, 0]} label={text("Depth", millimetres.y)}
        labelAt={[min.x - gap, (min.y + max.y) / 2, min.z]} />
      <Dimension start={[max.x + gap, max.y + gap, min.z]} end={[max.x + gap, max.y + gap, max.z]}
        tick={[0.06, 0, 0]} label={text("Height", millimetres.z)}
        labelAt={[max.x + gap, max.y + gap, (min.z + max.z) / 2]} />
    </group>
  );
}
