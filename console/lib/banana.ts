import { Float32BufferAttribute, LatheGeometry, Vector2 } from "three";

export const BANANA_LENGTH_MM = 180;
export const VIEW_UNITS_PER_MM = 2 / 150;

/** A curved, tapered banana with brown tips, measured end-to-end in millimetres. */
export function createBananaGeometry(): LatheGeometry {
  const lengthSegments = 128;
  const profile = Array.from({ length: lengthSegments + 1 }, (_, i) => {
    const t = i / lengthSegments;
    const radius = i === 0 || i === lengthSegments ? 0 : 2 + 13 * Math.pow(Math.sin(Math.PI * t), 0.65);
    return new Vector2(radius, t * BANANA_LENGTH_MM);
  });
  const geometry = new LatheGeometry(profile, 64);
  const positions = geometry.attributes.position;
  const colours: number[] = [];
  const angle = Math.PI / 3;
  const bendRadius = BANANA_LENGTH_MM / (2 * Math.sin(angle));
  for (let i = 0; i < positions.count; i++) {
    const t = positions.getY(i) / BANANA_LENGTH_MM;
    const theta = (t * 2 - 1) * angle;
    const peelAngle = Math.atan2(positions.getZ(i), positions.getX(i));
    // Five shallow ribs run along the peel. Fade them towards the tips so
    // the shoulders remain rounded rather than looking like a bent star.
    const rib = Math.cos(5 * peelAngle + 0.12 * Math.sin(Math.PI * t));
    const ribStrength = Math.pow(Math.sin(Math.PI * t), 0.6);
    const radiusScale = 1 + 0.085 * ribStrength * rib;
    const radial = positions.getX(i) * radiusScale;
    positions.setXYZ(i,
      (bendRadius + radial) * Math.cos(theta) - bendRadius,
      (bendRadius + radial) * Math.sin(theta),
      positions.getZ(i) * radiusScale,
    );
    const tip = t < 0.035 || t > 0.965;
    const groove = Math.pow((1 - rib) / 2, 5) * ribStrength;
    const shade = (0.93 + 0.07 * Math.sin(t * Math.PI)) * (1 - 0.13 * groove);
    // Stable, sparse freckles: re-renders never change the peel pattern.
    const grain = Math.sin(Math.round(Math.cos(peelAngle) * 1000) * 12.9898
      + Math.round(Math.sin(peelAngle) * 1000) * 78.233 + t * 437.1) * 43758.5453;
    const freckle = grain - Math.floor(grain) > 0.991 && t > 0.08 && t < 0.92;
    const colour = tip ? [0.18, 0.095, 0.035]
      : freckle ? [0.37, 0.22, 0.055]
      : [0.94 * shade, 0.64 * shade, 0.065 * (1 - 0.3 * groove)];
    colours.push(...colour);
  }
  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox!;
  geometry.scale(1, BANANA_LENGTH_MM / (bounds.max.y - bounds.min.y), 1);
  geometry.translate(-(bounds.min.x + bounds.max.x) / 2, 0, -bounds.min.z);
  geometry.setAttribute("color", new Float32BufferAttribute(colours, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return geometry;
}
