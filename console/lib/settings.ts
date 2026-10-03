import { z } from "zod";

// Mirrors backend/contour/settings.py — every field, every default,
// every validator. The wire format uses camelCase; both the backend and the
// frontend speak it natively.

export const SourceSchema = z.object({
  type: z.literal("gpx").default("gpx"),
  id: z.string(),
  sha256: z.string(),
});

export const FramingSchema = z.object({
  shape: z.literal("hex").default("hex"),
  paddingRatio: z.number().min(0).max(1).default(0.15),
});

export const PhysicalSchema = z.object({
  sizeMm: z.number().positive().default(100),
  resolutionMm: z.number().positive().default(0.2),
});

export const DEFAULT_COLOURS = { roads: "#555454", buildings: "#c5b6a2", snow: "#f3f4ef", terrain: "#7a8060", water: "#6a8aa0", route: "#c44545", woodland: "#365d38", rock: "#96928a" };
const hexColour = z.string().regex(/^#[0-9a-fA-F]{6}$/);
export const ColoursSchema = z.object({
  roads: hexColour.default(DEFAULT_COLOURS.roads),
  buildings: hexColour.default(DEFAULT_COLOURS.buildings),
  snow: hexColour.default(DEFAULT_COLOURS.snow),
  terrain: hexColour.default(DEFAULT_COLOURS.terrain),
  water: hexColour.default(DEFAULT_COLOURS.water),
  route: hexColour.default(DEFAULT_COLOURS.route),
  woodland: hexColour.default(DEFAULT_COLOURS.woodland),
  rock: hexColour.default(DEFAULT_COLOURS.rock),
});
export type ModelColours = z.infer<typeof ColoursSchema>;

export const StyleRefSchema = z.object({
  name: z.literal("monochrome-biome").default("monochrome-biome"),
  colours: ColoursSchema.default({}),
});

export const DetailSettingsSchema = z.object({
  forceSourceZoom: z.boolean().default(false),
  sampleMm: z.number().min(0.01).max(2).default(0.1),
  toleranceMm: z.number().min(0.001).max(0.5).default(0.0125),
  maxZoom: z.number().int().min(1).max(15).default(15),
  waterZoom: z.number().int().min(1).max(16).default(14),
  smoothingSigma: z.number().min(0).max(8).default(2.0),
  smoothingMaxMm: z.number().min(0).max(1).default(0.05),
  shorelinePixels: z.number().min(0.1).max(32).default(8.0),
  routeToleranceMm: z.number().min(0.001).max(0.5).default(0.0125),
  sourceRouteToleranceM: z.number().min(0.001).max(5).default(0.025),
  maxVertices: z.number().int().min(1000).max(5000000).default(1200000),
  maxReferencePoints: z.number().int().min(1000).max(32000000).default(8000000),
  maxPasses: z.number().int().min(1).max(64).default(16),
  maxTiles: z.number().int().min(1).max(4096).default(1024),
  maxRoutePoints: z.number().int().min(1000).max(1000000).default(200000),
});
export type DetailSettings = z.infer<typeof DetailSettingsSchema>;

export const TerrainSettingsSchema = z.object({
  detail: DetailSettingsSchema.optional(),
  maximumSourceDetail: z.boolean().default(false),
  verticalExaggeration: z.number().positive().default(1.5),
});

export const WaterBiomeSchema = z.object({
  enabled: z.boolean().default(true),
  depthFraction: z.number().min(0).max(0.5).default(0.07),
});

export const BiomesSchema = z.object({
  roads: z.object({ enabled: z.boolean().default(true) }).default({}),
  buildings: z.object({ enabled: z.boolean().default(true) }).default({}),
  snow: z.object({ enabled: z.boolean().default(false), snowline: z.number().min(0).max(1).default(0.72) }).default({}),
  rock: z.object({ enabled: z.boolean().default(true) }).default({}),
  woodland: z.object({ enabled: z.boolean().default(true) }).default({}),
  water: WaterBiomeSchema.default({}),
});

export const RouteSettingsSchema = z.object({
  enabled: z.boolean().default(true),
  widthMm: z.number().positive().default(1),
  heightAboveTerrainMm: z.number().min(0).default(1),
});

export const PlinthSchema = z.object({
  enabled: z.boolean().default(true),
  style: z.literal("default").default("default"),
});

export const SettingsSchema = z.object({
  schemaVersion: z.literal(1).default(1),
  source: SourceSchema,
  framing: FramingSchema.default({}),
  physical: PhysicalSchema.default({}),
  style: StyleRefSchema.default({}),
  terrain: TerrainSettingsSchema.default({}),
  biomes: BiomesSchema.default({}),
  route: RouteSettingsSchema.default({}),
  plinth: PlinthSchema.default({}),
});

export type Settings = z.infer<typeof SettingsSchema>;
export type Source = z.infer<typeof SourceSchema>;

/** Build a Settings with all defaults applied, given a source GPX reference. */
export function defaultSettings(source: Source): Settings {
  return SettingsSchema.parse({ schemaVersion: 1, source });
}
