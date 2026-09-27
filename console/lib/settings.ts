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
  rotationDegrees: z.number().min(0).lt(60).default(0),
});

export const PhysicalSchema = z.object({
  sizeMm: z.number().positive().default(150),
  resolutionMm: z.number().positive().default(0.2),
});

export const DEFAULT_COLOURS = { terrain: "#7a8060", water: "#6a8aa0", route: "#c44545" };
const hexColour = z.string().regex(/^#[0-9a-fA-F]{6}$/);
export const ColoursSchema = z.object({
  terrain: hexColour.default(DEFAULT_COLOURS.terrain),
  water: hexColour.default(DEFAULT_COLOURS.water),
  route: hexColour.default(DEFAULT_COLOURS.route),
});
export type ModelColours = z.infer<typeof ColoursSchema>;

export const StyleRefSchema = z.object({
  name: z.literal("monochrome-biome").default("monochrome-biome"),
  colours: ColoursSchema.default({}),
});

export const TerrainSettingsSchema = z.object({
  verticalExaggeration: z.number().positive().default(1.5),
});

export const WaterBiomeSchema = z.object({
  enabled: z.boolean().default(true),
  depthFraction: z.number().min(0).max(0.5).default(0.07),
});

export const BiomesSchema = z.object({
  water: WaterBiomeSchema.default({}),
});

export const RouteSettingsSchema = z.object({
  enabled: z.boolean().default(true),
  widthMm: z.number().positive().default(0.5),
  heightAboveTerrainMm: z.number().min(0).default(0.5),
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
