import { DEFAULT_COLOURS, type Settings } from "./settings.ts";

/** Route dimensions, colours and relief are applied locally by the viewer. */
export function previewSettings(settings: Settings): Settings {
  return {
    ...settings,
    physical: { ...settings.physical, resolutionMm: 0.1 },
    style: { ...settings.style, colours: DEFAULT_COLOURS },
    terrain: { ...settings.terrain, maximumSourceDetail: settings.terrain.maximumSourceDetail ?? false, verticalExaggeration: 1 },
    route: { ...settings.route, widthMm: 1, heightAboveTerrainMm: 1 },
  };
}
