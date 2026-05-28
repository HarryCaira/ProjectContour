import { create } from "zustand";
import { defaultSettings, type Settings, type Source } from "@/lib/schema/settings";

/**
 * Editor state. Splits real-time transform values (scale, vertical exaggeration)
 * from the full Settings object, because:
 *  - Transforms apply client-side every frame (Zustand selector subscribes the viewer)
 *  - Settings drive server-side mesh regeneration (debounced commits)
 *
 * `settings` is the source of truth for the model; `transforms` is what the
 * viewer currently *displays*. They diverge while a slider is being dragged.
 */

export interface RealtimeTransforms {
  /** Multiplier on Z to exaggerate terrain relief. 1.0 = no exaggeration. */
  verticalExaggeration: number;
  /** Uniform scale multiplier applied to the model root. */
  modelScale: number;
}

interface EditorState {
  source: Source | null;
  settings: Settings | null;
  transforms: RealtimeTransforms;

  setSource: (source: Source) => void;
  updateSettings: (patch: (s: Settings) => Settings) => void;
  setVerticalExaggeration: (n: number) => void;
  setModelScale: (n: number) => void;
  reset: () => void;
}

const DEFAULT_TRANSFORMS: RealtimeTransforms = {
  verticalExaggeration: 1.5,
  modelScale: 1.0,
};

export const useEditorStore = create<EditorState>((set) => ({
  source: null,
  settings: null,
  transforms: DEFAULT_TRANSFORMS,

  setSource: (source) =>
    set(() => ({
      source,
      settings: defaultSettings(source),
      transforms: DEFAULT_TRANSFORMS,
    })),

  updateSettings: (patch) =>
    set((state) => ({ settings: state.settings ? patch(state.settings) : null })),

  setVerticalExaggeration: (n) =>
    set((state) => ({ transforms: { ...state.transforms, verticalExaggeration: n } })),

  setModelScale: (n) =>
    set((state) => ({ transforms: { ...state.transforms, modelScale: n } })),

  reset: () => set({ source: null, settings: null, transforms: DEFAULT_TRANSFORMS }),
}));
