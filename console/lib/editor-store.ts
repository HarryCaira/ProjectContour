import { create } from "zustand";
import { defaultSettings, type Settings, type Source } from "@/lib/settings";

interface EditorState {
  source: Source | null;
  settings: Settings | null;
  setSource: (source: Source) => void;
  updateSettings: (patch: (s: Settings) => Settings) => void;
  setVerticalExaggeration: (n: number) => void;
  reset: () => void;
}

/** Physical controls belong to the serialisable settings used for export. */
export const useEditorStore = create<EditorState>((set) => ({
  source: null,
  settings: null,
  setSource: (source) => set({ source, settings: defaultSettings(source) }),
  updateSettings: (patch) =>
    set((state) => ({ settings: state.settings ? patch(state.settings) : null })),
  setVerticalExaggeration: (verticalExaggeration) =>
    set((state) => ({
      settings: state.settings ? {
        ...state.settings,
        terrain: { ...state.settings.terrain, verticalExaggeration },
      } : null,
    })),
  reset: () => set({ source: null, settings: null }),
}));
