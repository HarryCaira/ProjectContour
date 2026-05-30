import { Scene } from "@/components/viewer/Scene";
import { EditorPanel } from "@/components/editor/EditorPanel";

export default function Page() {
  return (
    <main className="h-screen w-screen flex">
      <div className="flex-1 min-w-0">
        <Scene />
      </div>
      <EditorPanel />
    </main>
  );
}
