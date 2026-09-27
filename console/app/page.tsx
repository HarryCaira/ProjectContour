import { Scene } from "@/components/Scene";
import { EditorPanel } from "@/components/EditorPanel";

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
