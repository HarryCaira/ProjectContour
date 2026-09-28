import { Color, Group, Mesh, MeshStandardMaterial, Vector4, type Texture } from "three";
import type { LandCover } from "./api";

/** Colour the existing surface; geometry, elevation, and STL export stay intact. */
export function applyLandCover(root: Group, texture: Texture | null, bounds: LandCover["bounds"], sizeMm: number): () => void {
  const scale = sizeMm / Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
  const box = new Vector4(bounds[0] * scale, bounds[1] * scale,
    (bounds[2] - bounds[0]) * scale, (bounds[3] - bounds[1]) * scale);
  const restore: (() => void)[] = [];
  root.traverse(object => {
    if (!(object instanceof Mesh) || object.name !== "land") return;
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      if (!(material instanceof MeshStandardMaterial)) continue;
      const previous = material.onBeforeCompile;
      const previousKey = material.customProgramCacheKey;
      material.onBeforeCompile = (shader, renderer) => {
        previous.call(material, shader, renderer);
        shader.uniforms.coverage = { value: texture };
        shader.uniforms.coverageBounds = { value: box };
        shader.uniforms.woodColour = { value: new Color("#365d38") };
        shader.uniforms.rockColour = { value: new Color("#96928a") };
        shader.vertexShader = 'uniform vec4 coverageBounds; varying vec2 vCoverage; varying float vSurface;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCoverage = (position.xy - coverageBounds.xy) / coverageBounds.zw; vSurface = step(0.001, normal.z);');
        shader.fragmentShader = 'uniform sampler2D coverage; uniform vec3 woodColour; uniform vec3 rockColour; varying vec2 vCoverage; varying float vSurface;\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\nvec4 cover = texture2D(coverage, vCoverage); diffuseColor.rgb = mix(diffuseColor.rgb, woodColour, cover.r * vSurface); diffuseColor.rgb = mix(diffuseColor.rgb, rockColour, cover.g * vSurface);');
      };
      // Re-enabling an overlay must not reuse uniforms holding a disposed texture.
      const baseKey = previousKey.call(material);
      material.customProgramCacheKey = () => `${baseKey}:contour-landcover:${texture?.uuid}:${box.toArray().join(",")}`;
      material.needsUpdate = true;
      restore.push(() => { material.onBeforeCompile = previous; material.customProgramCacheKey = previousKey; material.needsUpdate = true; });
    }
  });
  return () => restore.forEach(fn => fn());
}
