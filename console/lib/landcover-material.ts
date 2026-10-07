import { Color, Group, Mesh, MeshStandardMaterial, Vector4, type Texture } from "three";
import type { LandCover } from "./api";

/** Colour the existing surface; geometry, elevation, and STL export stay intact. */
export function applyLandCover(root: Group, texture: Texture | null, bounds: LandCover["bounds"], sizeMm: number, colours = { woodland: new Color("#365d38"), rock: new Color("#96928a") }, physicalScale = { value: 1 }, enabled = { woodland: { value: 1 }, rock: { value: 1 } }): () => void {
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
        shader.uniforms.woodlandEnabled = enabled.woodland;
        shader.uniforms.rockEnabled = enabled.rock;
        shader.uniforms.coverage = { value: texture };
        shader.uniforms.woodlandPhysicalScale = physicalScale;
        shader.uniforms.coverageBounds = { value: box };
        shader.uniforms.woodColour = { value: colours.woodland };
        shader.uniforms.rockColour = { value: colours.rock };
        shader.vertexShader = 'uniform vec4 coverageBounds; varying vec2 vCoverage; varying float vSurface;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCoverage = (position.xy - coverageBounds.xy) / coverageBounds.zw; vSurface = step(0.001, normal.z);');
        shader.fragmentShader = 'uniform float woodlandEnabled; uniform float rockEnabled; uniform sampler2D coverage; uniform vec3 woodColour; uniform vec3 rockColour; varying vec2 vCoverage; varying float vSurface;\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\nvec4 cover = texture2D(coverage, vCoverage); float woodlandEdgeDistance = cover.b * .02 * max(coverageBounds.z, coverageBounds.w) * woodlandPhysicalScale; diffuseColor.rgb = mix(diffuseColor.rgb, woodColour, cover.r * woodlandEnabled * vSurface * smoothstep(0.0, .3, woodlandEdgeDistance)); diffuseColor.rgb = mix(diffuseColor.rgb, rockColour, cover.g * rockEnabled * vSurface);');
        shader.fragmentShader = 'uniform float woodlandPhysicalScale; uniform vec4 coverageBounds;\n' + shader.fragmentShader;
      };
      // Re-enabling an overlay must not reuse uniforms holding a disposed texture.
      const baseKey = previousKey.call(material);
      material.customProgramCacheKey = () => `${baseKey}:contour-landcover-colour-v3:${texture?.uuid}:${box.toArray().join(",")}`;
      material.needsUpdate = true;
      restore.push(() => { material.onBeforeCompile = previous; material.customProgramCacheKey = previousKey; material.needsUpdate = true; });
    }
  });
  return () => restore.forEach(fn => fn());
}
