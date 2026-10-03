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
        shader.uniforms.woodlandMetresScale = { value: scale };
        shader.uniforms.coverageBounds = { value: box };
        shader.uniforms.woodColour = { value: colours.woodland };
        shader.uniforms.rockColour = { value: colours.rock };
        shader.vertexShader = 'uniform vec4 coverageBounds; varying vec2 vCoverage; varying float vSurface;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCoverage = (position.xy - coverageBounds.xy) / coverageBounds.zw; vSurface = step(0.001, normal.z);');
        shader.fragmentShader = 'uniform float woodlandEnabled; uniform float rockEnabled; uniform sampler2D coverage; uniform vec3 woodColour; uniform vec3 rockColour; varying vec2 vCoverage; varying float vSurface;\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\nvec4 cover = texture2D(coverage, vCoverage); float woodlandEdgeDistance = cover.b * .02 * max(coverageBounds.z, coverageBounds.w) * woodlandPhysicalScale; diffuseColor.rgb = mix(diffuseColor.rgb, woodColour, cover.r * woodlandEnabled * vSurface * smoothstep(0.0, .3, woodlandEdgeDistance)); diffuseColor.rgb = mix(diffuseColor.rgb, rockColour, cover.g * rockEnabled * vSurface);');
        shader.fragmentShader = 'uniform float woodlandPhysicalScale; uniform float woodlandMetresScale; uniform vec4 coverageBounds;\nfloat canopyRandom(vec2 p, float seed) { return fract(sin(dot(p,vec2(127.1,311.7))+seed)*43758.5453); }\n' + shader.fragmentShader;
        shader.fragmentShader = shader.fragmentShader.replace('#include <normal_fragment_begin>', `#include <normal_fragment_begin>
          vec2 woodXY = (vCoverage * coverageBounds.zw + coverageBounds.xy) * woodlandPhysicalScale;
          float canopyScale = clamp(woodlandMetresScale * woodlandPhysicalScale / .02, .5, 2.0);
          float spacing = .9 * canopyScale;
          float woodHeight = 0.0;
          for (int ix = -2; ix <= 2; ix++) {
            for (int iy = -2; iy <= 2; iy++) {
              vec2 cell = floor(woodXY/spacing) + vec2(float(ix), float(iy));
              vec2 centre = (cell + .5 + .9*(vec2(canopyRandom(cell,1.0),canopyRandom(cell,2.0))-.5))*spacing;
              float clusters = .5+.5*sin(centre.x/canopyScale*.65)*sin(centre.y/canopyScale*.53);
              if (canopyRandom(cell,7.0) < .65+.3*clusters) {
                float radius = (.28+.38*canopyRandom(cell,3.0))*canopyScale;
                float aspect = .8+.4*canopyRandom(cell,4.0);
                float angle = 6.2831853*canopyRandom(cell,5.0);
                vec2 delta = woodXY-centre;
                vec2 local = vec2(cos(angle)*delta.x+sin(angle)*delta.y, -sin(angle)*delta.x+cos(angle)*delta.y);
                float distance = length(local/vec2(radius*aspect,radius));
                float height = (.3+.4*canopyRandom(cell,6.0))*canopyScale*2.0/150.0;
                woodHeight = max(woodHeight, height*sqrt(max(0.0,1.0-distance*distance)));
              }
            }
          }
          float edgeDistance = texture2D(coverage, vCoverage).b * .02 * max(coverageBounds.z,coverageBounds.w) * woodlandPhysicalScale;
          float edgeTaper = smoothstep(0.0, .85*canopyScale, edgeDistance);
          woodHeight *= texture2D(coverage,vCoverage).r * woodlandEnabled * vSurface * edgeTaper;
          vec3 dpX = dFdx(-vViewPosition), dpY = dFdy(-vViewPosition);
          vec3 rX = cross(dpY, normal), rY = cross(normal, dpX);
          float det = dot(dpX, rX);
          if (abs(det) > 1e-12) normal = normalize(abs(det)*normal - sign(det)*(dFdx(woodHeight)*rX + dFdy(woodHeight)*rY));
        `);
      };
      // Re-enabling an overlay must not reuse uniforms holding a disposed texture.
      const baseKey = previousKey.call(material);
      material.customProgramCacheKey = () => `${baseKey}:contour-landcover-canopy-v2:${texture?.uuid}:${box.toArray().join(",")}`;
      material.needsUpdate = true;
      restore.push(() => { material.onBeforeCompile = previous; material.customProgramCacheKey = previousKey; material.needsUpdate = true; });
    }
  });
  return () => restore.forEach(fn => fn());
}
