import { Color, Group, Mesh, MeshStandardMaterial, Vector4 } from 'three';

/** Local preview of the same relief/noise/slope field used by snow.py. */
export function applySnow(root: Group, state: { enabled: { value: number }; line: { value: number }; colour: Color }): () => void {
  const undo: (() => void)[] = [];
  root.traverse(object => {
    if (!(object instanceof Mesh) || object.name !== 'land') return;
    const geometry = object.geometry;
    geometry.computeBoundingBox();
    const bounds = geometry.boundingBox!;
    const p = geometry.attributes.position, n = geometry.attributes.normal;
    let low = Infinity, high = -Infinity;
    for (let i = 0; i < p.count; i++) if (n.getZ(i) > .001) {
      low = Math.min(low, p.getZ(i)); high = Math.max(high, p.getZ(i));
    }
    const box = new Vector4(bounds.min.x, bounds.min.y, bounds.max.x-bounds.min.x, bounds.max.y-bounds.min.y);
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      if (!(material instanceof MeshStandardMaterial)) continue;
      const previous = material.onBeforeCompile, previousKey = material.customProgramCacheKey;
      const baseKey = previousKey.call(material);
      material.onBeforeCompile = (shader, renderer) => {
        previous.call(material, shader, renderer);
        Object.assign(shader.uniforms, { snowEnabled: state.enabled, snowLine: state.line,
          snowColour: { value: state.colour }, snowBounds: { value: box }, snowLow: { value: low }, snowRange: { value: high-low } });
        shader.vertexShader = 'varying vec3 vSnowPosition; varying float vSnowSurface;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvSnowPosition = position; vSnowSurface = step(.001, normal.z);');
        shader.fragmentShader = `uniform float snowEnabled, snowLine, snowLow, snowRange;
          uniform vec3 snowColour; uniform vec4 snowBounds;
          varying vec3 vSnowPosition; varying float vSnowSurface;\n` + shader.fragmentShader;
        // Apply after land-cover colour so the cap wins over rock and woodland.
        shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>', `
          vec2 snowUV = (vSnowPosition.xy-snowBounds.xy)/snowBounds.zw;
          float sx = snowUV.x, sy = snowUV.y;
          float snowNoise = .028*sin(sx*31.0+sy*7.0)*cos(sy*27.0-sx*5.0)+.012*sin(sx*83.0-sy*61.0);
          vec3 snowNormal = normalize(cross(dFdx(vSnowPosition),dFdy(vSnowPosition)));
          float snowSlope = abs(snowNormal.z);
          float snowField = (vSnowPosition.z-snowLow)/max(snowRange,.000001)-snowLine+snowNoise-.18*(1.0-snowSlope);
          float snowMask = snowEnabled*vSnowSurface*step(.25,snowSlope)*step(.000001,snowRange)*smoothstep(-.004,.004,snowField);
          diffuseColor.rgb = mix(diffuseColor.rgb,snowColour,snowMask);
          #include <roughnessmap_fragment>`);
        // Snow covers the woodland canopy's shading as well as its colour.
        shader.fragmentShader = shader.fragmentShader.replace('woodHeight *=', 'woodHeight *= (1.0-snowMask);\n          woodHeight *=');
      };
      material.customProgramCacheKey = () => `${baseKey}:snow-v1`;
      material.needsUpdate = true;
      undo.push(() => { material.onBeforeCompile = previous; material.customProgramCacheKey = previousKey; material.needsUpdate = true; });
    }
  });
  return () => undo.forEach(fn => fn());
}
