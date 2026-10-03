import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { ModelColours } from "./settings";
import { Box3, Group, Mesh, MeshStandardMaterial, Matrix3, Vector3 } from "three";

/** Give flush road interfaces stable depth ordering without moving print geometry. */
export function prepareRoadRendering(scene: Group): void {
  scene.traverse((object) => {
    if (!(object instanceof Mesh) || object.name !== "roads") return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    const roadMaterials = materials.map((material) => {
      const roadMaterial = material.clone();
      roadMaterial.polygonOffset = true;
      roadMaterial.polygonOffsetFactor = -1;
      roadMaterial.polygonOffsetUnits = -1;
      return roadMaterial;
    });
    object.material = Array.isArray(object.material) ? roadMaterials : roadMaterials[0];
  });
}

/** Smooth terrain lighting while retaining sharp cut walls and plinth edges.
 * This affects the preview normals only; printable vertex positions stay intact.
 */
export function prepareTerrainShading(scene: Group): void {
  scene.traverse((object) => {
    if (!(object instanceof Mesh) || object.name !== "land") return;
    const original = object.geometry;
    const faces = (original.index?.count ?? original.attributes.position.count) / 3;
    if (faces > 250_000) {
      // Keep shared vertices on large previews; crease expansion creates huge
      // JS object maps and multiplies CPU/GPU geometry memory.
      original.computeVertexNormals();
    } else {
      object.geometry = toCreasedNormals(original, Math.PI / 4);
      if (object.geometry !== original) original.dispose();
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (material instanceof MeshStandardMaterial) {
        material.flatShading = false;
        material.needsUpdate = true;
      }
    }
  });
}

interface RouteVertices {
  mesh: Mesh;
  original: Float32Array;
  base: Float32Array;
  offsets?: Float32Array;
  building?: boolean;
}
const routeTransforms = new WeakMap<Group, { pivot: number; routes: RouteVertices[] }>();

/** Centre the complete kit before scaling metres into viewer units. */
export function normaliseKit(scene: Group): Group {
  const box = new Box3().setFromObject(scene);
  const size = box.getSize(new Vector3());
  const centre = box.getCenter(new Vector3());
  const longest = Math.max(size.x, size.y);
  if (box.isEmpty() || !Number.isFinite(longest) || longest <= 0) {
    throw new Error("The model has no viewable geometry.");
  }

  // Exaggerate around the contact plane, leaving the plinth untouched.
  const plinth = scene.getObjectByName("plinth");
  const pivot = plinth ? new Box3().setFromObject(plinth).max.z : box.min.z;
  const fixed = new Group();
  if (plinth) fixed.attach(plinth);
  const routes: RouteVertices[] = [];
  const route = scene.getObjectByName("route");
  const meshes: Mesh[] = [];
  route?.traverse((object) => { if (object instanceof Mesh) meshes.push(object); });
  scene.getObjectByName("buildings")?.traverse((object) => { if (object instanceof Mesh) meshes.push(object); });
  for (const mesh of meshes) {
    // Work in the same coordinate frame as the terrain pivot, preserving the
    // original geometry so repeated slider changes never accumulate distortion.
    const offsetMatrix = new Matrix3().setFromMatrix4(mesh.matrixWorld);
    mesh.geometry.applyMatrix4(mesh.matrixWorld);
    const offsetAttribute = mesh.geometry.getAttribute("_route_offset");
    let offsets: Float32Array | undefined;
    if (offsetAttribute) {
      offsets = new Float32Array(offsetAttribute.count * 3);
      const offset = new Vector3();
      for (let i = 0; i < offsetAttribute.count; i++) {
        offset.fromBufferAttribute(offsetAttribute, i).applyMatrix3(offsetMatrix);
        offsets.set([offset.x, offset.y, offset.z], i * 3);
      }
    }
    mesh.position.set(0, 0, 0);
    mesh.quaternion.identity();
    mesh.scale.set(1, 1, 1);
    fixed.add(mesh);
    const positions = mesh.geometry.attributes.position;
    const original = new Float32Array(positions.count * 3);
    const bases = new Map<string, number>();
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      original.set([x, y, z], i * 3);
      const key = `${x},${y}`;
      bases.set(key, Math.min(bases.get(key) ?? Infinity, z));
    }
    const base = new Float32Array(positions.count);
    for (let i = 0; i < positions.count; i++) {
      base[i] = bases.get(`${original[i * 3]},${original[i * 3 + 1]}`)!;
    }
    routes.push({ mesh, original, base, offsets, building: mesh.name === "buildings" });
  }

  const terrainOffset = new Group();
  terrainOffset.position.z = -pivot;
  terrainOffset.add(scene);
  const relief = new Group();
  relief.name = "contour-relief";
  relief.position.z = pivot;
  relief.add(terrainOffset);

  const centred = new Group();
  centred.position.set(-centre.x, -centre.y, -box.min.z);
  centred.add(fixed, relief);

  const normalised = new Group();
  normalised.scale.setScalar(2 / longest);
  normalised.add(centred);
  routeTransforms.set(normalised, { pivot, routes });
  return normalised;
}

/** Keep the landscape attached to the fixed plinth as relief changes. */
export function setKitExaggeration(root: Group, exaggeration: number, widthScale = 1, heightScale = 1): void {
  const relief = root.getObjectByName("contour-relief");
  if (!relief) throw new Error("The model has no terrain transform.");
  relief.scale.z = exaggeration;
  const transform = routeTransforms.get(root);
  if (transform) {
    for (const { mesh, original, base, offsets, building } of transform.routes) {
      const positions = mesh.geometry.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const raisedHeight = offsets?.[i * 3 + 2] ?? original[i * 3 + 2] - base[i];
        const terrainZ = original[i * 3 + 2] - raisedHeight;
        if (offsets) {
          positions.setX(i, original[i * 3] + offsets[i * 3] * (widthScale - 1));
          positions.setY(i, original[i * 3 + 1] + offsets[i * 3 + 1] * (widthScale - 1));
        }
        positions.setZ(i, transform.pivot + (terrainZ - transform.pivot) * exaggeration + raisedHeight * (building ? 1 : heightScale));
      }
      positions.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
      mesh.geometry.computeBoundingBox();
      mesh.geometry.computeBoundingSphere();
      mesh.geometry.boundsTree?.refit();
    }
  }
}

/** Measure preview geometry without camera zoom or other parent transforms. */
export function measureKit(root: Group, physicalSizeMm: number): { bounds: Box3; millimetres: Vector3 } {
  const bounds = new Box3().setFromObject(root.clone(true));
  return { bounds, millimetres: bounds.getSize(new Vector3()).multiplyScalar(physicalSizeMm / 2) };
}

/** Change surface colours without rebuilding or modifying geometry. */
export function setKitColours(root: Group, colours: ModelColours): void {
  const parts = { land: colours.terrain, water: colours.water, route: colours.route, roads: colours.roads, buildings: colours.buildings };
  root.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    const colour = parts[object.name as keyof typeof parts];
    if (!colour) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (material instanceof MeshStandardMaterial) material.color.set(colour);
    }
  });
}

/** Release GPU buffers and BVH references when replacing a loaded preview. */
export function disposeKit(root: Group): void {
  root.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    object.geometry.disposeBoundsTree?.();
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) material.dispose();
  });
}
