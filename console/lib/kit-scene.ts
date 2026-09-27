import type { ModelColours } from "./settings";
import { Box3, Group, Mesh, MeshStandardMaterial, Matrix3, Vector3 } from "three";

interface RouteVertices {
  mesh: Mesh;
  original: Float32Array;
  base: Float32Array;
  offsets?: Float32Array;
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
  for (const mesh of meshes) {
    // Work in the same coordinate frame as the terrain pivot, preserving the
    // original geometry so repeated slider changes never accumulate distortion.
    const offsetMatrix = new Matrix3().setFromMatrix4(mesh.matrixWorld);
    mesh.geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
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
    routes.push({ mesh, original, base, offsets });
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
    for (const { mesh, original, base, offsets } of transform.routes) {
      const positions = mesh.geometry.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const raisedHeight = offsets?.[i * 3 + 2] ?? original[i * 3 + 2] - base[i];
        const terrainZ = original[i * 3 + 2] - raisedHeight;
        if (offsets) {
          positions.setX(i, original[i * 3] + offsets[i * 3] * (widthScale - 1));
          positions.setY(i, original[i * 3 + 1] + offsets[i * 3 + 1] * (widthScale - 1));
        }
        positions.setZ(i, transform.pivot + (terrainZ - transform.pivot) * exaggeration + raisedHeight * heightScale);
      }
      positions.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
      mesh.geometry.computeBoundingBox();
      mesh.geometry.computeBoundingSphere();
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
  const parts = { land: colours.terrain, water: colours.water, route: colours.route };
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
