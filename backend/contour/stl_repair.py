"""Bounded numerical regularization for STL's float32 coordinate format.

STL has no vertex indices. Distinct topological vertices at an identical
rounded position are welded by slicers. Separate these numerical contacts
without filling holes or opening surfaces. Complete shells that collapse to
exactly zero area in float32 may be discarded; individual faces are never removed.
"""
import manifold3d as md
import numpy as np
import trimesh

STEP_MM = 0.0001
MAX_PASSES = 8
MAX_DISPLACEMENT_MM = 0.0018


def repair_candidates(mesh: trimesh.Trimesh):
    solid = md.Manifold(md.Mesh64(np.array(mesh.vertices, dtype=np.float64, order='C'),
                                np.array(mesh.faces, dtype=np.uint64, order='C')))
    valid_solid = solid.status() == md.Error.NoError and not solid.is_empty()
    # Near-collinear fans can be retriangulated during Manifold import, before
    # simplification starts. If that already breaches the volume bound, first
    # separate rounded contacts while preserving the original indexed topology.
    direct_first = not valid_solid or abs(solid.volume() - mesh.volume) > abs(mesh.volume) * 0.0001
    if direct_first and mesh.is_volume:
        yield from _separate_contacts(mesh.copy(), mesh, 0.0)
    if valid_solid:
        # Keep the established fast path for dense terrain. Retry from the
        # original solid so simplification error never accumulates.
        for tolerance in (0.01 * STEP_MM, STEP_MM, 5 * STEP_MM):
            for candidate in _regularized_candidates(solid, mesh, tolerance):
                if abs(candidate.volume - mesh.volume) <= abs(mesh.volume) * 0.0001:
                    yield candidate
    if not direct_first and mesh.is_volume:
        yield from _separate_contacts(mesh.copy(), mesh, 0.0)


def _regularized_candidates(solid, mesh, tolerance):
    output = solid.simplify(tolerance).to_mesh64()
    candidate = trimesh.Trimesh(output.vert_properties[:, :3], output.tri_verts, process=False)
    yield from _separate_contacts(candidate, mesh, tolerance)


def _separate_contacts(candidate, mesh, tolerance):
    if candidate.is_empty:
        return
    anchors = candidate.vertices.copy()
    yield candidate
    for _ in range(MAX_PASSES):
        vertices = candidate.vertices.astype(np.float32).astype(np.float64)
        # Match the positional welding used when the serialized STL is read
        # back. Float32 values near zero can differ yet still weld at 1e-8 mm.
        digits = trimesh.util.decimal_to_digits(trimesh.constants.tol.merge)
        weld_keys = np.rint(vertices * 10**digits).astype(np.int64)
        _, inverse, counts = np.unique(weld_keys, axis=0, return_inverse=True, return_counts=True)
        contacts = counts[inverse] > 1
        if contacts.any():
            ids = np.flatnonzero(contacts)
            groups = inverse[ids]
            order = np.argsort(groups, kind='stable')
            sorted_groups = groups[order]
            starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(sorted_groups) != 0], np.arange(len(ids)), 0))
            rank = np.empty(len(ids), dtype=float)
            rank[order] = np.arange(len(ids)) - starts
            # Distinct magnitudes also separate contacts with identical normals.
            amount = STEP_MM * (.5 + .5 * rank / counts[groups])
            vertices[contacts] -= _contact_directions(candidate, ids) * amount[:, None]
        candidate.vertices = vertices
        collapsed = ~candidate.nondegenerate_faces(height=1e-8)
        if collapsed.any():
            # Float32 can flatten an entire microscopic closed shell into a
            # line. Such a shell occupies no volume and has no recoverable face
            # normal. Remove only complete, exactly zero-area shells; never
            # delete individual faces or an open patch of the main solid.
            _remove_zero_area_shells(candidate)
            if candidate.is_empty:
                return
            collapsed = ~candidate.nondegenerate_faces(height=1e-8)
            faces = candidate.faces[collapsed]
            triangles = candidate.vertices[faces]
            longest = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles, axis=2).argmax(axis=1)
            rows = np.arange(len(faces))
            a, b, c = (faces[rows, (longest + offset) % 3] for offset in range(3))
            edge = candidate.vertices[b] - candidate.vertices[a]
            direction = np.cross(edge, candidate.vertex_normals[c])
            length = np.linalg.norm(direction, axis=1)
            direction /= np.maximum(length[:, None], 1e-30)
            vertices = candidate.vertices.copy()
            vertices[c] += direction * STEP_MM
            candidate.vertices = vertices
        if np.linalg.norm(candidate.vertices - anchors, axis=1).max() > MAX_DISPLACEMENT_MM - tolerance:
            return
        # Reject repairs that materially inflate a genuinely tiny feature.
        # Each pass moves a vertex at most twice STEP_MM, plus float32 rounding.
        if abs(candidate.volume - mesh.volume) <= abs(mesh.volume) * 0.0001:
            yield candidate


def _remove_zero_area_shells(candidate: trimesh.Trimesh) -> None:
    """Discard only complete closed components with exactly zero surface area."""
    zero = np.flatnonzero(candidate.area_faces == 0)
    remove = []
    if len(zero):
        zero_faces = candidate.faces[zero]
        adjacency = trimesh.graph.face_adjacency(faces=zero_faces)
        for component in trimesh.graph.connected_components(adjacency, nodes=np.arange(len(zero_faces)), min_len=1):
            faces = zero_faces[component]
            edges = np.sort(trimesh.geometry.faces_to_edges(faces), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            if len(faces) >= 4 and np.all(counts == 2):
                remove.extend(zero[component])
    if remove:
        keep = np.ones(len(candidate.faces), dtype=bool)
        keep[remove] = False
        candidate.update_faces(keep)


def _contact_directions(mesh: trimesh.Trimesh, ids: np.ndarray) -> np.ndarray:
    """Use a real adjacent face when angle-weighted vertex normals cancel."""
    directions = mesh.vertex_normals[ids].copy()
    lengths = np.linalg.norm(directions, axis=1)
    weak = np.flatnonzero(lengths < 1e-12)
    if len(weak):
        incident = mesh.vertex_faces[ids[weak]]
        for row, faces in zip(weak, incident):
            faces = faces[faces >= 0]
            if len(faces):
                face = faces[np.argmax(mesh.area_faces[faces])]
                directions[row] = mesh.face_normals[face]
        lengths = np.linalg.norm(directions, axis=1)
    return directions / np.maximum(lengths[:, None], 1e-30)
