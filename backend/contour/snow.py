"""Artistic, elevation- and slope-based snow coverage (not observed weather)."""
import numpy as np
import shapely
import trimesh


def snow_region(land: trimesh.Trimesh, snowline: float, exaggeration: float = 1):
    # Undo display exaggeration for a stable coverage field. Translation is
    # irrelevant because elevations are normalized to this landscape's relief.
    vertices = land.vertices.copy()
    vertices[:, 2] /= exaggeration
    triangles = vertices[land.faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    top = normals[:, 2] > 1e-6
    if not top.any():
        return shapely.Polygon()
    triangles, nz = triangles[top], normals[top, 2]
    low, high = triangles[:, :, 2].min(), triangles[:, :, 2].max()
    if high-low < 1e-6:
        return shapely.Polygon()
    xymin, xymax = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    uv = (triangles[:, :, :2]-xymin)/np.maximum(xymax-xymin, 1e-9)
    x, y = uv[:, :, 0], uv[:, :, 1]
    noise = .028*np.sin(x*31+y*7)*np.cos(y*27-x*5) + .012*np.sin(x*83-y*61)
    field = (triangles[:, :, 2]-low)/(high-low) - snowline + noise - .18*(1-nz[:, None])
    field[nz < .25] = -1
    full = (field >= 0).all(axis=1)
    polygons = list(shapely.polygons(triangles[full, :, :2]))
    crossing = (field >= 0).any(axis=1) & ~full
    # Clip boundary triangles against their interpolated snow field. Interior
    # triangles are vectorized; no full-mesh remeshing or elevation API call.
    for triangle, values in zip(triangles[crossing, :, :2], field[crossing]):
        ring = []
        for i in range(3):
            j = (i+1) % 3
            if values[i] >= 0:
                ring.append(triangle[i])
            if (values[i] >= 0) != (values[j] >= 0):
                t = values[i]/(values[i]-values[j])
                ring.append(triangle[i] + t*(triangle[j]-triangle[i]))
        if len(ring) >= 3:
            polygons.append(shapely.Polygon(ring))
    return shapely.union_all(polygons)
