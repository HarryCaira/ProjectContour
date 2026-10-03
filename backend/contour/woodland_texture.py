"""Low rounded woodland relief, in final print millimetres."""
from __future__ import annotations

import numpy as np
import shapely
import trimesh

from contour.landcover_mesh import printable_coverage

BUMP_HEIGHT_MM = 0.7
BUMP_RADIUS_MM = 0.8
BUMP_SPACING_MM = 0.9


def woodland_scale(mm_per_m: float) -> float:
    """Scale with geographic extent, bounded for printable and modest relief."""
    return float(np.clip(mm_per_m / .02, .5, 2.0))


def add_woodland_texture(insert: trimesh.Trimesh, region: shapely.Geometry,
                         terrain: trimesh.Trimesh, route: trimesh.Trimesh | None = None, *, mm_per_m: float = .02) -> trimesh.Trimesh:
    """Union terrain-following rounded caps, inset from coverage and the route."""
    canopy_scale = woodland_scale(mm_per_m)
    spacing = BUMP_SPACING_MM * canopy_scale
    safe = shapely.union_all(printable_coverage(region))
    if route is not None:
        triangles = shapely.polygons(route.triangles[:, :, :2])
        outline = shapely.union_all(triangles[shapely.area(triangles) > 1e-10])
        safe = safe.difference(outline.buffer(.15))
    if safe.is_empty:
        return insert
    minx, miny, maxx, maxy = safe.bounds
    x, y = np.meshgrid(np.arange(np.floor(minx/spacing), np.ceil(maxx/spacing)+1),
                       np.arange(np.floor(miny/spacing), np.ceil(maxy/spacing)+1))
    cells = np.column_stack((x.ravel(), y.ravel()))
    def random(seed: float) -> np.ndarray:
        return np.mod(np.sin(cells[:,0]*127.1 + cells[:,1]*311.7 + seed)*43758.5453, 1)
    centres = (cells + .5 + .9*np.column_stack((random(1)-.5, random(2)-.5))) * spacing
    radius = (.28 + .38*random(3)) * canopy_scale
    aspect = .8 + .4*random(4)
    angle = 2*np.pi*random(5)
    height = (.3 + .4*random(6)) * canopy_scale
    clusters = .5 + .5*np.sin(centres[:,0]/canopy_scale*.65)*np.sin(centres[:,1]/canopy_scale*.53)
    keep = (random(7) < .65 + .3*clusters) & shapely.contains_xy(safe, centres[:,0], centres[:,1])
    centres, radius, aspect, angle, height = (a[keep] for a in (centres, radius, aspect, angle, height))
    if not len(centres):
        return insert
    distance = shapely.distance(shapely.points(centres), safe.boundary)
    radius = np.minimum(radius, np.maximum(0, distance-.03) / np.maximum(1, aspect))
    fade = np.clip(distance/(.85*canopy_scale), 0, 1)
    height *= fade*fade*(3-2*fade)
    keep = (radius >= .12) & (height >= .02)
    centres, radius, aspect, angle, height = (a[keep] for a in (centres, radius, aspect, angle, height))
    if not len(centres):
        return insert
    surface = insert.triangles[insert.face_normals[:, 2] > 1e-8]
    tree = shapely.STRtree(shapely.polygons(surface[:, :, :2]))

    def heights(points: np.ndarray) -> np.ndarray:
        result = np.full(len(points), np.nan)
        ids, faces = tree.query(shapely.points(points), predicate='intersects')
        a, b, c = surface[faces, 0], surface[faces, 1], surface[faces, 2]
        ab, ac, ap = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], points[ids]-a[:, :2]
        determinant = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
        u = (ap[:, 0]*ac[:, 1]-ap[:, 1]*ac[:, 0])/determinant
        v = (ab[:, 0]*ap[:, 1]-ab[:, 1]*ap[:, 0])/determinant
        result[ids] = a[:, 2] + u*(b[:, 2]-a[:, 2]) + v*(c[:, 2]-a[:, 2])
        return result

    # A flattened ellipsoid supplies closed topology; bend its base to the
    # exact existing terrain. Its lower half embeds into the woodland insert.
    template = trimesh.creation.uv_sphere(radius=1, count=[8, 16])
    unit = template.vertices
    local_x = unit[None,:,0]*radius[:,None]*aspect[:,None]
    local_y = unit[None,:,1]*radius[:,None]
    xy = centres[:,None,:] + np.stack((local_x*np.cos(angle[:,None])-local_y*np.sin(angle[:,None]),
                                      local_x*np.sin(angle[:,None])+local_y*np.cos(angle[:,None])),axis=2)
    z = heights(xy.reshape(-1,2)).reshape(len(centres), -1)
    valid = np.isfinite(z).all(axis=1)
    xy, z, height = xy[valid], z[valid], height[valid]
    if not len(z):
        return insert
    z += np.where(unit[None,:,2] >= 0, unit[None,:,2]*height[:,None], unit[None,:,2]*.08)
    vertices = np.concatenate((xy, z[:,:,None]), axis=2)
    # Crowns overlap. Union them as individual closed solids rather than
    # passing an intersecting triangle soup as a single input solid.
    crowns = [trimesh.Trimesh(v, template.faces, process=False) for v in vertices]
    textured = trimesh.boolean.union([insert, *crowns], engine='manifold')
    # Protect the existing terrain interface even on very shallow inserts.
    textured = trimesh.boolean.difference([textured, terrain], engine='manifold')
    if not textured.is_volume:
        raise ValueError('Woodland texture did not produce a closed positive solid')
    return textured
