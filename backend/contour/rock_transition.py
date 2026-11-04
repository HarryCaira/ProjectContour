"""Subtle printable interruptions along mapped rock edges, in print millimetres."""
import numpy as np
import shapely
from contour.biome_data import _flatten_polygons

EDGE_DEPTH_MM = 0.55
EDGE_SPACING_MM = 0.75


def natural_rock_region(region, domain, water=None):
    """Keep rock interiors and small patches; vary only a narrow inward band.

    Coordinates and wavelengths are physical, so geographic extent cannot make
    the texture unprintably fine. No randomness or new elevation data is used.
    Inward cuts cannot spread rock into another mapped material.
    """
    protected = domain.boundary.buffer(EDGE_DEPTH_MM)
    if water is not None and not water.is_empty:
        protected = shapely.union_all([protected, water.boundary.buffer(EDGE_DEPTH_MM)])
    result = []
    for polygon in _flatten_polygons(shapely.make_valid(region)):
        # Preserve isolated stones and narrow strips instead of eroding them.
        if polygon.area < 4 or polygon.buffer(-EDGE_DEPTH_MM * 1.5).is_empty:
            result.append(polygon)
            continue
        cutters = []
        for ring in (polygon.exterior, *polygon.interiors):
            distances = np.arange(0, ring.length, EDGE_SPACING_MM)
            points = shapely.line_interpolate_point(ring, distances)
            xy = shapely.get_coordinates(points)
            # Several wavelengths avoid a repeated row of identical scallops.
            phase = np.sin(xy[:, 0] * 1.71 + xy[:, 1] * 2.13)
            variation = .5 + .5 * np.sin(xy[:, 0] * 4.37 - xy[:, 1] * 3.19 + phase)
            radii = .20 + .35 * variation
            keep = (phase > -.45) & ~shapely.intersects(points, protected)
            cutters.extend(shapely.buffer(points[keep], radii[keep], quad_segs=5))
        if not cutters:
            result.append(polygon)
            continue
        cut = polygon.difference(shapely.union_all(cutters))
        # Remove sub-resolution vertices before the final material partition.
        result.extend(_flatten_polygons(shapely.make_valid(cut.simplify(.025, preserve_topology=True))))
    return shapely.union_all(result)
