"""Prominent mapped buildings and ground-level roads at printable scale."""
from collections import defaultdict
from contour.biome_data import vector_zoom
import math
import numpy as np
import shapely
from shapely.geometry import shape
from shapely.affinity import scale
import mapbox_vector_tile
import trimesh
from contour.biome_data import _fetch_tile_with_cache, _hex_geographic_bbox, _project_to_enu, _flatten_polygons
from contour.tiles import tiles_covering_bbox

ROAD_WIDTHS_M = {'motorway': 14, 'motorway_link': 7, 'trunk': 10, 'trunk_link': 6,
                 'primary': 8, 'primary_link': 6, 'secondary': 7, 'secondary_link': 6, 'tertiary': 6}
MIN_ROAD_MM = .4
MIN_BUILDING_WIDTH_MM = .6
MIN_BUILDING_AREA_MM2 = .5


def fetch_infrastructure(frame, client, cache, token, mm_per_m):
    domain = frame.polygon_enu()
    roads = []
    buildings = defaultdict(list)
    heights = {}
    for tile in tiles_covering_bbox(*_hex_geographic_bbox(frame), vector_zoom(frame)):
        decoded = mapbox_vector_tile.decode(_fetch_tile_with_cache(client, cache, token, tile))
        for name in ('building', 'road'):
            layer = decoded.get(name, {})
            extent = layer.get('extent', 4096)
            for feature in layer.get('features', []):
                props = feature.get('properties', {})
                geom = shape(feature['geometry'])
                if name == 'road':
                    if props.get('class') not in ROAD_WIDTHS_M or props.get('structure') in ('bridge', 'tunnel'):
                        continue
                    if geom.geom_type not in ('LineString', 'MultiLineString'):
                        continue
                elif (geom.geom_type not in ('Polygon', 'MultiPolygon') or
                      str(props.get('underground', 'false')).lower() == 'true' or
                      str(props.get('extrude', 'true')).lower() == 'false' or props.get('type') == 'building:part'):
                    continue
                def lonlat(coords):
                    x = (tile.x + coords[:, 0]/extent)/2**tile.zoom
                    y = (tile.y + 1-coords[:, 1]/extent)/2**tile.zoom
                    return np.column_stack((x*360-180, np.degrees(np.arctan(np.sinh(np.pi*(1-2*y))))))
                geom = _project_to_enu(shapely.transform(geom, lonlat), frame.local_enu())
                geom = shapely.make_valid(geom).intersection(domain)
                if geom.is_empty:
                    continue
                if name == 'road':
                    width = max(MIN_ROAD_MM/mm_per_m, ROAD_WIDTHS_M[props['class']])
                    # Sub-visible bends make wide print-scale road buffers
                    # extremely costly on regional maps. Keep displacement
                    # below 0.005 mm and below 1% of the rendered road width.
                    geom = geom.simplify(min(.005/mm_per_m, width*.01), preserve_topology=True)
                    # Buffer disconnected lines separately, then use the
                    # cascaded union below. A single dense MultiLineString
                    # buffer has quadratic overlap work at regional scales.
                    for line in shapely.get_parts(geom):
                        roads.append(line.buffer(width/2).intersection(domain))
                else:
                    # Merge tile fragments before applying physical feature filters.
                    key = feature.get('id') or geom.wkb_hex
                    buildings[key].append(geom)
                    try:
                        height = float(props.get('height', 6))
                    except (ValueError, TypeError):
                        height = 6
                    heights[key] = max(heights.get(key, 0), height if math.isfinite(height) and height > 0 else 6)
    road_region = scale(shapely.union_all(roads), mm_per_m, mm_per_m, origin=(0, 0))
    candidates = []
    for key, pieces in buildings.items():
        region = scale(shapely.union_all(pieces), mm_per_m, mm_per_m, origin=(0, 0))
        for polygon in _flatten_polygons(region):
            if polygon.area < MIN_BUILDING_AREA_MM2 or polygon.buffer(-MIN_BUILDING_WIDTH_MM/2).is_empty:
                continue
            candidates.append((polygon, max(.4, heights[key]*mm_per_m)))
    return road_region, candidates


def build_buildings(land, candidates, exclusion):
    """Flat roofs with terrain-conforming bases; keep buildings off the route/water."""
    top = land.triangles[land.face_normals[:, 2] > 1e-8]
    tree = shapely.STRtree(shapely.polygons(top[:, :, :2]))
    pieces, offsets, footprints = [], [], []
    occupied = exclusion
    for polygon, height in sorted(candidates, key=lambda item: item[0].area, reverse=True):
        if polygon.intersects(occupied):
            continue
        hits = tree.query(polygon, predicate='intersects')
        if not len(hits):
            continue
        ground = float(top[hits, :, 2].max())
        floor = float(land.bounds[0, 2])
        roof = ground + height
        prism = trimesh.creation.extrude_polygon(polygon, roof-floor, engine='triangle')
        prism.apply_translation([0, 0, floor])
        part = trimesh.boolean.difference([prism, land], engine='manifold')
        if part.is_empty:
            continue
        if not part.is_volume:
            raise ValueError('Building did not produce a closed positive solid')
        offset = np.zeros_like(part.vertices)
        offset[np.isclose(part.vertices[:, 2], roof, atol=1e-5, rtol=0), 2] = height
        pieces.append(part)
        offsets.append(offset)
        footprints.append(polygon)
        occupied = shapely.union_all([occupied, polygon])
    if not pieces:
        return None, shapely.Polygon()
    mesh = trimesh.util.concatenate(pieces)
    mesh.vertex_attributes['_route_offset'] = np.concatenate(offsets)
    return mesh, shapely.union_all(footprints)
