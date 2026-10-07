"""Prominent mapped buildings and ground-level roads at printable scale."""
from collections import defaultdict
from contour.biome_data import vector_zoom
import math
import logging
from typing import NamedTuple
from contour.hex_frame import HexFrame
from contour.http_client import HttpClient
from contour.tile_cache import TileCache
import numpy as np
import shapely
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.affinity import scale
import mapbox_vector_tile
import trimesh
import manifold3d as md
from contour.biome_data import _fetch_tile_with_cache, _hex_geographic_bbox, _project_to_enu, _flatten_polygons
from contour.tiles import tiles_covering_bbox
from contour.building_roofs import Roof, fetch_roofs, matched_roof, apply_roof

ROAD_WIDTHS_M = {'motorway': 14, 'motorway_link': 7, 'trunk': 10, 'trunk_link': 6,
                 'primary': 8, 'primary_link': 6, 'secondary': 7, 'secondary_link': 6, 'tertiary': 6,
                 'tertiary_link': 5, 'street': 6, 'street_limited': 5, 'service': 3,
                 'track': 3, 'pedestrian': 4, 'path': 1.5}
logger = logging.getLogger(__name__)

MIN_ROAD_MM = .2
MIN_BUILDING_WIDTH_MM = .25
MIN_BUILDING_AREA_MM2 = MIN_BUILDING_WIDTH_MM ** 2


class BuildingCandidate(NamedTuple):
    footprint: shapely.Geometry
    height: float
    min_height: float = 0.0
    estimated_height: bool = False
    roof: Roof | None = None


def printable_road_width(properties: dict, mm_per_m: float) -> float | None:
    """Estimated ground width by Mapbox class, with a minimum printed width.

    Streets tiles do not supply surveyed carriageway widths. Narrow roads
    are widened to 0.2 mm so the street network remains visible.
    """
    if properties.get('structure') == 'tunnel':
        return None
    width = ROAD_WIDTHS_M.get(properties.get('class'))
    if width is None:
        return None
    return max(width, MIN_ROAD_MM / mm_per_m)


def _printable(polygon: shapely.Geometry) -> bool:
    return polygon.area >= MIN_BUILDING_AREA_MM2 and not polygon.buffer(-MIN_BUILDING_WIDTH_MM / 2).is_empty


def fetch_infrastructure(frame: HexFrame, client: HttpClient, cache: TileCache, token: str, mm_per_m: float, *, include_roofs: bool = True) -> tuple[shapely.Geometry, list[BuildingCandidate], shapely.Geometry]:
    domain = frame.polygon_enu()
    roads = []
    bridges = []
    buildings = defaultdict(list)
    heights = {}
    bases = {}
    estimated = {}
    # Share z16 tiles for local streets and buildings; regional requests remain bounded.
    requests = {}
    for layer_name, zoom in (('road', vector_zoom(frame, requested=16)), ('building', vector_zoom(frame, requested=16))):
        for tile in tiles_covering_bbox(*_hex_geographic_bbox(frame), zoom):
            requests.setdefault(tile, set()).add(layer_name)
    for tile, layer_names in requests.items():
        decoded = mapbox_vector_tile.decode(_fetch_tile_with_cache(client, cache, token, tile))
        for name in sorted(layer_names):
            layer = decoded.get(name, {})
            extent = layer.get('extent', 4096)
            for feature in layer.get('features', []):
                props = feature.get('properties', {})
                geom = shape(feature['geometry'])
                if name == 'road':
                    if printable_road_width(props, mm_per_m) is None:
                        continue
                    if geom.geom_type not in ('LineString', 'MultiLineString'):
                        continue
                elif (geom.geom_type not in ('Polygon', 'MultiPolygon') or
                      str(props.get('underground', 'false')).lower() == 'true' or
                      str(props.get('extrude', 'true')).lower() == 'false'):
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
                    width = printable_road_width(props, mm_per_m)
                    # Sub-visible bends make wide print-scale road buffers
                    # extremely costly on regional maps. Keep displacement
                    # below 0.005 mm and below 1% of the rendered road width.
                    geom = geom.simplify(min(.005/mm_per_m, width*.01), preserve_topology=True)
                    # Buffer disconnected lines separately, then use the
                    # cascaded union below. A single dense MultiLineString
                    # buffer has quadratic overlap work at regional scales.
                    for line in shapely.get_parts(geom):
                        target = bridges if props.get("structure") == "bridge" else roads
                        target.append(line.buffer(width/2).intersection(domain))
                else:
                    # Merge tile fragments before applying physical feature filters.
                    key = (props.get('type', 'building'), feature.get('id', geom.wkb_hex))
                    buildings[key].append(geom)
                    try:
                        height = float(props.get('height', 6))
                    except (ValueError, TypeError):
                        height = float("nan")
                    estimated[key] = props.get('height') is None or not math.isfinite(height) or height <= 0
                    try:
                        base = float(props.get('min_height', 0) or 0)
                    except (ValueError, TypeError):
                        base = 0
                    bases[key] = max(0, base) if math.isfinite(base) else 0
                    heights[key] = max(heights.get(key, 0), height if math.isfinite(height) and height > 0 else 6)
    road_region = scale(shapely.union_all(roads), mm_per_m, mm_per_m, origin=(0, 0))
    # Filter connected blocks, not individual terrace houses or building parts.
    regions = {key: scale(shapely.union_all(pieces), mm_per_m, mm_per_m, origin=(0, 0))
               for key, pieces in buildings.items()}
    coverage = shapely.union_all([p for p in _flatten_polygons(shapely.union_all(list(regions.values())))
                                 if _printable(p)])
    candidates = []
    for key, region in regions.items():
        for polygon in _flatten_polygons(region.intersection(coverage)):
            height = max(.4, heights[key]*mm_per_m)
            base = bases[key]*mm_per_m
            if base < height:
                candidates.append(BuildingCandidate(polygon, height, base, estimated[key]))
    if include_roofs and candidates and cache is not None:
        roof_features = fetch_roofs(frame, cache, mm_per_m)
        roof_tree = shapely.STRtree([feature.footprint for feature in roof_features])
        candidates = [c._replace(roof=matched_roof(c.footprint, roof_features, roof_tree, mm_per_m,
                                                   c.height, c.min_height)) if not c.estimated_height else c
                      for c in candidates]
    logger.info("Buildings: %d mapped features, %d printable pieces in %d source tiles (%d estimated heights)",
                len(buildings), len(candidates), len(requests), sum(c.estimated_height for c in candidates))
    bridge_region = scale(shapely.union_all(bridges), mm_per_m, mm_per_m, origin=(0, 0))
    return road_region, candidates, bridge_region


def build_buildings(land: trimesh.Trimesh, candidates: list[BuildingCandidate], exclusion: shapely.Geometry) -> tuple[trimesh.Trimesh | None, shapely.Geometry]:
    """Union adjacent building parts and trim route clearance before solid creation."""
    trimmed = []
    for candidate in candidates:
        polygon, height = candidate[:2]
        base = candidate[2] if len(candidate) > 2 else 0.0
        roof = getattr(candidate, "roof", None)
        for piece in _flatten_polygons(polygon.difference(exclusion)):
            trimmed.append((piece, height, base, roof))
    if not trimmed:
        return None, shapely.Polygon()
    regions = [item[0] for item in trimmed]
    index = shapely.STRtree(regions)
    blocks = _flatten_polygons(shapely.union_all(regions))
    top = land.triangles[land.face_normals[:, 2] > 1e-8]
    tree = shapely.STRtree(shapely.polygons(top[:, :, :2]))
    pieces, offsets, footprints = [], [], []
    floor = float(land.bounds[0, 2])
    terrain_solid = md.Manifold(md.Mesh64(np.asarray(land.vertices, dtype=np.float64),
                                         np.asarray(land.faces, dtype=np.uint64)))
    for block in blocks:
        if not _printable(block):
            continue
        hits = tree.query(block, predicate='intersects')
        if not len(hits):
            continue
        # Evaluate terrain only inside the footprint, not remote triangle corners.
        ground = -np.inf
        for triangle in top[hits]:
            clipped = shapely.Polygon(triangle[:, :2]).intersection(block)
            xy = shapely.get_coordinates(clipped)
            if not len(xy):
                continue
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            z = triangle[0, 2] - ((xy-triangle[0, :2]) @ normal[:2]) / normal[2]
            ground = max(ground, float(z.max()))
        if not np.isfinite(ground):
            continue
        prisms = []
        for idx in index.query(block, predicate='intersects'):
            polygon, height, base, roof = trimmed[idx]
            for footprint in _flatten_polygons(polygon.intersection(block)):
                bottom = ground + base if base > 0 else floor
                footprint = orient(footprint, sign=1.0)
                rings = [np.asarray(ring.coords)[:-1, :2] for ring in (footprint.exterior, *footprint.interiors)]
                prism = md.CrossSection(rings).extrude(ground+height-bottom).translate((0, 0, bottom))
                if roof is not None:
                    prism = apply_roof(prism, roof, ground+height)
                prisms.append(prism)
        if not prisms:
            continue
        solid = md.Manifold.batch_boolean(prisms, md.OpType.Add)
        result = solid - terrain_solid
        if result.status() != md.Error.NoError:
            raise ValueError('Building solid construction failed')
        if result.is_empty():
            continue
        # Terrain subtraction leaves microscopic coplanar slivers. Regularize
        # the boolean output before STL quantization (0.1 micrometre tolerance).
        result = result.simplify(0.0001)
        output = result.to_mesh64()
        part = trimesh.Trimesh(output.vert_properties[:, :3], output.tri_verts, process=False)
        if not part.is_volume:
            raise ValueError('Building did not produce a closed positive solid')
        offset = np.zeros_like(part.vertices)
        offset[:, 2] = np.maximum(0, part.vertices[:, 2] - ground)
        pieces.append(part)
        offsets.append(offset)
        footprints.append(block)
    if not pieces:
        return None, shapely.Polygon()
    mesh = trimesh.util.concatenate(pieces)
    mesh.vertex_attributes['_route_offset'] = np.concatenate(offsets)
    return mesh, shapely.union_all(footprints)
