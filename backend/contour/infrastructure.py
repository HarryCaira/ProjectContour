"""Prominent mapped buildings and ground-level roads at printable scale."""
from collections import defaultdict
from collections.abc import Callable
import hashlib
import json
from threading import Lock
from weakref import WeakValueDictionary
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


def _fetch_infrastructure(frame: HexFrame, client: HttpClient, cache: TileCache, token: str, mm_per_m: float, *, checkpoint: Callable[[], None] | None = None) -> tuple[shapely.Geometry, list[BuildingCandidate], shapely.Geometry]:
    checkpoint = checkpoint or (lambda: None)
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
        checkpoint()
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
    shapely.prepare(coverage)
    candidates = []
    for key, region in regions.items():
        checkpoint()
        clipped = region if coverage.covers(region) else region.intersection(coverage)
        for polygon in _flatten_polygons(clipped):
            height = max(.4, heights[key]*mm_per_m)
            base = bases[key]*mm_per_m
            if base < height:
                candidates.append(BuildingCandidate(polygon, height, base, estimated[key]))
    logger.info("Buildings: %d mapped features, %d printable pieces in %d source tiles (%d estimated heights)",
                len(buildings), len(candidates), len(requests), sum(c.estimated_height for c in candidates))
    bridge_region = scale(shapely.union_all(bridges), mm_per_m, mm_per_m, origin=(0, 0))
    return road_region, candidates, bridge_region



# Share the expensive projected data between coverage, previews and exports.
# Roof enrichment remains separate so a transient optional lookup is not cached.
_cache_guard = Lock()
_cache_locks = WeakValueDictionary()


def fetch_infrastructure(frame, client, cache, token, mm_per_m, *, include_roofs=True, checkpoint=None):
    checkpoint = checkpoint or (lambda: None)
    if cache is None:
        result = _fetch_infrastructure(frame, client, cache, token, mm_per_m, checkpoint=checkpoint)
    else:
        key = hashlib.sha256(json.dumps([frame.centre_lon, frame.centre_lat,
            frame.circumradius_m, mm_per_m]).encode()).hexdigest()
        identity = (str(cache.root), key)
        with _cache_guard:
            lock = _cache_locks.setdefault(identity, Lock())
        while not lock.acquire(timeout=.1):
            checkpoint()
        try:
            checkpoint()
            data = cache.get('derived', 'infrastructure-v1', 0, key, 0, 'json')
            result = None
            if data is not None:
                try:
                    raw = json.loads(data)
                    result = (shapely.from_wkb(raw['roads']),
                        [BuildingCandidate(shapely.from_wkb(c[0]), *c[1:]) for c in raw['buildings']],
                        shapely.from_wkb(raw['bridges']))
                except (ValueError, KeyError, TypeError, shapely.errors.GEOSException):
                    pass
            if result is None:
                result = _fetch_infrastructure(frame, client, cache, token, mm_per_m, checkpoint=checkpoint)
                roads, candidates, bridges = result
                cache.set('derived', 'infrastructure-v1', 0, key, 0, 'json', json.dumps({
                    'roads': roads.wkb_hex, 'bridges': bridges.wkb_hex,
                    'buildings': [[c.footprint.wkb_hex, c.height, c.min_height, c.estimated_height]
                                  for c in candidates]}).encode())
                # Bound derived data independently of retained source tiles.
                directory = cache.root / 'derived' / 'infrastructure-v1' / '0'
                entries = []
                for path in directory.glob('*/0.json'):
                    try:
                        entries.append((path.stat().st_mtime, path))
                    except FileNotFoundError:
                        pass
                for _, path in sorted(entries, reverse=True)[32:]:
                    path.unlink(missing_ok=True)

        finally:
            lock.release()
    road_region, candidates, bridge_region = result
    if include_roofs and candidates and cache is not None:
        roof_features = fetch_roofs(frame, cache, mm_per_m)
        roof_tree = shapely.STRtree([feature.footprint for feature in roof_features])
        candidates = [c._replace(roof=matched_roof(c.footprint, roof_features, roof_tree, mm_per_m,
                                                   c.height, c.min_height)) if not c.estimated_height else c
                      for c in candidates]
    return road_region, candidates, bridge_region


def build_buildings(land: trimesh.Trimesh, candidates: list[BuildingCandidate], exclusion: shapely.Geometry, *, preview: bool = False, checkpoint: Callable[[], None] | None = None) -> tuple[trimesh.Trimesh | None, shapely.Geometry]:
    """Union adjacent building parts and trim route clearance before solid creation."""
    checkpoint = checkpoint or (lambda: None)
    trimmed = []
    for candidate in candidates:
        polygon, height = candidate[:2]
        base = candidate[2] if len(candidate) > 2 else 0.0
        roof = getattr(candidate, "roof", None)
        for piece in _flatten_polygons(polygon.difference(exclusion)):
            # Clipping can leave effectively zero-area triangles with nearly
            # coincident vertices. These are numerical residue, not buildings.
            if piece.area <= 1e-12:
                continue
            trimmed.append((piece, height, base, roof))
    if not trimmed:
        return None, shapely.Polygon()
    regions = [item[0] for item in trimmed]
    index = shapely.STRtree(regions)
    blocks = _flatten_polygons(shapely.union_all(regions))
    top = land.triangles[land.face_normals[:, 2] > 1e-8]
    top_polygons = shapely.polygons(top[:, :, :2])
    tree = shapely.STRtree(top_polygons)
    normals = np.cross(top[:, 1]-top[:, 0], top[:, 2]-top[:, 0])
    pieces, offsets, footprints = [], [], []
    floor = float(land.bounds[0, 2])
    terrain_solid = None if preview else md.Manifold(md.Mesh64(np.asarray(land.vertices, dtype=np.float64),
                                         np.asarray(land.faces, dtype=np.uint64)))
    for block in blocks:
        checkpoint()
        if not _printable(block):
            continue
        hits = tree.query(block, predicate='intersects')
        if not len(hits):
            continue
        # Evaluate terrain only inside the footprint, not remote triangle corners.
        clipped = shapely.intersection(top_polygons[hits], block)
        xy, owners = shapely.get_coordinates(clipped, return_index=True)
        if not len(xy):
            continue
        triangles = hits[owners]
        delta = xy - top[triangles, 0, :2]
        z = top[triangles, 0, 2] - np.einsum('ij,ij->i', delta, normals[triangles, :2]) / normals[triangles, 2]
        ground = float(z.max())
        if not np.isfinite(ground):
            continue
        prisms = []
        for idx in index.query(block, predicate='intersects'):
            polygon, height, base, roof = trimmed[idx]
            for footprint in _flatten_polygons(polygon.intersection(block)):
                if footprint.area <= 1e-12:
                    continue
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
        # Hidden foundations can overlap terrain in a display scene. Exports
        # still subtract terrain and validate disjoint closed solids.
        result = solid if preview else solid - terrain_solid
        if result.status() != md.Error.NoError:
            raise ValueError('Building solid construction failed')
        if result.is_empty():
            continue
        # Terrain subtraction leaves microscopic coplanar slivers. Regularize
        # the boolean output before STL quantization (0.1 micrometre tolerance).
        result = result.simplify(0.0001)
        output = result.to_mesh64()
        part = trimesh.Trimesh(output.vert_properties[:, :3], output.tri_verts, process=False)
        if not preview and not part.is_volume:
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
