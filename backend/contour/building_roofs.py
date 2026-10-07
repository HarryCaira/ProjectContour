"""Conservative OSM roof enrichment and printable planar roof solids."""
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re

import manifold3d as md
import numpy as np
import requests
import shapely
from shapely.affinity import scale

from contour.biome_data import _hex_geographic_bbox, _project_to_enu
from contour.hex_frame import HexFrame
from contour.http_client import HttpClient
from contour.tile_cache import TileCache

logger = logging.getLogger(__name__)
SUPPORTED = {'gabled', 'hipped', 'pyramidal', 'skillion', 'dome', 'cone', 'round'}
MIN_ROOF_HEIGHT_MM = .1
MIN_ROOF_SPAN_MM = .4
MAX_RADIUS_M = 2500


@dataclass(frozen=True)
class Roof:
    shape: str
    rise: float
    axis: tuple[float, float]  # Down-slope direction (across the ridge).
    bounds: tuple[float, float, float, float]  # Across min/max, along min/max.


@dataclass(frozen=True)
class RoofFeature:
    footprint: shapely.Polygon
    tags: dict


def metres(value: object) -> float | None:
    match = re.fullmatch(r'\s*((?:\d+(?:\.\d*)?|\.\d+))\s*(m|ft)?\s*', str(value))
    if not match:
        return None
    number = float(match[1]) * (.3048 if match[2] == 'ft' else 1)
    return number if math.isfinite(number) and number > 0 else None


def fetch_roofs(frame: HexFrame, cache: TileCache, mm_per_m: float,
                client: HttpClient | None = None) -> list[RoofFeature]:
    """One bounded optional query; successful responses persist across size edits."""
    if frame.circumradius_m > MAX_RADIUS_M:
        return []
    west, south, east, north = _hex_geographic_bbox(frame)
    bbox = ','.join(f'{v:.7f}' for v in (south, west, north, east))
    query = (f'[out:json][timeout:5][maxsize:8388608];'
             f'way["roof:shape"~"^(gabled|hipped|pyramidal|skillion|dome|cone|round)$"]({bbox});out tags geom;')
    key = hashlib.sha256(query.encode()).hexdigest()
    try:
        data = cache.get('osm', 'roofs-v2', 0, key, 0, 'json')
        fresh = data is None
        if fresh:
            source = client or HttpClient(timeout=7, max_retries=0)
            if client is None:
                source.session.headers.update({"User-Agent": "ProjectContour/0.1 (building roof enrichment)"})
            try:
                data = source.post('https://overpass-api.de/api/interpreter', data={'data': query})
            finally:
                if client is None:
                    source.session.close()
        if len(data) > 8_388_608:
            return []
        decoded = json.loads(data)
        if not isinstance(decoded, dict) or 'remark' in decoded or not isinstance(decoded.get('elements'), list):
            return []  # Never cache incomplete/error responses.
        if fresh:
            cache.set('osm', 'roofs-v2', 0, key, 0, 'json', data)
    except (requests.RequestException, ValueError, OSError):
        logger.warning('Optional roof lookup unavailable; keeping flat building roofs')
        return []
    features = []
    for element in decoded['elements']:
        try:
            tags = element['tags']
            if not isinstance(tags, dict):
                continue
            if not ('building' in tags or 'building:part' in tags):
                continue
            coords = [(p['lon'], p['lat']) for p in element['geometry']]
            if len(coords) < 4 or coords[0] != coords[-1]:
                continue
            polygon = shapely.Polygon(coords)
            if not polygon.is_valid or polygon.is_empty:
                continue
            polygon = scale(_project_to_enu(polygon, frame.local_enu()), mm_per_m, mm_per_m, origin=(0, 0))
            features.append(RoofFeature(polygon, tags))
        except (KeyError, TypeError, ValueError):
            continue
    return features


def matched_roof(polygon: shapely.Polygon, features: list[RoofFeature], tree: shapely.STRtree,
                 mm_per_m: float, total_height: float, base: float) -> Roof | None:
    matches = []
    for idx in tree.query(polygon, predicate='intersects'):
        source = features[idx].footprint
        overlap = polygon.intersection(source).area
        score = overlap / polygon.union(source).area
        if score >= .85:
            matches.append((score, idx))
    matches.sort(reverse=True)
    if not matches or (len(matches) > 1 and matches[0][0] - matches[1][0] < .1):
        return None
    feature = features[matches[0][1]]
    return roof_spec(feature.footprint, feature.tags, mm_per_m, total_height, base)


def roof_spec(polygon: shapely.Polygon, tags: dict, mm_per_m: float,
              total_height: float, base: float = 0) -> Roof | None:
    shape = tags.get('roof:shape')
    rise_m = metres(tags.get('roof:height'))
    if shape not in SUPPORTED or polygon.is_empty or not polygon.is_valid or polygon.interiors:
        return None
    rect = polygon.minimum_rotated_rectangle
    circular = shape in {'dome', 'cone'}
    if not circular and shape != 'skillion' and polygon.area / rect.area < .95:
        return None  # Complex roofs need mapped subparts, not an invented ridge.
    vertices = np.asarray(rect.exterior.coords)[:4, :2]
    edges = np.roll(vertices, -1, axis=0) - vertices
    ridge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    ridge /= np.linalg.norm(ridge)
    axis = np.array([-ridge[1], ridge[0]])
    if 'roof:direction' in tags:
        compass = {'N':0, 'NE':45, 'E':90, 'SE':135, 'S':180, 'SW':225, 'W':270, 'NW':315}
        raw = str(tags['roof:direction']).upper()
        try:
            angle = compass[raw] if raw in compass else float(raw)
        except ValueError:
            return None
        if not math.isfinite(angle) or not 0 <= angle <= 360:
            return None
        axis = np.array([math.sin(math.radians(angle)), math.cos(math.radians(angle))])
    elif shape == 'skillion':
        return None  # Its downhill direction cannot be inferred unambiguously.
    elif tags.get('roof:orientation', 'along') == 'across':
        axis = ridge
    elif tags.get('roof:orientation', 'along') != 'along':
        return None
    along = np.array([-axis[1], axis[0]])
    xy = np.asarray(polygon.exterior.coords)[:, :2]
    a, b = xy @ axis, xy @ along
    width, length = np.ptp(a), np.ptp(b)
    if min(width, length) < MIN_ROOF_SPAN_MM:
        return None
    # Reject a direction inconsistent with the mapped rectangular footprint.
    if circular:
        # A dome/cone on an arbitrary outline would invent architecture. Require
        # a close elliptical footprint, not merely a similar bounding-box area.
        centre = axis*((a.min()+a.max())/2) + along*((b.min()+b.max())/2)
        ellipse = shapely.Point(0, 0).buffer(1, quad_segs=32)
        ellipse = shapely.transform(ellipse, lambda xy: xy[:, :1]*axis*width/2 + xy[:, 1:]*along*length/2 + centre)
        if polygon.intersection(ellipse).area / polygon.union(ellipse).area < .95:
            return None
    elif shape != 'skillion' and polygon.area / (width*length) < .95:
        return None
    if rise_m is not None:
        rise = rise_m * mm_per_m
    elif 'roof:height' in tags:
        return None  # An invalid explicit height must not be silently replaced.
    elif shape in {'gabled', 'hipped', 'skillion', 'cone'}:
        try:
            angle = float(tags.get('roof:angle', 'nan'))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(angle) or not 0 < angle < 85:
            return None
        if shape == 'cone' and abs(width-length)/max(width,length) > .05:
            return None
        run = width if shape == 'skillion' else width/2
        rise = math.tan(math.radians(angle))*run
    else:
        return None
    # Explicit single-slope parts may taper all the way to their base.
    # Reject contradictory heights; retain the wall margin for other shapes.
    wall_height = total_height - rise - base
    if rise < MIN_ROOF_HEIGHT_MM or wall_height < (-1e-9 if shape == 'skillion' else .1):
        return None
    return Roof(shape, rise, tuple(axis), (float(a.min()), float(a.max()), float(b.min()), float(b.max())))


def apply_roof(solid: md.Manifold, roof: Roof, top: float) -> md.Manifold:
    """Cut roof planes below the existing total height; never add height twice."""
    a0, a1, b0, b1 = roof.bounds
    axis = np.array(roof.axis)
    along = np.array([-axis[1], axis[0]])
    if roof.shape in {'dome', 'cone', 'round'}:
        return curved_roof(solid, roof, top)
    planes = []
    if roof.shape == 'skillion':
        slope = -roof.rise / (a1-a0)
        planes.append((slope*axis, top-slope*a0))
    else:
        slope = roof.rise / ((a1-a0)/2)
        planes.extend([(slope*axis, top-roof.rise-slope*a0),
                       (-slope*axis, top-roof.rise+slope*a1)])
        if roof.shape in {'hipped', 'pyramidal'}:
            end_slope = roof.rise / ((b1-b0)/2) if roof.shape == 'pyramidal' else slope
            planes.extend([(end_slope*along, top-roof.rise-end_slope*b0),
                           (-end_slope*along, top-roof.rise+end_slope*b1)])
    for gradient, intercept in planes:
        normal = np.array([*gradient, -1.0])
        solid = solid.trim_by_plane(normal, -intercept / np.linalg.norm(normal))
    return solid


def curved_roof(solid: md.Manifold, roof: Roof, top: float) -> md.Manifold:
    """Mapped curved caps tessellated to approximately 0.01 mm chord error."""
    a0, a1, b0, b1 = roof.bounds
    width, length = a1-a0, b1-b0
    axis = np.array(roof.axis)
    along = np.array([-axis[1], axis[0]])
    centre = axis*((a0+a1)/2) + along*((b0+b1)/2)
    eave = top-roof.rise
    radius = max(width/2, length/2, roof.rise)
    segments = max(16, int(math.ceil(math.pi/math.acos(max(-1, 1-.01/radius))/4))*4)
    segments = min(512, segments)
    if roof.shape == 'dome':
        cap = md.Manifold.sphere(1, segments).scale((width/2, length/2, roof.rise))
    elif roof.shape == 'cone':
        cap = md.Manifold.cylinder(1, 1, 0, segments).scale((width/2, length/2, roof.rise))
    else:
        # Cylinder axis becomes the ridge axis; cross-section is an ellipse.
        cap = md.Manifold.cylinder(length, 1, circular_segments=segments, center=True)
        cap = cap.rotate((90, 0, 0)).scale((width/2, 1, roof.rise))
    matrix = np.array([[axis[0], along[0], 0, centre[0]],
                       [axis[1], along[1], 0, centre[1]], [0, 0, 1, eave]])
    cap = cap.transform(matrix)
    walls = solid.trim_by_plane((0, 0, -1), -eave)
    return walls + (solid ^ cap)
