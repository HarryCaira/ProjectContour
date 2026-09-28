"""Mapped woodland/rock coverage for a non-destructive preview overlay."""
from __future__ import annotations

import base64
import io

import mapbox_vector_tile
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import shape

from contour.biome_data import (_hex_geographic_bbox, _fetch_tile_with_cache,
                                _convert_to_lonlat, _project_to_enu, _flatten_polygons)
from contour.hex_frame import HexFrame
from contour.http_client import HttpClient
from contour.tile_cache import TileCache
from contour.tiles import tiles_covering_bbox


def fetch_landcover(frame: HexFrame, client: HttpClient, cache: TileCache, token: str) -> dict:
    """Reuse Streets v8 tiles; water wins over rock, which wins over woodland."""
    regions: dict[str, list] = {'wood': [], 'rock': [], 'water': []}
    enu = frame.local_enu()
    for tile in tiles_covering_bbox(*_hex_geographic_bbox(frame), 14):
        decoded = mapbox_vector_tile.decode(_fetch_tile_with_cache(client, cache, token, tile))
        for layer_name in ('landuse', 'water'):
            layer = decoded.get(layer_name, {})
            for feature in layer.get('features', []):
                category = 'water' if layer_name == 'water' else feature.get('properties', {}).get('class')
                if category not in regions or feature['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
                    continue
                polygon = shape(_convert_to_lonlat(feature['geometry'], tile, layer.get('extent', 4096)))
                if not polygon.is_valid:
                    polygon = shapely.make_valid(polygon)
                regions[category].append(_project_to_enu(polygon, enu))
    domain = frame.polygon_enu()
    merged = {name: shapely.union_all(items).intersection(domain) for name, items in regions.items()}
    merged['rock'] = merged['rock'].difference(merged['water'])
    merged['wood'] = merged['wood'].difference(shapely.union_all([merged['water'], merged['rock']]))
    return coverage_preview(merged, domain)


def coverage_preview(regions: dict, domain: shapely.Geometry, resolution: int = 1024) -> dict:
    bounds = domain.bounds
    def pixels(ring):
        return [((x - bounds[0]) / (bounds[2] - bounds[0]) * (resolution - 1),
                 (bounds[3] - y) / (bounds[3] - bounds[1]) * (resolution - 1)) for x, y, *_ in ring.coords]
    image = Image.new('RGB', (resolution, resolution))
    percentages = {}
    for name, colour in [('wood', (255, 0, 0)), ('rock', (0, 255, 0))]:
        region = regions[name]
        mask = Image.new('L', image.size)
        draw = ImageDraw.Draw(mask)
        for polygon in _flatten_polygons(region):
            draw.polygon(pixels(polygon.exterior), fill=255)
            for ring in polygon.interiors:
                draw.polygon(pixels(ring), fill=0)
        image.paste(colour, mask=mask)
        percentages[name] = round(100 * region.area / domain.area, 2)
    data = io.BytesIO()
    image.save(data, format='PNG')
    return {'image': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode('ascii'),
            'bounds': list(bounds), 'percentages': percentages, 'zoom': 14}
