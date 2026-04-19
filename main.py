from __future__ import annotations
import logging
import os
from pathlib import Path
import click
from coordinate_transform import LonLatToENU
from route import LatLonRoute, ZoomLevel, EnuRoute
from tile_client import MapboxTileClient, TileCache
from heightmap import create_heightmap_from_tiles
from parameters import GlobalParameters, ModelResolution
from terrain_data import EnuTerrain
from mesh_generator import build_terrain_mesh, build_route_mesh

log = logging.getLogger(__name__)


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--manual-zoom", "-z", default=None, type=int, help="Tile zoom level (auto-calculated if not specified)")
@click.option("--model-size", "-s", default=100.0, help="Target model size in mm (default: 100)")
@click.option("--resolution", "-r", default=0.2, help="Target print resolution in mm (default: 0.2)")
@click.option("--output-dir", "-o", default=".", type=click.Path(file_okay=False), help="Directory for output STL files (default: current dir)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging")
def main(file_path: str, manual_zoom: int | None, model_size: float, resolution: float, output_dir: str, verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    access_token = os.environ.get("MAPBOX_TOKEN")
    if not access_token:
        raise click.ClickException("MAPBOX_TOKEN environment variable is not set. Get a token at https://account.mapbox.com/access-tokens/")
    client = MapboxTileClient(access_token=access_token, cache=TileCache("./terrain_cache"))

    params = GlobalParameters(SIZE_MM=model_size, PRINT_RESOLUTION_MM=resolution)

    route = LatLonRoute.new(gpx_file_path=file_path)
    bbox_ll = route.bbox

    model_res = ModelResolution.new(params=params, latitude_span=bbox_ll.latitude_span, longitude_span=bbox_ll.longitude_span, central_latitude=bbox_ll.central_latitude)
    zoom = ZoomLevel.new(params=params, bbox=bbox_ll, manual_zoom=manual_zoom, model_res=model_res)

    tiles = bbox_ll.tiles_to_cover(zoom.value)
    heightmap = create_heightmap_from_tiles(client, tiles)

    ll_to_enu = LonLatToENU.new(origin=route.origin)
    route_enu = EnuRoute.new(route=route, transform=ll_to_enu)
    terrain_enu = EnuTerrain.new(heightmap=heightmap, tiles=tiles, zoom=zoom.value, transform=ll_to_enu)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    terrain_mesh = build_terrain_mesh(terrain_enu, target_size_mm=model_size)
    terrain_path = out / "terrain_model.stl"
    terrain_mesh.export(terrain_path)
    log.info("Exported terrain: %s (%d vertices, %d faces, %.2f mm³)", terrain_path, len(terrain_mesh.vertices), len(terrain_mesh.faces), terrain_mesh.volume)

    route_mesh = build_route_mesh(route_enu, terrain_enu, target_size_mm=model_size)
    route_path = out / "route_model.stl"
    route_mesh.export(route_path)
    log.info("Exported route: %s (%d vertices, %d faces, %.2f mm³)", route_path, len(route_mesh.vertices), len(route_mesh.faces), route_mesh.volume)


if __name__ == "__main__":
    main()
