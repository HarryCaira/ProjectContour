from __future__ import annotations
import click
from coordinate_transform import LonLatToENU, LonLatToRasterTile
from route import Route_LL, BBox_LL, ZoomLevel, Route_ENU
from tile_client import MapboxTileClient, TileCache
from heightmap import create_heightmap_from_tiles
from parameters import GlobalParameters, ModelResolution
from terrain_data import Terrain_ENU
from mesh_generator import MeshGenerator
from visualization import VisualizationBuilder


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--manual-zoom", "-z", default=None, type=int, help="Tile zoom level (auto-calculated if not specified)")
@click.option("--model-size", "-s", default=100.0, help="Target model size in mm (default: 100)")
@click.option("--resolution", "-r", default=0.2, help="Target print resolution in mm (default: 0.2)")
def main(file_path: str, manual_zoom: int | None, model_size: float, resolution: float) -> None:
    client = MapboxTileClient(access_token="pk.eyJ1IjoiaGFycnljYWlyYSIsImEiOiJjbWlzeWk4NmwwcmxtM2ZxeTZycGY2b2JqIn0.U2wkVaheUTFwrhiwBzLX3Q", cache=TileCache("./terrain_cache"))

    params = GlobalParameters(SIZE_MM=model_size, PRINT_RESOLUTION_MM=resolution)
    
    route_ll = Route_LL.new(gpx_file_path=file_path)
    bbox_ll = BBox_LL.new(route_ll.bounds)

    model_res = ModelResolution.new(params=params, latitude_span=bbox_ll.latitude_span, longitude_span=bbox_ll.longitude_span, central_latitude=bbox_ll.central_latitude)
    zoom = ZoomLevel.new(params=params, bbox=bbox_ll, manual_zoom=manual_zoom, model_res=model_res)
    
    ll_to_rt = LonLatToRasterTile()
    tiles = bbox_ll.tiles_to_cover(zoom.value, transform=ll_to_rt)
    heightmap = create_heightmap_from_tiles(client, tiles)

    ll_to_enu = LonLatToENU.new(origin=route_ll.origin)
    route_enu = Route_ENU.new(route=route_ll, transform=ll_to_enu)
    terrain_enu = Terrain_ENU.new(heightmap=heightmap, tiles=tiles, zoom=zoom.value, transform=ll_to_enu)

    mesh_gen = MeshGenerator.new(terrain=terrain_enu, route_enu=route_enu)
    mesh_gen.export_meshes(target_size_mm=model_size)

    # viz = VisualizationBuilder.new(terrain=terrain_enu, route_enu=route_enu, route_terrain_elevation=mesh_gen.route_terrain_elevation)
    # viz.show(downsample_size=500)


if __name__ == "__main__":
    main()
