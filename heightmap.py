from tile_client import MapboxTileClient, decode_terrain_rgb
import numpy as np
from route import RasterTile


def create_heightmap_from_tiles(client: MapboxTileClient, tile_coords: list[RasterTile]) -> np.ndarray:
    """
    Fetch multiple tiles and stitch them into a single heightmap.

    Args:
        client: MapboxTileClient instance
        tile_coords: List of Tile instances

    Returns:
        2D numpy array of elevations in meters
    """
    if not tile_coords:
        raise ValueError("No tiles specified")

    # Group by zoom level (all tiles should be at same zoom)
    z_values = {tile.zoom for tile in tile_coords}
    if len(z_values) > 1:
        raise ValueError(f"All tiles must be at same zoom level, got: {z_values}")

    # Fetch all tiles
    tiles = {}
    for tile in tile_coords:
        print(f"Fetching tile {tile.zoom}/{tile.x}/{tile.y}...")
        png_bytes = client.fetch_tile(tile.zoom, tile.x, tile.y)
        tiles[(tile.x, tile.y)] = decode_terrain_rgb(png_bytes)

    # Find grid bounds
    xs = [x for x, _ in tiles.keys()]
    ys = [y for _, y in tiles.keys()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Stitch tiles together
    height = (max_y - min_y + 1) * 256
    width = (max_x - min_x + 1) * 256
    heightmap = np.zeros((height, width), dtype=np.float32)

    for (x, y), tile_data in tiles.items():
        row = y - min_y
        col = x - min_x
        heightmap[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = tile_data

    print(f"Created heightmap: {heightmap.shape} ({len(tiles)} tiles)")
    print(f"Elevation range: {heightmap.min():.1f}m to {heightmap.max():.1f}m")

    return heightmap
