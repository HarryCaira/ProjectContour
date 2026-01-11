import io
import time
import requests
from typing import Optional
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry
import numpy as np
from PIL import Image


class TileCache:
    """
    Simple file-based cache for tile PNGs.
    Stores tiles in `<cache_dir>/<z>/<x>/<y>.png`.
    """

    def __init__(self, cache_dir: str = ".tile_cache"):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, z: int, x: int, y: int) -> Path:
        p = self.root / str(z) / str(x)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{y}.png"

    def get(self, z: int, x: int, y: int) -> Optional[bytes]:
        path = self._path(z, x, y)
        if path.exists():
            return path.read_bytes()
        return None

    def set(self, z: int, x: int, y: int, data: bytes) -> None:
        path = self._path(z, x, y)
        path.write_bytes(data)


class MapboxTileClient:
    """
    REST client for fetching Mapbox Terrain-RGB tiles over HTTP.
    Includes:
    - Retry logic
    - 429 rate-limit handling
    - Optional caching
    """

    BASE_URL = "https://api.mapbox.com/v4/mapbox.terrain-rgb"

    def __init__(self, access_token: str, cache: Optional[TileCache] = None):
        self.token = access_token
        self.cache = cache

        # HTTP session with retry strategy
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)

        self.http = requests.Session()
        self.http.mount("https://", adapter)

    # ------------------------------
    # Public API
    # ------------------------------

    def fetch_tile(self, z: int, x: int, y: int) -> bytes:
        """
        Fetch a tile. Returns PNG bytes.
        Uses cache if available.
        """

        # Check cache first
        if self.cache:
            cached = self.cache.get(z, x, y)
            if cached:
                return cached

        url = f"{self.BASE_URL}/{z}/{x}/{y}.pngraw"
        params = {"access_token": self.token}

        response = self.http.get(url, params=params, timeout=10)

        # Handle rate limit backoff manually (429)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            return self.fetch_tile(z, x, y)

        if response.status_code != 200:
            raise RuntimeError(f"Tile fetch failed [{response.status_code}]: {response.text}")

        data = response.content

        # Save to cache
        if self.cache:
            self.cache.set(z, x, y, data)

        return data


def decode_terrain_rgb(png_bytes: bytes) -> np.ndarray:
    """
    Convert Terrain-RGB PNG bytes into a 256x256 elevation matrix (meters).
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    elevation = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
    return elevation
