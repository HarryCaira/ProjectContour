"""Disk-backed cache for fetched tiles and other binary artefacts."""
from __future__ import annotations

from pathlib import Path


class TileCache:
    """Filesystem cache keyed by (provider, layer, z, x, y, ext).

    Layout: <root>/<provider>/<layer>/<z>/<x>/<y>.<ext>
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, provider: str, layer: str, z: int, x: int, y: int, ext: str) -> Path:
        return self.root / provider / layer / str(z) / str(x) / f"{y}.{ext}"

    def get(self, provider: str, layer: str, z: int, x: int, y: int, ext: str) -> bytes | None:
        path = self._path(provider, layer, z, x, y, ext)
        if path.exists():
            return path.read_bytes()
        return None

    def set(self, provider: str, layer: str, z: int, x: int, y: int, ext: str, data: bytes) -> None:
        path = self._path(provider, layer, z, x, y, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
