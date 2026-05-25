"""Disk-backed cache for fetched tiles and other artefacts."""
from __future__ import annotations

from pathlib import Path


class TileCache:
    def __init__(self, root: Path) -> None:
        raise NotImplementedError
