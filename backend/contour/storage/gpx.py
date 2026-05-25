"""Filesystem-backed storage for uploaded GPX files.

Each upload is keyed by a UUID and stored as two files: <id>.gpx (raw bytes)
and <id>.sha256 (hex digest sidecar). The sidecar lets us verify a
Settings.source.sha256 matches the stored bytes without re-hashing.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


class GpxStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes) -> tuple[str, str]:
        """Persist `data` and return `(id, sha256)`."""
        gpx_id = str(uuid.uuid4())
        sha256 = hashlib.sha256(data).hexdigest()
        (self.root / f"{gpx_id}.gpx").write_bytes(data)
        (self.root / f"{gpx_id}.sha256").write_text(sha256)
        return gpx_id, sha256

    def exists(self, gpx_id: str) -> bool:
        return (self.root / f"{gpx_id}.gpx").exists()

    def load(self, gpx_id: str) -> bytes:
        path = self.root / f"{gpx_id}.gpx"
        if not path.exists():
            raise FileNotFoundError(f"GPX {gpx_id} not found")
        return path.read_bytes()

    def stored_hash(self, gpx_id: str) -> str | None:
        path = self.root / f"{gpx_id}.sha256"
        if not path.exists():
            return None
        return path.read_text().strip()

    def verify_hash(self, gpx_id: str, expected_sha256: str) -> bool:
        stored = self.stored_hash(gpx_id)
        return stored is not None and stored == expected_sha256
