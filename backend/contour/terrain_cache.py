"""Reusable terrain geometry, independent of route dimensions and colours."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from uuid import uuid4
from weakref import WeakValueDictionary

import numpy as np
import trimesh

_guard = Lock()
_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()
CACHE_VERSION = 1


def cached_terrain(
    root: Path,
    identity: dict,
    quality_size_mm: float,
    build: Callable[[], trimesh.Trimesh],
    checkpoint: Callable[[], None],
) -> trimesh.Trimesh:
    """Reuse a mesh certified for this size or larger; return independently owned arrays.

    One build per key at a time prevents overlapping slider requests from doing
    duplicate work. Data is numeric NPZ, never pickle. Writes are atomic.
    """
    digest = hashlib.sha256(json.dumps([CACHE_VERSION, identity], sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.npz"
    with _guard:
        lock = _locks.setdefault(str(path.resolve()), Lock())
    while not lock.acquire(timeout=.1):
        checkpoint()
    try:
        checkpoint()
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if float(data["size_mm"]) >= quality_size_mm:
                        return trimesh.Trimesh(vertices=data["vertices"].copy(), faces=data["faces"].copy(), process=False)
            except (OSError, ValueError, KeyError):
                # A stale/corrupt cache is replaceable; the source remains intact.
                pass
        mesh = build()
        checkpoint()
        temporary = root / f".{digest}-{uuid4().hex}.npz"
        try:
            np.savez(temporary, size_mm=quality_size_mm, vertices=mesh.vertices, faces=mesh.faces)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        # Bound this generated cache without touching source uploads or tiles.
        entries = []
        for candidate in root.glob("*.npz"):
            if len(candidate.stem) != 64:
                continue
            try:
                entries.append((candidate.stat().st_mtime, candidate))
            except FileNotFoundError:
                pass
        for _, stale in sorted(entries, reverse=True)[16:]:
            if stale != path:
                stale.unlink(missing_ok=True)
        return mesh
    finally:
        lock.release()
