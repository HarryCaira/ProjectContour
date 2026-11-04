"""Bounded completed previews, with cancellation-aware in-flight deduplication."""
from collections import OrderedDict
from threading import Lock
from weakref import WeakValueDictionary
import hashlib
import json

MAX_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 4
_guard = Lock()
_locks = WeakValueDictionary()
_entries = OrderedDict()


def cached_preview(settings, route, deps, build, checkpoint):
    # Include actual source coordinates as well as settings: uploaded IDs/hashes
    # alone are not trusted to describe an unchanged route.
    digest = hashlib.sha256(settings.model_dump_json().encode())
    digest.update(str(deps.tile_cache.root.resolve()).encode())
    for array in (route.latitudes, route.longitudes, route.elevations):
        if array is not None:
            digest.update(array.tobytes())
    digest.update(json.dumps(route.segment_starts).encode())
    key = digest.digest()
    with _guard:
        lock = _locks.setdefault(key, Lock())
    while not lock.acquire(timeout=.1):
        checkpoint()
    try:
        checkpoint()
        with _guard:
            cached = _entries.get(key)
            if cached is not None:
                _entries.move_to_end(key)
                return dict(cached)
        result = build()
        checkpoint()
        if len(result['binary']) <= MAX_BYTES:
            with _guard:
                _entries[key] = result
                while (len(_entries) > MAX_ENTRIES or
                       sum(len(value['binary']) for value in _entries.values()) > MAX_BYTES):
                    _entries.popitem(last=False)
        # Streaming consumes the event dictionary (including popping binary).
        # Keep the cached envelope private; immutable GLB bytes can be shared.
        return dict(result)
    finally:
        lock.release()
