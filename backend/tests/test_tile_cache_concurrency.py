from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from contour.tile_cache import TileCache


def test_concurrent_writes_to_same_tile(tmp_path, monkeypatch):
    """Both consumers may finish downloading the same uncached tile together."""
    cache = TileCache(tmp_path)
    barrier = Barrier(2)
    original = Path.replace

    def simultaneous_replace(self, target):
        barrier.wait(timeout=5)
        return original(self, target)

    monkeypatch.setattr(Path, 'replace', simultaneous_replace)
    key = ('mapbox', 'streets', 14, 1, 2, 'pbf')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cache.set, *key, payload) for payload in (b'first', b'second')]
        for future in futures:
            future.result()
    assert cache.get(*key) in (b'first', b'second')
    assert len(list(tmp_path.rglob('*.*'))) == 1
