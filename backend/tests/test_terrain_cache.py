from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import trimesh

from contour.terrain_cache import cached_terrain


def test_cache_reuses_larger_mesh_without_mutating_saved_geometry(tmp_path):
    calls = []
    def build():
        calls.append(1)
        return trimesh.creation.box()
    args = dict(root=tmp_path, identity={"frame": 1}, build=build, checkpoint=lambda: None)
    first = cached_terrain(quality_size_mm=150, **args)
    first.apply_scale(8)
    second = cached_terrain(quality_size_mm=100, **args)
    assert np.allclose(second.extents, [1, 1, 1])
    assert len(calls) == 1
    cached_terrain(quality_size_mm=200, **args)
    assert len(calls) == 2
    cached_terrain(quality_size_mm=200, **{**args, "identity": {"frame": 2}})
    assert len(calls) == 3


def test_concurrent_requests_share_one_terrain_build(tmp_path):
    started, release = Event(), Event()
    calls = []
    def build():
        calls.append(1)
        started.set()
        assert release.wait(5)
        return trimesh.creation.box()
    def request():
        return cached_terrain(tmp_path, {"frame": 1}, 100, build, lambda: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request)
        assert started.wait(5)
        second = pool.submit(request)
        release.set()
        assert first.result().is_volume
        assert second.result().is_volume
    assert len(calls) == 1
