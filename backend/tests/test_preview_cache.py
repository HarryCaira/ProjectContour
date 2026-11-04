from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from contour import preview_cache as module
from contour.route import Route
from contour.settings import Settings
from contour.tile_cache import TileCache


@pytest.fixture
def args(tmp_path):
    with module._guard:
        module._entries.clear()
    return (Settings.model_validate({'source': {'id': 'test', 'sha256': 'a' * 64}}),
            Route(np.array([0., 1.]), np.array([0., 1.]), np.zeros(2)),
            SimpleNamespace(tile_cache=TileCache(tmp_path)))


def test_reuses_completed_preview_but_invalidates_size(args):
    calls = []
    def build():
        calls.append(1)
        return {'binary': b'glTF'}
    assert module.cached_preview(*args, build, lambda: None)['binary'] == b'glTF'
    module.cached_preview(*args, build, lambda: None)
    assert len(calls) == 1
    settings = args[0].model_copy(deep=True)
    settings.physical.size_mm = 100
    module.cached_preview(settings, *args[1:], build, lambda: None)
    assert len(calls) == 2


def test_concurrent_requests_generate_once(args):
    entered, release, waiting = Event(), Event(), Event()
    calls = []
    def build():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {'binary': b'glTF'}
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(module.cached_preview, *args, build, lambda: None)
        assert entered.wait(5)
        second = pool.submit(module.cached_preview, *args, build, waiting.set)
        assert waiting.wait(5)
        release.set()
        assert first.result() == second.result()
    assert len(calls) == 1


def test_failed_or_cancelled_preview_is_not_cached(args):
    def fail():
        raise ValueError('broken')
    with pytest.raises(ValueError):
        module.cached_preview(*args, fail, lambda: None)
    assert not module._entries
    def cancel():
        raise InterruptedError()
    with pytest.raises(InterruptedError):
        module.cached_preview(*args, lambda: {'binary': b'glTF'}, cancel)
    assert not module._entries


def test_cache_has_byte_budget(args, monkeypatch):
    monkeypatch.setattr(module, 'MAX_BYTES', 5)
    for size in (100, 150):
        settings = args[0].model_copy(deep=True)
        settings.physical.size_mm = size
        module.cached_preview(settings, *args[1:], lambda: {'binary': b'glTF'}, lambda: None)
    assert len(module._entries) == 1


def test_stream_consuming_event_does_not_mutate_cached_preview(args):
    def build():
        return {'binary': b'glTF', 'metadata': {}}
    first = module.cached_preview(*args, build, lambda: None)
    assert first.pop('binary') == b'glTF'
    second = module.cached_preview(*args, build, lambda: None)
    assert second.pop('binary') == b'glTF'
    assert module.cached_preview(*args, build, lambda: None)['binary'] == b'glTF'
