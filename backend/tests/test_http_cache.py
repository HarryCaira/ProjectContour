"""Tests for the disk-backed TileCache."""
from __future__ import annotations

from contour.http.cache import TileCache


def test_cache_returns_none_for_missing(tmp_path):
    cache = TileCache(tmp_path)
    assert cache.get("p", "l", 1, 2, 3, "png") is None


def test_cache_set_then_get(tmp_path):
    cache = TileCache(tmp_path)
    cache.set("mapbox", "terrain-rgb", 14, 8192, 5446, "png", b"binary data")
    assert cache.get("mapbox", "terrain-rgb", 14, 8192, 5446, "png") == b"binary data"


def test_cache_path_layout(tmp_path):
    cache = TileCache(tmp_path)
    cache.set("mapbox", "streets-v8", 12, 100, 200, "mvt", b"x")
    expected = tmp_path / "mapbox" / "streets-v8" / "12" / "100" / "200.mvt"
    assert expected.exists()
    assert expected.read_bytes() == b"x"


def test_cache_set_overwrites(tmp_path):
    cache = TileCache(tmp_path)
    cache.set("p", "l", 1, 2, 3, "png", b"first")
    cache.set("p", "l", 1, 2, 3, "png", b"second")
    assert cache.get("p", "l", 1, 2, 3, "png") == b"second"


def test_cache_different_keys_isolated(tmp_path):
    cache = TileCache(tmp_path)
    cache.set("p", "l", 1, 2, 3, "png", b"a")
    cache.set("p", "l", 1, 2, 4, "png", b"b")
    cache.set("p", "l", 2, 2, 3, "png", b"c")
    cache.set("p", "m", 1, 2, 3, "png", b"d")
    assert cache.get("p", "l", 1, 2, 3, "png") == b"a"
    assert cache.get("p", "l", 1, 2, 4, "png") == b"b"
    assert cache.get("p", "l", 2, 2, 3, "png") == b"c"
    assert cache.get("p", "m", 1, 2, 3, "png") == b"d"
