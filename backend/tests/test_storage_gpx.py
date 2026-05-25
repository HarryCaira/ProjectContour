"""Tests for the GpxStore."""
from __future__ import annotations

import hashlib

import pytest

from contour.storage.gpx import GpxStore


def test_save_returns_id_and_hash(tmp_path):
    store = GpxStore(tmp_path)
    data = b"<gpx></gpx>"
    gpx_id, sha = store.save(data)
    assert len(gpx_id) > 0
    assert sha == hashlib.sha256(data).hexdigest()


def test_load_returns_saved_bytes(tmp_path):
    store = GpxStore(tmp_path)
    data = b"<gpx>track</gpx>"
    gpx_id, _ = store.save(data)
    assert store.load(gpx_id) == data


def test_load_missing_raises(tmp_path):
    store = GpxStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("does-not-exist")


def test_exists(tmp_path):
    store = GpxStore(tmp_path)
    gpx_id, _ = store.save(b"x")
    assert store.exists(gpx_id)
    assert not store.exists("nope")


def test_verify_hash_matches_saved(tmp_path):
    store = GpxStore(tmp_path)
    data = b"contents"
    gpx_id, sha = store.save(data)
    assert store.verify_hash(gpx_id, sha)
    assert not store.verify_hash(gpx_id, "deadbeef" * 8)


def test_stored_hash_returns_none_for_unknown(tmp_path):
    store = GpxStore(tmp_path)
    assert store.stored_hash("unknown") is None


def test_save_each_call_gives_unique_id(tmp_path):
    store = GpxStore(tmp_path)
    a, _ = store.save(b"a")
    b, _ = store.save(b"b")
    assert a != b
