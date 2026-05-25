"""Tests for the FastAPI app: health, validation, stub responses, error handler."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contour.api.errors import ContourError, register_exception_handlers
from contour.api.server import app

VALID_SETTINGS = {
    "schemaVersion": 1,
    "source": {"type": "gpx", "id": "abc", "sha256": "a" * 64},
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_mesh_returns_422_on_invalid_payload(client):
    r = client.post("/mesh", json={"schemaVersion": 1})  # missing source
    assert r.status_code == 422


def test_mesh_returns_422_on_invalid_schema_version(client):
    r = client.post("/mesh", json={"schemaVersion": 99, "source": VALID_SETTINGS["source"]})
    assert r.status_code == 422


def test_mesh_returns_501_for_valid_stub_payload(client):
    r = client.post("/mesh", json=VALID_SETTINGS)
    assert r.status_code == 501
    assert "Not implemented" in r.json()["detail"]


def test_export_returns_501_for_valid_payload(client):
    r = client.post("/export", json=VALID_SETTINGS)
    assert r.status_code == 501


def test_upload_requires_file(client):
    r = client.post("/upload")
    assert r.status_code == 422


def test_upload_returns_501_with_file(client):
    r = client.post("/upload", files={"file": ("track.gpx", b"<gpx/>")})
    assert r.status_code == 501


def test_contour_error_handler_serialises_payload():
    """The ContourError handler turns a raised error into the expected JSON shape."""
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/boom")
    def boom():
        raise ContourError(
            code="test_error",
            message="Boom!",
            status_code=418,
            details={"x": 1},
        )

    test_client = TestClient(test_app)
    r = test_client.get("/boom")
    assert r.status_code == 418
    assert r.json() == {"code": "test_error", "message": "Boom!", "details": {"x": 1}}


def test_contour_error_handler_empty_details_defaults_to_empty_dict():
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/boom")
    def boom():
        raise ContourError(code="x", message="y", status_code=400)

    test_client = TestClient(test_app)
    r = test_client.get("/boom")
    assert r.json()["details"] == {}
