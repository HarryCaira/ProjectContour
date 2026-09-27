"""Tests for the FastAPI app: health, validation, upload + mesh + export flow."""
from __future__ import annotations

from contour.errors import MeshDetailLimitError

import base64
import json
import io
import zipfile

import mapbox_vector_tile
import numpy as np
import pytest
import responses
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from contour.errors import ContourError, register_exception_handlers

SIMPLE_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk>
    <name>Test track</name>
    <trkseg>
      <trkpt lat="0.0" lon="0.0"><ele>0.0</ele></trkpt>
      <trkpt lat="0.0005" lon="0.0005"><ele>0.0</ele></trkpt>
      <trkpt lat="0.001" lon="0.001"><ele>0.0</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


def _make_png() -> bytes:
    rgb = np.full((256, 256, 3), (0, 0, 100), dtype=np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_empty_mvt() -> bytes:
    return mapbox_vector_tile.encode([{"name": "water", "features": []}])


def _add_mapbox_mocks() -> None:
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.terrain-rgb/.*"),
        body=_make_png(),
        status=200,
    )
    responses.add(
        responses.GET,
        responses.matchers.re.compile(r"https://api\.mapbox\.com/v4/mapbox\.mapbox-streets-v8/.*"),
        body=_make_empty_mvt(),
        status=200,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTOUR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAPBOX_TOKEN", "test-token")
    from contour.server import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------- Health & validation ----------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_mesh_returns_422_on_invalid_payload(client):
    r = client.post("/mesh", json={"schemaVersion": 1})
    assert r.status_code == 422


def test_mesh_returns_422_on_invalid_schema_version(client):
    r = client.post(
        "/mesh",
        json={"schemaVersion": 99, "source": {"type": "gpx", "id": "x", "sha256": "a" * 64}},
    )
    assert r.status_code == 422


def test_upload_requires_file(client):
    r = client.post("/upload")
    assert r.status_code == 422


# ---------- Upload ----------


def test_upload_returns_id_and_stats(client):
    r = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)})
    assert r.status_code == 200
    body = r.json()
    assert len(body["id"]) > 0
    assert len(body["sha256"]) == 64
    assert body["stats"]["points"] == 3
    assert body["stats"]["distance_km"] > 0


def test_upload_rejects_invalid_gpx(client):
    r = client.post("/upload", files={"file": ("bad.gpx", b"not xml at all")})
    assert r.status_code == 400


# ---------- Mesh & Export flow ----------


@responses.activate
def test_mesh_end_to_end(client):
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    _add_mapbox_mocks()

    r = client.post(
        "/mesh",
        json={
            "schemaVersion": 1,
            "source": {"type": "gpx", "id": upload["id"], "sha256": upload["sha256"]},
            "physical": {"sizeMm": 150, "resolutionMm": 2.0},
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "model/gltf-binary"
    assert r.content.startswith(b"glTF")
    assert "land" in r.headers["x-kit-parts"]


@responses.activate
def test_export_end_to_end(client):
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    _add_mapbox_mocks()

    r = client.post(
        "/export",
        json={
            "schemaVersion": 1,
            "source": {"type": "gpx", "id": upload["id"], "sha256": upload["sha256"]},
            "physical": {"sizeMm": 150, "resolutionMm": 2.0},
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert any(n.endswith(".stl") for n in names)


def test_mesh_404_for_unknown_source(client):
    r = client.post(
        "/mesh",
        json={
            "schemaVersion": 1,
            "source": {"type": "gpx", "id": "unknown", "sha256": "a" * 64},
            "physical": {"sizeMm": 150, "resolutionMm": 2.0},
        },
    )
    assert r.status_code == 404


def test_mesh_409_on_hash_mismatch(client):
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    r = client.post(
        "/mesh",
        json={
            "schemaVersion": 1,
            "source": {"type": "gpx", "id": upload["id"], "sha256": "b" * 64},
            "physical": {"sizeMm": 150, "resolutionMm": 2.0},
        },
    )
    assert r.status_code == 409


def test_mesh_500_when_token_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTOUR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    from contour.server import create_app

    app = create_app()
    with TestClient(app) as c:
        upload = c.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
        r = c.post(
            "/mesh",
            json={
                "schemaVersion": 1,
                "source": {"type": "gpx", "id": upload["id"], "sha256": upload["sha256"]},
            },
        )
        assert r.status_code == 500
        assert "MAPBOX_TOKEN" in r.json()["detail"]


# ---------- ContourError handler (independent app) ----------


def test_contour_error_handler_serialises_payload():
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/boom")
    def boom():
        raise ContourError(code="test_error", message="Boom!", status_code=418, details={"x": 1})

    with TestClient(test_app) as c:
        r = c.get("/boom")
    assert r.status_code == 418
    assert r.json() == {"code": "test_error", "message": "Boom!", "details": {"x": 1}}


def test_contour_error_handler_empty_details_defaults_to_empty_dict():
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/boom")
    def boom():
        raise ContourError(code="x", message="y", status_code=400)

    with TestClient(test_app) as c:
        r = c.get("/boom")
    assert r.json()["details"] == {}


@responses.activate
def test__build_mesh_stream__reports_stages_and_returns_glb(client):
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    _add_mapbox_mocks()
    response = client.post("/mesh/stream", json={
        "source": {"id": upload["id"], "sha256": upload["sha256"]},
        "physical": {"sizeMm": 150, "resolutionMm": 2.0},
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["stage"] for event in events if event["type"] == "progress"] == [
        "frame", "terrain", "water", "land", "details", "preview",
    ]
    assert events[-1]["type"] == "result"
    assert base64.b64decode(events[-1]["glb"]).startswith(b"glTF")
    assert "land" in events[-1]["metadata"]["parts"]
    assert len(events[-1]["metadata"]["parts"]) == len(events[-1]["metadata"]["triangles"])


def test__build_mesh_stream__validates_before_streaming(client):
    response = client.post("/mesh/stream", json={"source": {"id": "missing", "sha256": "a" * 64}})
    assert response.status_code == 404


def test__build_mesh_stream__reports_failure_without_exposing_provider_credentials(client, monkeypatch):
    def fail(*args, **kwargs):
        kwargs["on_progress"]("terrain")
        raise RuntimeError("provider url?access_token=private-secret")

    monkeypatch.setattr("contour.mesh_progress.build_kit", fail)
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    response = client.post("/mesh/stream", json={"source": {"id": upload["id"], "sha256": upload["sha256"]}})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0] == {"type": "progress", "stage": "terrain"}
    assert events[-1]["type"] == "error"
    assert "private-secret" not in response.text


def test_mesh_stream_explains_detail_limit(client, monkeypatch):
    upload = client.post("/upload", files={"file": ("track.gpx", SIMPLE_GPX)}).json()
    def fail(*args, **kwargs):
        raise MeshDetailLimitError()
    monkeypatch.setattr("contour.mesh_progress.build_kit", fail)
    response = client.post("/mesh/stream", json={
        "source": {"id": upload["id"], "sha256": upload["sha256"]},
    })
    event = json.loads(response.text.splitlines()[-1])
    assert event["type"] == "error"
    assert event["code"] == "mesh_detail_limit"
    assert "smaller physical size" in event["message"]
    assert "try again" not in event["message"].lower()
