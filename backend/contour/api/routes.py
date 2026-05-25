"""HTTP route handlers."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import Response

from contour.export.gltf import to_glb
from contour.export.stl import to_stl_zip
from contour.input.gpx import parse_gpx
from contour.pipeline import PipelineDependencies, build_kit
from contour.schema.settings import Settings
from contour.storage.gpx import GpxStore

router = APIRouter()


def get_gpx_store(request: Request) -> GpxStore:
    return request.app.state.gpx_store


def get_pipeline_deps(request: Request) -> PipelineDependencies:
    token = os.environ.get("MAPBOX_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="MAPBOX_TOKEN not configured")
    return PipelineDependencies(
        http_client=request.app.state.http_client,
        tile_cache=request.app.state.tile_cache,
        mapbox_token=token,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/upload")
async def upload(file: UploadFile, store: GpxStore = Depends(get_gpx_store)) -> dict:
    raw = await file.read()
    try:
        route = parse_gpx(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    gpx_id, sha256 = store.save(raw)
    return {
        "id": gpx_id,
        "sha256": sha256,
        "stats": {
            "points": route.num_points,
            "distance_km": route.distance_km(),
        },
    }


@router.post("/mesh")
def build_mesh(
    settings: Settings,
    store: GpxStore = Depends(get_gpx_store),
    deps: PipelineDependencies = Depends(get_pipeline_deps),
) -> Response:
    _validate_source(store, settings)
    route = parse_gpx(store.load(settings.source.id))
    kit = build_kit(settings, route, deps)
    glb_bytes = to_glb(kit)
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "X-Kit-Parts": ",".join(p.name for p in kit.parts),
            "X-Kit-Triangles": ",".join(str(len(p.mesh.faces)) for p in kit.parts),
        },
    )


@router.post("/export")
def export_kit(
    settings: Settings,
    store: GpxStore = Depends(get_gpx_store),
    deps: PipelineDependencies = Depends(get_pipeline_deps),
) -> Response:
    _validate_source(store, settings)
    route = parse_gpx(store.load(settings.source.id))
    kit = build_kit(settings, route, deps)
    zip_bytes = to_stl_zip(kit)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=contour-kit.zip"},
    )


def _validate_source(store: GpxStore, settings: Settings) -> None:
    if not store.exists(settings.source.id):
        raise HTTPException(status_code=404, detail=f"GPX {settings.source.id} not found")
    if not store.verify_hash(settings.source.id, settings.source.sha256):
        raise HTTPException(status_code=409, detail="Source GPX hash mismatch")
