"""HTTP route handlers. Endpoints are stubs returning 501 until the pipeline is implemented."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from contour.schema.settings import Settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/upload")
async def upload(file: UploadFile) -> dict:
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/mesh")
async def build_mesh(settings: Settings):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/export")
async def export_kit(settings: Settings):
    raise HTTPException(status_code=501, detail="Not implemented")
