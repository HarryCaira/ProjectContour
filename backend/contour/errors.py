"""Structured error responses."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ContourError(Exception):
    """Application error with a structured payload."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class MeshDetailLimitError(ContourError, ValueError):
    """A safe, actionable quality-budget failure for previews and exports."""

    def __init__(
        self,
        reason: Literal["vertices", "reference_points", "iterations", "stalled", "tiles"],
        count: int,
        limit: int | None = None,
        *,
        details: dict | None = None,
    ) -> None:
        diagnostics = {"reason": reason, "count": count, "limit": limit, **(details or {})}
        if reason == "vertices":
            message = f"Terrain vertex limit reached: {count:,} vertices (limit {limit:,}). The requested accuracy could not be reached within this limit."
        elif reason == "reference_points":
            message = f"Terrain sampling limit exceeded: {count:,} reference-grid points required (limit {limit:,})."
        elif reason == "iterations":
            message = f"Terrain refinement stopped after {count:,} passes (limit {limit:,}) without reaching the requested accuracy."
        elif reason == "stalled":
            message = f"Terrain refinement stalled at {count:,} vertices: no new vertices could be added, but the requested accuracy has not been reached."
        elif reason == "tiles":
            message = f"Elevation tile budget exceeded: approximately {count:,} tiles estimated at zoom {diagnostics['zoom']} (limit {limit:,})."
        else:
            raise ValueError(f"Unknown terrain limit: {reason}")
        if "max_error_m" in diagnostics:
            message += (f" {diagnostics['triangles_over_tolerance']:,} triangles remain over tolerance."
                        f" Worst sampled height error: {diagnostics['max_error_m']:.6g} m;"
                        f" target: {diagnostics['tolerance_m']:.6g} m (before exaggeration).")
        super().__init__(code="mesh_detail_limit", message=message, status_code=422, details=diagnostics)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ContourError)
    async def _handle_contour_error(request: Request, exc: ContourError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )
