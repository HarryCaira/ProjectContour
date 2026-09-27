"""Structured error responses."""
from __future__ import annotations

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

    def __init__(self):
        super().__init__(
            code="mesh_detail_limit",
            message="This landscape exceeds the current mesh budget. Try a smaller physical size.",
            status_code=422,
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ContourError)
    async def _handle_contour_error(request: Request, exc: ContourError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )
