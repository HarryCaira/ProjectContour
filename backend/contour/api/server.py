"""FastAPI application entry point."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contour.api.errors import register_exception_handlers
from contour.api.routes import router
from contour.http.cache import TileCache
from contour.http.client import HttpClient
from contour.storage.gpx import GpxStore

load_dotenv(find_dotenv(usecwd=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(os.environ.get("CONTOUR_DATA_DIR", "./contour_cache"))
    app.state.gpx_store = GpxStore(data_dir / "gpx")
    app.state.tile_cache = TileCache(data_dir / "tiles")
    app.state.http_client = HttpClient()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ProjectContour Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("contour.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
