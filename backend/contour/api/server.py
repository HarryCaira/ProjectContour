"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contour.api.errors import register_exception_handlers
from contour.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="ProjectContour Backend", version="0.1.0")
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
