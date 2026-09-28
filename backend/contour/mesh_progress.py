"""Stream pipeline milestones and the completed preview over one request."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from threading import Event

from contour.errors import ContourError
from contour.gltf_export import to_glb
from contour.pipeline import PipelineDependencies, build_kit
from contour.route import Route
from contour.settings import Settings


async def stream_mesh(settings: Settings, route: Route, deps: PipelineDependencies) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[dict] = asyncio.Queue()
    cancelled = Event()

    def send(event: dict) -> None:
        if not cancelled.is_set():
            loop.call_soon_threadsafe(events.put_nowait, event)

    last_stage: str | None = None

    def progress(stage: str) -> None:
        nonlocal last_stage
        if cancelled.is_set():
            raise InterruptedError("Preview request cancelled")
        if stage != last_stage:
            last_stage = stage
            send({"type": "progress", "stage": stage})

    def build() -> None:
        try:
            kit = build_kit(settings, route, deps, on_progress=progress)
            progress("preview")
            glb = to_glb(kit)
            send({
                "type": "result", "glb": base64.b64encode(glb).decode("ascii"),
                "metadata": {
                    "parts": [part.name for part in kit.parts],
                    "triangles": [len(part.mesh.faces) for part in kit.parts],
                },
            })
        except InterruptedError:
            return
        except ContourError as error:
            send({"type": "error", "code": error.code, "message": error.message, "details": error.details})
        except Exception as error:
            logging.getLogger(__name__).error("Model build failed: %s", type(error).__name__)
            # Provider exceptions can contain credential-bearing URLs.
            send({"type": "error", "message": "We couldn't build this model. Please try again."})

    task = asyncio.create_task(asyncio.to_thread(build))
    try:
        while True:
            try:
                event = await asyncio.wait_for(events.get(), timeout=10)
            except TimeoutError:
                yield json.dumps({"type": "heartbeat"}) + "\n"
                continue
            yield json.dumps(event) + "\n"
            if event["type"] in {"result", "error"}:
                break
    finally:
        cancelled.set()
        task.cancel()
