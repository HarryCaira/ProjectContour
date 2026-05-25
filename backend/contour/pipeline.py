"""Pipeline orchestration: Settings -> MeshKit."""
from __future__ import annotations

from contour.schema.kit import MeshKit
from contour.schema.settings import Settings


def build_kit(settings: Settings) -> MeshKit:
    """Run the full pipeline and produce a MeshKit.

    Stages:
    1. Load route (input)
    2. Frame route into a hex region (framing)
    3. Fetch terrain heightmap (data)
    4. Fetch water polygons (data)
    5. Generate neutral meshes (mesh)
    6. Apply the selected style (styles)
    """
    raise NotImplementedError
