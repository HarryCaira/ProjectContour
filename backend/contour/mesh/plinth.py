"""Plinth mesh: hex prism beneath the model."""
from __future__ import annotations

import trimesh

from contour.framing.hex import HexFrame


def build_plinth_mesh(hex_frame: HexFrame, height_m: float, top_z: float) -> trimesh.Trimesh:
    """Build a closed hex prism extending downward from `top_z` by `height_m`.

    The prism's top face sits at `top_z` and its bottom at `top_z - height_m`.
    """
    if height_m <= 0:
        raise ValueError("Plinth height must be positive")

    polygon = hex_frame.polygon_enu()
    mesh = trimesh.creation.extrude_polygon(polygon, height=height_m)
    # extrude_polygon places the base at z=0 and the top at z=height.
    # Shift so the top is at `top_z`.
    mesh.apply_translation([0.0, 0.0, top_z - height_m])
    return mesh
