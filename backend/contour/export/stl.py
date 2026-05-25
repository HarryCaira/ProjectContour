"""STL kit serialisation: one STL per part, bundled as a zip with a manifest."""
from __future__ import annotations

import io
import json
import zipfile

from contour.schema.kit import MeshKit


def to_stl_zip(kit: MeshKit) -> bytes:
    """Bundle per-part STLs and a manifest into a zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for part in kit.parts:
            stl_bytes = part.mesh.export(file_type="stl")
            zf.writestr(f"{part.name}.stl", stl_bytes)
        zf.writestr("manifest.json", json.dumps(_manifest(kit), indent=2))
    return buf.getvalue()


def _manifest(kit: MeshKit) -> dict:
    return {
        "parts": [
            {
                "name": part.name,
                "material": {
                    "colour": part.material.colour,
                    "roughness": part.material.roughness,
                    "metalness": part.material.metalness,
                },
                "stats": {
                    "vertices": int(len(part.mesh.vertices)),
                    "faces": int(len(part.mesh.faces)),
                    "volume_m3": float(part.mesh.volume),
                },
            }
            for part in kit.parts
        ]
    }
