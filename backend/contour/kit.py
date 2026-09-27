"""MeshKit data model. The output of the pipeline; a labelled set of meshes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import trimesh

PartName = Literal["land", "water", "route", "plinth"]


@dataclass
class Material:
    colour: str
    roughness: float = 0.85
    metalness: float = 0.0


@dataclass
class KitPart:
    name: PartName
    mesh: trimesh.Trimesh
    material: Material
    exportable_as_separate_part: bool = True


@dataclass
class MeshKit:
    parts: list[KitPart] = field(default_factory=list)

    def part(self, name: PartName) -> KitPart | None:
        return next((p for p in self.parts if p.name == name), None)
