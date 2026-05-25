"""Style strategy interface. The seam that lets future styles (topographic, realistic) plug in."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import trimesh

from contour.schema.kit import MeshKit
from contour.schema.settings import Settings


@dataclass
class NeutralScene:
    """Intermediate representation before a style is applied.

    Each field is a watertight mesh in the shared ENU coordinate frame; styles
    receive this and decide how to colour, finish, and combine them into a MeshKit.
    """

    land: trimesh.Trimesh
    water: trimesh.Trimesh | None
    route: trimesh.Trimesh | None
    plinth: trimesh.Trimesh


class Style(ABC):
    """A strategy that turns a NeutralScene into a styled MeshKit."""

    @abstractmethod
    def apply(self, scene: NeutralScene, settings: Settings) -> MeshKit: ...
