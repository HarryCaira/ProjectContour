"""Monochrome biome style: muted earth palette with a contrasting accent for the route."""
from __future__ import annotations

from contour.schema.kit import MeshKit
from contour.schema.settings import Settings
from contour.styles.base import NeutralScene, Style


class MonochromeBiome(Style):
    def apply(self, scene: NeutralScene, settings: Settings) -> MeshKit:
        raise NotImplementedError
