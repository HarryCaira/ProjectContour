"""Monochrome biome style: muted earth palette with a contrasting accent for the route."""
from __future__ import annotations

from contour.schema.kit import KitPart, Material, MeshKit
from contour.schema.settings import Settings
from contour.styles.base import NeutralScene, Style


class MonochromeBiome(Style):
    LAND = Material(colour="#7a8060", roughness=0.85)
    WATER = Material(colour="#6a8aa0", roughness=0.6)
    ROUTE = Material(colour="#c44545", roughness=0.55)
    PLINTH = Material(colour="#2a2a2a", roughness=0.9)

    def apply(self, scene: NeutralScene, settings: Settings) -> MeshKit:
        parts: list[KitPart] = [KitPart(name="land", mesh=scene.land, material=self.LAND)]
        if scene.water is not None:
            parts.append(KitPart(name="water", mesh=scene.water, material=self.WATER))
        if scene.route is not None:
            parts.append(KitPart(name="route", mesh=scene.route, material=self.ROUTE))
        if scene.plinth is not None:
            parts.append(KitPart(name="plinth", mesh=scene.plinth, material=self.PLINTH))
        return MeshKit(parts=parts)
