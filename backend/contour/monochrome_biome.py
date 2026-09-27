"""Monochrome biome style: muted earth palette with a contrasting accent for the route."""
from __future__ import annotations

from dataclasses import replace

from contour.kit import KitPart, Material, MeshKit
from contour.settings import Settings
from contour.style import NeutralScene, Style


class MonochromeBiome(Style):
    LAND = Material(colour="#7a8060", roughness=0.85)
    WATER = Material(colour="#6a8aa0", roughness=0.6)
    ROUTE = Material(colour="#c44545", roughness=0.55)
    PLINTH = Material(colour="#2a2a2a", roughness=0.9)

    def apply(self, scene: NeutralScene, settings: Settings) -> MeshKit:
        parts: list[KitPart] = [KitPart(name="land", mesh=scene.land, material=replace(self.LAND, colour=settings.style.colours.terrain))]
        if scene.water is not None:
            parts.append(KitPart(name="water", mesh=scene.water, material=replace(self.WATER, colour=settings.style.colours.water)))
        if scene.route is not None:
            parts.append(KitPart(name="route", mesh=scene.route, material=replace(self.ROUTE, colour=settings.style.colours.route)))
        if scene.plinth is not None:
            parts.append(KitPart(name="plinth", mesh=scene.plinth, material=self.PLINTH))
        return MeshKit(parts=parts)
