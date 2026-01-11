from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GlobalParameters:
    SIZE_MM: float
    PRINT_RESOLUTION_MM: float
    EARTH_CIRCUMFERENCE_M: float = 40075017.0
    M_PER_DEG_LAT: float = 111320.0  # Approx meters per degree at equator


@dataclass(frozen=True)
class ModelResolution:
    """
    Represents the size of real world features that correspond to
    a single voxel/pixel in the 3D model.
    """

    meters: float

    @classmethod
    def new(cls, params: GlobalParameters, latitude_span: float, longitude_span: float, central_latitude: float) -> ModelResolution:
        world_height_m = latitude_span * params.M_PER_DEG_LAT
        world_width_m = longitude_span * (params.M_PER_DEG_LAT * math.cos(math.radians(central_latitude)))

        world_max_dimension = max(world_height_m, world_width_m)
        world_to_model_ratio = world_max_dimension / params.SIZE_MM

        model_res_m = params.PRINT_RESOLUTION_MM * world_to_model_ratio
        return cls(meters=model_res_m)
