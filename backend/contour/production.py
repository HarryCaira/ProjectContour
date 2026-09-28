"""Internal print profile; these are production choices, not editor controls."""
NOZZLE_DIAMETER_MM = 0.2
TERRAIN_SAMPLE_MM = NOZZLE_DIAMETER_MM / 2
SURFACE_TOLERANCE_MM = NOZZLE_DIAMETER_MM / 16
# The preview exaggerates geometry locally up to 5x, without another API fetch.
PREVIEW_MAX_EXAGGERATION = 5.0
MAX_TERRAIN_ZOOM = 15  # Terrain-RGB's native detail at 256px per tile.

# One quarter of Terrain-RGB's 0.1 m encoded height increment; independent of nozzle.
SOURCE_ROUTE_TOLERANCE_M = 0.025
