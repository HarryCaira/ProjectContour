# ProjectContour

Turn a GPX route into a printable hexagonal landscape model.

## Structure

```text
backend/contour/      Python API and generation modules, in one flat package
backend/tests/        Backend tests
console/app/         Next.js page, layout, and styles
console/components/  Editor and 3D viewer components
console/lib/         API client, hooks, settings, and editor state
PRD.md                Product requirements
TECHNICAL_DESIGN.md    Architecture and module guide
```

Backend entry points are `contour/server.py` for the API and
`contour/pipeline.py` for model generation. The frontend starts at `app/page.tsx`.
The earlier `reference/` prototype has been removed; it remains in Git history.

## Production detail

The editor uses an internal 0.2 mm nozzle profile (`backend/contour/production.py`),
without exposing printer settings. Terrain fetches target 0.1 mm sample spacing
at model scale, capped at Terrain-RGB's native zoom 15; existing tile caches are
reused. The legacy `physical.resolutionMm` API field remains accepted but does
not change this production profile.

Terrain triangles are refined against a reference grid and edge/interior probes
with a 0.025 mm vertical error target at final scale. Refinement accounts for up
to 5x exaggeration so the preview can adjust relief immediately. Flat areas stay
coarse. Routes use physical-error simplification instead of a fixed 500-point
limit. These tolerances measure approximation of the available source data,
not its geographic accuracy. Budget exhaustion fails the build instead of
silently exporting a lower-quality mesh.

## Interactive editing

Route width and height are previewed locally using centreline offsets stored in
the GLB. Colours and exaggeration are local too. These edits make no mesh request.
Size changes scale the existing preview immediately; increases request detail
in the background after 350 ms of inactivity, while decreases retain the more
detailed geometry. Export always uses the actual requested physical dimensions.

The backend caches unscaled terrain separately from route dimensions and colours
under the tile cache's `_derived/terrain_meshes` directory (up to 16 entries).
Cached geometry is copied before transforms, is reusable at smaller sizes, and
is built once per geometry key when requests overlap. Terrain face winding is
constructed directly, avoiding a whole-mesh normal repair on every build.

TODO:
- The GPX reader does not handle multiple routes/tracks well. This means a lot of more complicated GPX files aren't read well.
