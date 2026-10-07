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
with a 0.0125 mm vertical error target at final scale. Refinement accounts for up
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
Cached geometry is copied before transforms and built once per geometry key
when requests overlap. Its key includes physical size because the smoothing
budget is expressed in print millimetres. Terrain face winding is
constructed directly, avoiding a whole-mesh normal repair on every build.

GPX uploads preserve all non-empty track segments, including recording breaks.
Distance calculations and route ribbons do not bridge those breaks.

## Terrain surface conditioning

A two-pixel Gaussian filter reduces fine raster noise. Its elevation correction
is smoothly bounded to 0.05 mm at 1x relief (up to 0.25 mm at 5x); cached source
tiles are unchanged. This suppresses artifacts rather than adding survey detail.
Lakes use their median sampled bank elevation instead of the lowest outlier.
Their banks blend with a smoothstep transition at least eight native pixels wide,
expanding for larger height differences. Terrain and routes share this adjusted
surface; the shoreline deformation is separate from the general smoothing bound.

The internal `maximumSourceDetail` setting bypasses smoothing, requests native zoom 15,
and triangulates native raster sample locations directly, independent of nozzle
size. It preserves the unsmoothed sample heights away from shoreline transitions;
triangle interiors interpolate linearly rather than chasing a centimetre-level
fit to bilinear interpolation. Cut boundaries are sampled at native spacing. Shoreline transitions and smooth preview
lighting remain enabled. The choice also applies to exports. The two modes have
separate geometry caches. This setting is not exposed in the editor. Tile and
mesh budgets still apply, and oversized requests fail instead of downsampling.

Adaptive refinement in normal mode spends each batch on the largest sampled
height errors first, reserving vertex capacity rather than refining all failing
triangles in the triangulator's internal order.


## Editor and data repairs

The editor offers Medium (100 mm, default) and Large (150 mm) sizes. Route width
and raised height default to 1 mm. Dimension labels dim behind the model, and
the dimensions toggle shows or hides them immediately.

Woodland and rock coverage uses cached Mapbox Streets tiles through `/landcover`.
Water and land cover use at most 256 vector tiles per layer and area, reducing
zoom for regional routes while retaining zoom 14 for local models. Roads and buildings
share zoom 16 tiles for local models, with the same 256-tile cap and disk cache. Missing all-ocean
Terrain-RGB tiles are decoded as sea level; other provider failures remain errors.
The preview colours the existing surface and stays aligned when changing size.
With coverage enabled, STL export includes separate `woodland.stl` and `rock.stl` parts
where printable coverage exists, with complementary cutouts in `land.stl`. Import the STLs
as aligned parts of one object in the slicer and assign filament colours there;
STL itself does not store colours (the manifest includes the intended palette).

Woodland and rock follow the exact finished terrain surface with a fixed 0.6 mm vertical
material depth and at least 0.3 mm of underlying terrain where space is limited.
Coverage boundaries are simplified within 0.025 mm; strips below approximately
0.4 mm and tiny patches are removed for printing. The export may therefore omit
small regions shown in the map preview. Material partitioning runs only on export,
after physical scaling and exaggeration, and does not slow interactive edits.

Terrain stitching conservatively repairs long, isolated zero-height stripes on
internal tile edges when both neighbouring elevations support interpolation.
Original cached tiles remain untouched. Point-touching water regions receive
consistent surface levels and tiny finite connections to avoid discontinuities
and non-manifold solids. Concurrent tile-cache writes use separate temporary files.

Woodland follows the terrain surface without added canopy texture or raised relief.
Its colour remains editable, with a separate shallow material part in STL exports.

STL exports are validated after float32 serialization and reloading. Numerical contacts are separated with bounded sub-0.002 mm adjustments; open surfaces and repairs that change volume materially are rejected. Every exported part must have closed, consistently oriented topology with no collapsed or duplicate triangles. This validates the file itself rather than relying on the in-memory mesh alone.

Buildings retain mapped component heights and minimum heights, union shared walls and
trim route/water clearance instead of dropping whole intersecting buildings. Connected
terraces are filtered as blocks using a 0.25 mm width and 0.0625 mm² area threshold.
Ground elevation is sampled inside the footprint. Missing heights use a marked
6 m estimate internally; the minimum printed building height remains 0.4 mm. Roofs
use mapped shape tags where available, retaining flat roofs otherwise. Component extrusion uses
Manifold cross sections to handle dense footprints and courtyards robustly.

For local models (frame radius up to 2.5 km), a cached OpenStreetMap Overpass query
supplements Mapbox with roof tags on closed building ways. Gabled, hipped,
pyramidal, skillion, barrel, dome and conical roofs require a close unambiguous
footprint match and mapped roof height (or a supported mapped pitch). Gabled, hipped, pyramidal and
barrel roofs require near-rectangular footprints; explicitly directed single-slope
parts may use irregular footprints and taper to their base; domes and cones require near-elliptical
footprints. Curved caps use adaptive tessellation targeting approximately 0.01 mm
chord error, capped at 512 segments. Pitch-derived height is supported for gabled,
hipped, skillion and circular conical roofs; absent or invalid measurements stay flat. Unsupported/complex shapes,
relations, uncertain building heights and incomplete tags retain flat roofs.
Roof relief below 0.1 mm or spans below 0.4 mm are omitted. Source roof height is
included within total building height, and roof alignment survives clearance trimming.
The optional lookup has a short timeout and no retries; failures preserve buildings.
Successful source responses are cached across physical size edits. Roof data:
© OpenStreetMap contributors, https://www.openstreetmap.org/copyright.

Roads include local streets, service roads, tracks, pedestrian streets and paths.
Road-class width estimates are preserved for wider roads; narrower roads are widened
to a minimum 0.2 mm in both preview and export so the street network stays visible.
Widths are recalculated for model size changes. Roads bypass land-cover erosion and
outline simplification to preserve narrow bends and junctions. Mapped bridges are
supported surface crossings, including shallow inserts in water, and export with
the roads material. These are stylised crossings, not elevated structural bridge
models. Tunnels remain excluded.
