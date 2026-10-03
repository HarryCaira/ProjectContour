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
Water, land cover and infrastructure use at most 256 vector tiles per area, reducing
zoom for regional routes while retaining zoom 14 for local models. Missing all-ocean
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

Woodland relief uses overlapping, irregular oval canopy crowns with varied
sizes, orientation and heights, clustered spacing and gaps. At 100 mm across
5 km, crown heights are 0.30–0.70 mm. Crown spacing, radius and height scale
with millimetres per geographic metre, bounded to 0.5–2 times that reference. Crowns
follow the terrain and keep clear of woodland boundaries and the route. Export
contains real raised geometry; the fast preview uses a shading approximation.

Woodland crowns shrink and taper down near coverage boundaries and route edges.
The viewer uses a distance-based edge fade instead of an abrupt canopy mask;
printed material boundaries remain discrete between filament colours.

STL exports are validated after float32 serialization and reloading. Numerical contacts are separated with bounded sub-0.002 mm adjustments; open surfaces and repairs that change volume materially are rejected. Every exported part must have closed, consistently oriented topology with no collapsed or duplicate triangles. This validates the file itself rather than relying on the in-memory mesh alone.
