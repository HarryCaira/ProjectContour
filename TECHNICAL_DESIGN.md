# ProjectContour — Technical Design (v1)

Operational technical design for v1, described by `PRD.md`. Captures the decisions made during design discussion: stack, data flow, module structure, key algorithms, and the seams that allow future extension. v1 is the editor experience; checkout, fulfilment, and the developer API are out of scope.

## 1. System Shape

Three components:

- **Frontend** — Next.js + React + React Three Fiber + Tailwind, served from `/frontend`. Owns the editor UI, the 3D viewer, real-time transforms, and export.
- **Backend** — FastAPI on top of a Python meshing pipeline, served from `/backend`. Owns upload handling, data fetching, mesh generation, and asset serving.
- **Mapbox** — DEM (Terrain-RGB tiles) and biome polygons (vector tiles, `mapbox-streets-v8`).

```
Browser ── HTTP ──> FastAPI ── HTTP ──> Mapbox
   │                  │
   │                  ├── tile cache (disk)
   │                  └── kit cache (disk)
   │
   └── 3D viewer (R3F) + editor (Zustand)
```

## 2. Data Flow

### Upload (server-driven, one-shot)

1. User uploads a GPX file.
2. Backend parses, normalises, validates, stores the file and returns `{ id, sha256, stats }`.
3. Client builds a default `Settings` object referencing the upload id.
4. Client POSTs `Settings` to `/mesh`.
5. Backend executes the pipeline → returns a glTF mesh kit + metadata.
6. Frontend renders the kit in the viewer.

### Edits

- **Pure transforms (scale, vertical exaggeration)**: applied entirely in the frontend on the existing meshes. No server roundtrip. Real-time.
- **Topology-changing edits (resolution, frame, rotation, future biome toggles)**: client POSTs updated `Settings` to `/mesh`, backend regenerates, frontend swaps the kit. The kit cache (keyed on `(gpx_sha256, settings_hash)`) makes repeated edits over a settings space near-instant.

## 3. The Pipeline

Stages, each with typed input/output. Each stage lives in its own module under `backend/contour/`.

### 3.1 Input — `contour/input/`
GPX → normalised `Route` (lat/lon/elevation arrays + metadata). Handles multi-track files (first non-empty track), missing elevation (fall back to DEM lookup), and validates non-empty.

### 3.2 Framing — `contour/framing/`
`Route` + `Settings.framing` → `HexFrame` (centre lat/lon, circumradius in metres, orientation). The hex is the smallest regular **pointy-top** hexagon containing the route bbox expanded by `paddingRatio`, oriented by `rotationDegrees` around its centre.

### 3.3 Coordinate frame — `contour/geo/`
A `LocalENU` transform anchored on the hex centre is established once per request and reused by every stage that needs to convert lat/lon to local coordinates. Routes, heightmap pixel positions, and biome polygons all share this frame.

### 3.4 Terrain data — `contour/data/terrain.py`
`HexFrame` + `Settings.physical.resolutionMm` + `Settings.physical.sizeMm` → `Heightmap` (numpy array of elevations in metres + tile/ENU metadata). Picks the smallest tile zoom whose pixel resolution meets the requested print resolution (subject to a tile budget). Fetches Mapbox Terrain-RGB tiles, decodes, stitches.

### 3.5 Biome data — `contour/data/biomes.py`
`HexFrame` → `BiomePolygons` (currently `{water: [shapely.Polygon, ...]}`) in ENU coordinates. Fetches vector tiles from `mapbox-streets-v8`, extracts the `water` layer, reprojects, unions across tile boundaries, and clips to the hex.

### 3.6 Meshing — `contour/mesh/`

Produces an unstyled **neutral scene**:

- `land`: 2D polygon `hex \ water` plus heightmap → triangulated 3D mesh via constrained Delaunay triangulation. Watertight; vertical walls along the hex boundary and along water edges down to the water surface; flat base below the minimum elevation.
- `water`: 2D polygon `water ∩ hex` plus a fixed lowered Z → closed solid (flat top, vertical sides, flat bottom).
- `route`: GPX line projected to ENU, sampled onto the land surface, extruded as a ribbon.
- `plinth`: hex prism beneath the model in the default style.

All meshes share the same ENU coordinate frame and assemble correctly.

### 3.7 Style — `contour/styles/`

`base.py` defines a `Style` interface. v1 has one implementation:

- `monochrome_biome.py` takes the neutral scene and returns a `MeshKit`:

```
MeshKit
  parts: list[KitPart]

KitPart
  name: "land" | "water" | "route" | "plinth"
  mesh: trimesh.Trimesh
  material: { colour, roughness, ... }
  exportable_as_separate_part: bool
```

For monochrome biome, materials are a small palette of muted earth tones plus a contrasting accent for the route. Geometry is unchanged from neutral; only materials are applied.

This is the seam where future styles (topographic stack, realistic) will plug in — they receive the same neutral scene and produce a different `MeshKit`, possibly with different geometry decisions.

### 3.8 Export — `contour/export/`

- `gltf.py`: serialise a `MeshKit` to `.glb` for the frontend.
- `stl.py`: serialise a `MeshKit` as a zip of per-part `.stl` files at user export time.

## 4. Settings Schema

The source of truth for a model. Validated on both ends — Pydantic in Python, Zod in TypeScript.

```ts
{
  schemaVersion: 1,
  source: { type: "gpx", id: string, sha256: string },
  framing: { shape: "hex", paddingRatio: number, rotationDegrees: number },
  physical: { sizeMm: number, resolutionMm: number },
  style: { name: "monochrome-biome" },
  terrain: { verticalExaggeration: number },
  biomes: { water: { enabled: boolean, depthFraction: number } },
  route: { enabled: boolean, widthMm: number, heightAboveTerrainMm: number },
  plinth: { enabled: boolean, style: "default" },
}
```

- All dimensions in millimetres; inches are a UI presentation concern only.
- Defaults applied server-side; clients may omit optional fields.
- `schemaVersion` exists for future migrations; v1 only handles version 1.
- Biomes are keyed sub-objects, not a list, so new biomes (`forest`, `snow`, `urban`) slot in as optional additions.

## 5. API Surface

```
POST /upload
  multipart: file=<gpx bytes>
  → { id: string, sha256: string, stats: { distanceKm, points } }

POST /mesh
  json: <Settings>
  → application/octet-stream: glTF .glb
     + response headers with kit metadata (parts, triangle counts, bounds)

POST /export
  json: <Settings>
  → application/zip: { land.stl, water.stl, route.stl, plinth.stl, manifest.json }
```

Synchronous calls. `/mesh` and `/export` may take 5–20s at high resolution; the UI shows a generative state and updates the viewer when ready. Async job pattern is a v1.5 upgrade if the latency proves unacceptable.

Errors are structured: `{ code, message, details? }` with an appropriate HTTP status.

## 6. Caching

- **Tile cache**: `<cache>/tiles/<provider>/<layer>/<z>/<x>/<y>.<ext>`. Persists indefinitely.
- **Kit cache**: keyed on `(gpx_sha256, settings_hash)`. Stores the produced glTF and STL kit. Persists indefinitely; LRU later if needed.
- **Settings hash**: SHA-256 of the canonical JSON serialisation of the `Settings` object, with `source.sha256` retained inside the hashed payload.

## 7. Coordinate Frames

- **Geodetic** (lat/lon/elevation): the universal frame; everything enters and leaves here.
- **Tile** (z, x, y, pixel): Mapbox tile-space; used only inside the data layer.
- **Local ENU** (E, N, U in metres, anchored on the hex centre): the working frame for all geometry. Established once per request and shared by every downstream stage.

## 8. Frontend

- **Next.js** (app router). The editor is a single client-side page; server routes proxy the backend when needed.
- **React Three Fiber + drei** for the 3D viewer. The mesh kit loads as glTF; each part is a `<primitive>` with its material applied.
- **Zustand** for the editor settings state. Single store; `useSettings()` everywhere.
- **TanStack Query** for `/upload`, `/mesh`, `/export` calls.
- **Tailwind** for styling, with a small design-system layer for the earth palette and typography.
- **URL state**: settings serialised to a compact base64-encoded form in the URL hash so sessions are shareable.

### Real-time transforms

- **Scale**: a single `THREE.Object3D.scale` on the kit root. Free.
- **Vertical exaggeration**: a uniform on a shader that multiplies vertex Z. The water depth and route base scale proportionally so coupling stays visually correct.

Both are bound to Zustand state; sliders update the store; R3F's reactivity propagates.

## 9. The Hex Frame

- **Shape**: pointy-top regular hexagon (flat-bottom horizontal edges).
- **Sizing**: smallest hexagon centred on the route's centroid whose bounding circle contains the route bbox expanded by `paddingRatio`.
- **Rotation**: `rotationDegrees ∈ [0°, 60°)` rotates the hex around its centre.
- **Clipping**: every 2D polygon (land, water) is clipped against the hex; meshes are built within the clipped region with vertical walls along the hex boundary.

## 10. Water Rendering

Approach **(A): carved land + recessed water solid**.

- The land polygon is `hex \ water`. Land mesh has holes where water sits, with vertical side walls along the water boundary down to the water surface.
- The water mesh is a closed solid (flat top at lowered Z, vertical sides matching the water polygon, flat bottom at the model's base).
- **Depth**: `depthFraction × model height`. 5–8% by default.
- **Exaggeration coupling**: water depth scales with `verticalExaggeration` so the water step remains visually proportionate.
- **Polygon source for v1**: lakes, oceans, reservoirs (polygonal water). Rivers (linestrings) are deferred — they need buffering to become printable polygons and are noisy at small scales.

## 11. Reproducibility & Versioning

A `(schemaVersion, Settings, gpx_sha256, mapboxDataVersion)` tuple uniquely determines the output. We persist the GPX bytes for every stored upload so any past `Settings` can be re-rendered on demand. Mapbox data drifts over time — accepted for v1 personal use; flagged as a Commerce-phase concern.

## 12. Module Structure

```
backend/
  contour/
    schema/          # Pydantic models
      settings.py
      kit.py
    http/            # cached, retried HTTP client
      client.py
      cache.py
    geo/             # coordinate math
      transforms.py
      tiles.py
    input/
      gpx.py
    framing/
      hex.py
    data/
      terrain.py
      biomes.py
    mesh/
      terrain.py
      water.py
      route.py
      plinth.py
      hex_clip.py
    styles/
      base.py
      monochrome_biome.py
    export/
      gltf.py
      stl.py
    pipeline.py
    api/
      server.py
      routes.py
      errors.py
  pyproject.toml
  tests/

frontend/
  app/                      # Next.js routes
  components/
    editor/                 # sliders, controls
    viewer/                 # R3F scene
  lib/
    schema/                 # Zod settings schema
    api/                    # TanStack Query bindings
    state/                  # Zustand stores
    transforms/             # client-side mesh transforms
  package.json

reference/                  # existing Python prototype, kept temporarily for context
```

## 13. Things Deliberately Not Built (v1)

- Plugin / registry / DI system.
- Multiple style implementations.
- Multiple DEM providers.
- Configurable biome list (hardcoded `[land, water]`).
- Style hot-swapping in the editor.
- Accounts, auth, sessions.
- Checkout, payment, fulfilment.
- Async job system / websockets.
- Mobile-optimised editor.

## 14. Known Risks

- **Constrained Delaunay performance.** Clipping the heightmap against complex water polygons and re-meshing with CDT is the most computationally interesting step. Profile early on a realistic input.
- **Coastal routes.** Ocean polygons are conceptually enormous; clipping them to the hex must be robust. Plan: rely on shapely's clipping, validate on a coastal test route.
- **Mapbox vector tile licensing.** Confirm before launch that our usage falls within the developer plan; consider Protomaps for the Commerce phase if not.
- **High-resolution viewer performance.** Hundreds of thousands of triangles in the browser is fine on desktop but may need decimation for the preview on weaker machines.
