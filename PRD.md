# ProjectContour — PRD

## 1. Overview
ProjectContour is a web application that turns a recorded GPS route into a beautiful custom 3D-printed object — a hexagonal landscape model with the route raised across it, designed by the user in real time, ordered from the site, and shipped to their door.

## 2. Problem Statement
People accumulate GPS recordings of meaningful trips — hikes, rides, runs, climbs — and want a physical memento that captures both the route and the landscape it crossed. Existing options on Etsy and similar marketplaces solve the manufacturing side, but the design experience is poor: users send a GPX over email or fill in a clunky form and trust the seller to produce something tasteful. There is no product where the *design moment itself* — seeing your trip turn into a physical object you'd be proud to own, and shaping it in real time — is the experience.

## 3. Goals
- Make the design moment the product. Upload a GPX, immediately see a beautiful real-time 3D model, and shape it to taste.
- Lower the barrier to ownership. Upload to "looks great" in minutes, with sensible defaults at every step.
- Produce an object the user is proud to display, every time. Quality is non-negotiable; defaults must already look good.
- Build an extensible foundation that supports more styles, biomes, frames, and input flows over time without architectural rewrites.

## 4. Non-Goals (initial release)
- Checkout, payment, and order fulfilment.
- A developer API.
- Multiple visual styles, frames, or plinth options.
- Strava / Komoot / AllTrails / Garmin Connect integrations.
- Mobile-optimised editing.
- Slicing, printer tuning, or print-side concerns.

## 5. Target Users
Initial users are **outdoor enthusiasts** — hikers, cyclists, runners, climbers — who have completed a meaningful trip and want a physical keepsake. They are not GIS or 3D experts and should not need to be. They are comfortable using a polished modern web app.

Secondary users, in later phases: gift-makers, and 3D-printing hobbyists who would consume an API.

## 6. The Object (v1)
The v1 object is a **hexagonal landscape tile** mounted on a polished plinth, with:
- A **lowered water surface** (lakes, rivers, coastlines) clearly distinguished from the land.
- The **route itself** raised as a ribbon across the terrain in a contrasting accent colour.
- A **plinth** beneath the tile, in a single default style.

The object is produced as a **mesh kit** — multiple labelled parts (`land`, `water`, `route`, `plinth`) — designed to be printed in different colours and assembled, or printed together on multi-material hardware.

The signature visual treatment is **monochrome biome**: each biome gets one muted colour, the route is the single accent, the plinth is neutral. Minimalist, reliably printable, aligned with the Apple-meets-nature aesthetic.

## 7. The Editor (v1)
A single-page web app:
1. **Upload** a GPS track.
2. **See** the model immediately in a large, well-lit 3D viewer — orbit, pan, zoom.
3. **Adjust** the model in real time:
   - **Physical size** (mm) — longest dimension.
   - **Print resolution** (mm) — controls terrain detail.
   - **Vertical exaggeration** — multiplier on terrain height for dramatic effect.
4. **Export** the mesh kit. (Becomes "Order" in a later phase.)

All edits feel real-time. Scale and exaggeration are pure client-side transforms; resolution may require a server round-trip or pre-computed LODs.

The editor is **the product**. Its polish, performance, and aesthetic are the primary thing being built.

## 8. Aesthetic & Design Language
- **Apple minimalism, nature-inspired.** Generous whitespace, clean sans-serif typography, the 3D viewer as the hero, controls as quiet adjacent affordances.
- **Muted, natural palette.** Earth tones, a single accent colour for the route, neutral plinth.
- **Tactile rendering.** The viewer should feel like a real object — soft shadows, subtle surface variation, matte finishes — not glossy plastic.
- **Calm, deliberate motion.** Edits propagate smoothly; no jank, no flicker, no spinners during interaction.

## 9. Functional Requirements

### 9.1 Input
- Accept a standard GPS track file via upload.
- Validate and surface clear errors for unsupported variants.
- Handle real-world variations gracefully: multiple tracks or segments, missing elevation, very short or very long tracks.

### 9.2 Terrain & Biome Acquisition
- Fetch global elevation data covering the route's hexagonal framing.
- Fetch biome data sufficient to identify **land and water** for v1.
- Automatically choose a level of detail appropriate to the chosen physical size and print resolution.
- Cache fetched data so re-runs of the same area are near-instant.

### 9.3 Mesh Generation
- Produce a **kit of labelled meshes**: `land`, `water`, `route`, `plinth`. Each is a watertight, printable solid.
- Water sits at a lower height than land.
- The route is a raised ribbon following the GPS path, sampled to the terrain surface.
- All meshes are spatially coregistered so the assembly is correct.

### 9.4 Viewer
- Real-time 3D rendering of the mesh kit in the browser.
- Orbit, pan, zoom.
- Materials and lighting tuned to the v1 aesthetic.
- Edits propagate with no perceptible lag.

### 9.5 Output
- Export the mesh kit as a set of standard 3D-printing-format files.
- Surface basic stats (dimensions, part counts).

## 10. Architectural Commitments
Three commitments enable evolution without rewrites:

**1. The output is a labelled mesh kit, not a mesh.** Every component is a separate mesh tagged with its material/colour. New biomes, new accent layers, new plinth components are added by appending tagged meshes — no surgery in upstream stages.

**2. Style is a strategy on a neutral scene.** The pipeline produces a neutral intermediate (heightmap, biome polygons, route, frame) and a *style* turns it into the mesh kit. v1 has one style ("monochrome biome"). New styles plug in without affecting the data layer.

**3. The model is fully described by a versioned, serialisable settings object.** The editor manipulates it; future orders persist it; the renderer consumes it; URLs share it. New features are optional fields, not migrations. A year-old order can be re-rendered on improved infrastructure.

These are seams, not subsystems. We build single implementations behind each seam; we do not build plugin frameworks, registries, or strategy interfaces in v1.

## 11. Non-Functional Requirements
- **Performance.** Upload to first render: target under 15 seconds on a typical route. Editor interactions: 60fps where possible, no perceptible lag on slider adjustments.
- **Reliability.** A successful upload must always produce a printable mesh kit. Failures are loud and explanatory.
- **Reproducibility.** The same settings object always produces equivalent output.
- **Cost discipline.** Data fetching is bounded and cached; quotas keep usage predictable as we scale.
- **Browser support.** Modern Chromium / Safari / Firefox on desktop. Mobile is a tolerated viewing experience but not optimised for editing.

## 12. Out of Scope for v1
- Checkout, payment, fulfilment.
- Developer API.
- Forest, snow, urban biomes.
- Building extrusion.
- Visual styles other than monochrome biome.
- Plinth options or engraved text.
- Frame shapes other than hexagonal.
- Strava / Komoot / etc. integrations.
- Accounts, saved projects, sharing.

## 13. Future Roadmap (directional)
- **Phase 2 — Commerce.** Checkout, fulfilment partner, accounts, order tracking, saved projects.
- **Phase 3 — Customisation depth.** Additional plinth styles, engraving (route name, distance, date), additional biomes (forest, snow), buildings on urban routes.
- **Phase 4 — Visual styles.** Topographic-stack and realistic styles as alternatives to monochrome biome.
- **Phase 5 — Distribution.** Strava / Komoot / AllTrails / Garmin Connect integrations to capture intent at the source.
- **Phase 6 — Developer API.** Headless access for hobbyists and integrators.

## 14. Success Metrics
- **Time to first delight.** Upload → beautiful, customisable model in the viewer.
- **Edit engagement.** % of sessions that touch at least one editor control.
- **Export rate.** % of upload sessions that result in an exported model.
- **Aesthetic confidence.** Qualitative: do users describe the default output as "beautiful" without prompting?
