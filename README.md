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

TODO:
- The GPX reader does not handle multiple routes/tracks well. This means a lot of more complicated GPX files aren't read well.
