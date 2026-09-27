"""The scale reference has a known physical size and sits on the same ground."""
from pathlib import Path
import subprocess


def test__create_banana_geometry__has_calibrated_length_and_finite_vertices() -> None:
    script = """
import assert from 'node:assert/strict';
import { createBananaGeometry, BANANA_LENGTH_MM, VIEW_UNITS_PER_MM } from './lib/banana.ts';
const geometry = createBananaGeometry();
const bounds = geometry.boundingBox;
assert.ok(Math.abs(bounds.max.y - bounds.min.y - 180) < 1e-4);
assert.ok(Math.abs(bounds.min.z) < 1e-4);
assert.ok(bounds.max.z > 20 && bounds.max.z < 40);
assert.ok(Array.from(geometry.attributes.position.array).every(Number.isFinite));
assert.ok(Array.from(geometry.attributes.normal.array).every(Number.isFinite));
for (const sizeMm of [50, 150, 300]) {
  const modelWidth = 2 * sizeMm / 150;
  const bananaLength = BANANA_LENGTH_MM * VIEW_UNITS_PER_MM;
  assert.ok(Math.abs(modelWidth / bananaLength - sizeMm / BANANA_LENGTH_MM) < 1e-8);
}
geometry.dispose();
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                   cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
