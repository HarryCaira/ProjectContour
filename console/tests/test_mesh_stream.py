"""Exercise the browser stream parser with split chunks and interrupted requests."""
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def console_dir() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("chunk_size", [1, 13, 4096])
def test__read_mesh_stream__handles_chunk_boundaries(console_dir: Path, chunk_size: int) -> None:
    script = """
import assert from 'node:assert/strict';
import { readMeshStream } from './lib/mesh-stream.ts';
const size = Number(process.argv[1]);
const events = [
  {type: 'progress', stage: 'terrain'},
  {type: 'heartbeat'},
  {type: 'progress', stage: 'preview'},
  {type: 'result', glb: btoa('glTF'), metadata: {parts: ['land'], triangles: [12]}},
];
const bytes = new TextEncoder().encode(events.map(e => JSON.stringify(e)).join('\\n'));
const stream = new ReadableStream({ start(controller) {
  for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
  controller.close();
}});
const stages = [];
const result = await readMeshStream(stream, p => stages.push(p.stage));
assert.deepEqual(stages, ['terrain', 'preview', 'display']);
assert.equal(new TextDecoder().decode(result.glb), 'glTF');
assert.deepEqual(result.metadata, {parts: ['land'], triangles: [12]});
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script, str(chunk_size)],
                   cwd=console_dir, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("mode", ["error", "truncated", "malformed", "disconnected"])
def test__read_mesh_stream__rejects_failures(console_dir: Path, mode: str) -> None:
    script = """
import assert from 'node:assert/strict';
import { readMeshStream } from './lib/mesh-stream.ts';
const mode = process.argv[1];
const stream = new ReadableStream({ start(controller) {
  if (mode === 'disconnected') { controller.error(new Error('Network disconnected')); return; }
  const text = mode === 'error' ? JSON.stringify({type:'error', message:'Build failed'}) :
    mode === 'malformed' ? 'not JSON' : JSON.stringify({type:'progress',stage:'terrain'});
  controller.enqueue(new TextEncoder().encode(text));
  controller.close();
}});
await assert.rejects(() => readMeshStream(stream, () => {}));
assert.equal(stream.locked, false);
"""
    subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script, mode],
                   cwd=console_dir, check=True, capture_output=True, text=True)
