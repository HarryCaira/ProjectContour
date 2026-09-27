"""Route detail selected by physical error rather than a fixed point count."""
from __future__ import annotations

import numpy as np


def resample_route(points: np.ndarray, spacing: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    if counts.sum() > 200_000:
        raise ValueError("Route exceeds the production mesh budget")
    pieces = [a + np.arange(n)[:, None] / n * (b - a)
              for a, b, n in zip(points[:-1], points[1:], counts) if np.any(a != b)]
    return np.concatenate([*pieces, points[-1:]])


def simplify_route(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices retained by 3D Douglas–Peucker, using segment distances."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        direction = points[last] - points[first]
        delta = points[first + 1:last] - points[first]
        squared_length = np.dot(direction, direction)
        fraction = np.clip(delta @ direction / squared_length, 0, 1) if squared_length else np.zeros(len(delta))
        distances = np.linalg.norm(delta - fraction[:, None] * direction, axis=1)
        offset = int(np.argmax(distances))
        if distances[offset] > tolerance:
            index = first + 1 + offset
            keep[index] = True
            stack.extend([(first, index), (index, last)])
    return np.flatnonzero(keep)
