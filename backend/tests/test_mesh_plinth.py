"""Tests for the plinth mesh."""
from __future__ import annotations

import math

import pytest

from contour.framing.hex import HexFrame
from contour.mesh.plinth import build_plinth_mesh


def test_plinth_is_watertight():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=1000)
    mesh = build_plinth_mesh(frame, height_m=50.0, top_z=0.0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent


def test_plinth_volume_matches_formula():
    r = 1000.0
    h = 50.0
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=r)
    mesh = build_plinth_mesh(frame, height_m=h, top_z=0.0)
    expected_area = (3 * math.sqrt(3) / 2) * r**2
    assert mesh.volume == pytest.approx(expected_area * h, rel=1e-3)


def test_plinth_top_at_top_z():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=500)
    top_z = 12.5
    mesh = build_plinth_mesh(frame, height_m=10.0, top_z=top_z)
    assert mesh.bounds[1, 2] == pytest.approx(top_z)
    assert mesh.bounds[0, 2] == pytest.approx(top_z - 10.0)


def test_plinth_zero_height_raises():
    frame = HexFrame(centre_lon=0, centre_lat=0, circumradius_m=500)
    with pytest.raises(ValueError, match="positive"):
        build_plinth_mesh(frame, height_m=0.0, top_z=0.0)
