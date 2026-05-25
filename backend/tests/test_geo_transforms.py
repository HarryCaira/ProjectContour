"""Tests for the LocalENU geodetic <-> ENU transform."""
from __future__ import annotations

import numpy as np
import pytest

from contour.geo.transforms import LocalENU


def test_origin_maps_to_zero():
    enu = LocalENU(lat0=51.5, lon0=-0.1, h0=10.0)
    result = enu.to_enu(51.5, -0.1, 10.0)
    np.testing.assert_allclose(result, [0, 0, 0], atol=1e-6)


def test_round_trip_geodetic_enu_geodetic():
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    lats = np.array([51.51, 51.52, 51.53])
    lons = np.array([-0.11, -0.12, -0.13])
    hs = np.array([100.0, 200.0, 300.0])
    coords = enu.to_enu(lats, lons, hs)
    back = enu.to_geodetic(coords[:, 0], coords[:, 1], coords[:, 2])
    np.testing.assert_allclose(back[:, 0], lats, atol=1e-9)
    np.testing.assert_allclose(back[:, 1], lons, atol=1e-9)
    np.testing.assert_allclose(back[:, 2], hs, atol=1e-3)


def test_to_enu_array_shape():
    enu = LocalENU(lat0=51.5, lon0=-0.1)
    lats = np.array([51.51, 51.52, 51.53])
    lons = np.array([-0.11, -0.12, -0.13])
    result = enu.to_enu(lats, lons, 0.0)
    assert result.shape == (3, 3)


def test_to_enu_scalar_shape():
    enu = LocalENU(lat0=0.0, lon0=0.0)
    result = enu.to_enu(0.0, 0.0, 0.0)
    assert result.shape == (3,)


def test_one_degree_north_is_about_111km():
    enu = LocalENU(lat0=0.0, lon0=0.0)
    coords = enu.to_enu(1.0, 0.0, 0.0)
    assert coords[1] == pytest.approx(110_574, abs=200)  # ~111 km, ellipsoidal
    assert abs(coords[0]) < 1.0  # essentially zero east


def test_one_degree_east_at_equator_is_about_111km():
    enu = LocalENU(lat0=0.0, lon0=0.0)
    coords = enu.to_enu(0.0, 1.0, 0.0)
    assert coords[0] == pytest.approx(111_319, abs=200)
    assert abs(coords[1]) < 1.0
