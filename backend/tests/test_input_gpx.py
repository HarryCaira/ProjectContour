"""Tests for GPX parsing."""
from __future__ import annotations

import numpy as np
import pytest

from contour.input.gpx import parse_gpx

SIMPLE_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk>
    <name>Test track</name>
    <trkseg>
      <trkpt lat="51.5" lon="-0.1"><ele>10.0</ele></trkpt>
      <trkpt lat="51.51" lon="-0.11"><ele>12.0</ele></trkpt>
      <trkpt lat="51.52" lon="-0.12"><ele>15.0</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

MULTI_TRACK_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk><name>Empty</name><trkseg></trkseg></trk>
  <trk>
    <name>Real</name>
    <trkseg>
      <trkpt lat="51.5" lon="-0.1"><ele>10.0</ele></trkpt>
      <trkpt lat="51.51" lon="-0.11"><ele>12.0</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

NO_ELEVATION_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
    <trkpt lat="51.5" lon="-0.1"/>
    <trkpt lat="51.51" lon="-0.11"/>
  </trkseg></trk>
</gpx>
"""

MIXED_ELEVATION_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
    <trkpt lat="51.5" lon="-0.1"/>
    <trkpt lat="51.51" lon="-0.11"><ele>20.0</ele></trkpt>
    <trkpt lat="51.52" lon="-0.12"/>
    <trkpt lat="51.53" lon="-0.13"><ele>40.0</ele></trkpt>
  </trkseg></trk>
</gpx>
"""

EMPTY_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test"></gpx>
"""

ROUTE_FALLBACK_GPX = b"""<?xml version="1.0"?>
<gpx version="1.1" creator="test">
  <rte>
    <name>A route</name>
    <rtept lat="51.5" lon="-0.1"><ele>5.0</ele></rtept>
    <rtept lat="51.51" lon="-0.11"><ele>6.0</ele></rtept>
  </rte>
</gpx>
"""


def test_parse_simple_gpx():
    route = parse_gpx(SIMPLE_GPX)
    assert route.num_points == 3
    assert route.latitudes[0] == pytest.approx(51.5)
    assert route.longitudes[0] == pytest.approx(-0.1)
    assert route.elevations[0] == pytest.approx(10.0)
    assert route.name == "Test track"


def test_parse_multi_track_picks_first_non_empty():
    route = parse_gpx(MULTI_TRACK_GPX)
    assert route.num_points == 2
    assert route.name == "Real"


def test_parse_no_elevation_returns_zeros():
    route = parse_gpx(NO_ELEVATION_GPX)
    assert route.num_points == 2
    np.testing.assert_allclose(route.elevations, [0.0, 0.0])


def test_parse_mixed_elevation_fills_neighbours():
    route = parse_gpx(MIXED_ELEVATION_GPX)
    assert route.num_points == 4
    # Index 0 is back-filled from index 1 (= 20.0)
    # Index 2 is forward-filled from index 1 (= 20.0)
    np.testing.assert_allclose(route.elevations, [20.0, 20.0, 20.0, 40.0])


def test_parse_empty_gpx_raises():
    with pytest.raises(ValueError, match="no track or route points"):
        parse_gpx(EMPTY_GPX)


def test_parse_invalid_xml_raises():
    with pytest.raises(ValueError, match="could not be parsed"):
        parse_gpx(b"not xml at all <<< invalid")


def test_parse_route_fallback():
    route = parse_gpx(ROUTE_FALLBACK_GPX)
    assert route.num_points == 2
    assert route.name == "A route"


def test_route_bbox_and_centroid():
    route = parse_gpx(SIMPLE_GPX)
    west, south, east, north = route.bbox
    assert west == pytest.approx(-0.12)
    assert east == pytest.approx(-0.1)
    assert south == pytest.approx(51.5)
    assert north == pytest.approx(51.52)
    lon, lat = route.centroid
    assert lon == pytest.approx(-0.11)
    assert lat == pytest.approx(51.51)


def test_route_distance_km_positive():
    route = parse_gpx(SIMPLE_GPX)
    assert route.distance_km() > 0
