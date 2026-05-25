"""Tests for the HTTP client: success path, error handling, retry behaviour."""
from __future__ import annotations

import pytest
import requests
import responses

from contour.http.client import HttpClient


@responses.activate
def test_get_returns_response_bytes():
    responses.add(responses.GET, "https://example.com/data", body=b"hello world", status=200)
    client = HttpClient()
    assert client.get("https://example.com/data") == b"hello world"


@responses.activate
def test_get_passes_query_params():
    responses.add(responses.GET, "https://example.com/data", body=b"ok", status=200)
    client = HttpClient()
    client.get("https://example.com/data", params={"access_token": "secret"})
    assert "access_token=secret" in responses.calls[0].request.url


@responses.activate
def test_get_retries_on_500_then_succeeds():
    responses.add(responses.GET, "https://example.com/data", status=500)
    responses.add(responses.GET, "https://example.com/data", status=500)
    responses.add(responses.GET, "https://example.com/data", body=b"recovered", status=200)
    client = HttpClient(max_retries=5, backoff_factor=0.0)
    assert client.get("https://example.com/data") == b"recovered"
    assert len(responses.calls) == 3


@responses.activate
def test_get_retries_on_429_then_succeeds():
    responses.add(responses.GET, "https://example.com/data", status=429)
    responses.add(responses.GET, "https://example.com/data", body=b"ok", status=200)
    client = HttpClient(max_retries=3, backoff_factor=0.0)
    assert client.get("https://example.com/data") == b"ok"


@responses.activate
def test_get_raises_after_exhausted_retries():
    for _ in range(10):
        responses.add(responses.GET, "https://example.com/data", status=500)
    client = HttpClient(max_retries=2, backoff_factor=0.0)
    with pytest.raises(requests.HTTPError):
        client.get("https://example.com/data")
    # Total: initial + 2 retries = 3 calls
    assert len(responses.calls) == 3


@responses.activate
def test_get_raises_on_404():
    responses.add(responses.GET, "https://example.com/missing", status=404)
    client = HttpClient(max_retries=2, backoff_factor=0.0)
    with pytest.raises(requests.HTTPError):
        client.get("https://example.com/missing")
