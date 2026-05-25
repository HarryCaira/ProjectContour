"""Cached, retried HTTP client. Shared by terrain and biome data fetchers."""
from __future__ import annotations


class HttpClient:
    def __init__(self) -> None:
        raise NotImplementedError
