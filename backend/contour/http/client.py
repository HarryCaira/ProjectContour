"""Cached, retried HTTP client. Shared by terrain and biome data fetchers."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HttpClient:
    """Thin requests.Session wrapper with retry/backoff and 429 Retry-After handling.

    Stateless beyond the underlying session; safe to share across the pipeline.
    """

    def __init__(
        self,
        timeout: float = 15.0,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    ) -> None:
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=list(retry_statuses),
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.timeout = timeout

    def get(self, url: str, params: dict | None = None) -> bytes:
        """GET the URL with retries; raise for non-2xx responses; return raw bytes."""
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.content
