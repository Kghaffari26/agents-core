"""All outbound HTTP: retries, per-host rate limiting, request budgets, on-disk cache.

Re-running an agent during development reads from `.cache/http` instead of
re-downloading. Secrets passed as query params or headers are never written to the
cache and never logged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from agents_core import settings

log = logging.getLogger(__name__)

USER_AGENT = "agents-hub/0.1 (+https://github.com/Kghaffari26/agents-core)"

# Query params and headers treated as secrets: excluded from cache keys, cache files, logs.
SECRET_PARAMS = frozenset({"api_key", "apikey", "api-key", "key", "token", "access_token"})
SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "cookie"})

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RequestBudgetExceeded(HttpError):
    """A host's request budget (e.g. SAM.gov's ~10/day) would be exceeded."""


@dataclass(frozen=True)
class HostPolicy:
    """Per-host limits. `daily_budget` counts network requests only; cache hits are free."""

    min_interval_seconds: float = 0.0
    daily_budget: int | None = None


@dataclass
class Response:
    url: str
    status: int
    content: bytes
    headers: dict[str, str]
    fetched_at: datetime
    from_cache: bool = False

    @property
    def text(self) -> str:
        return self.content.decode(self._charset(), errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    def _charset(self) -> str:
        ctype = self.headers.get("content-type", "")
        for part in ctype.split(";"):
            k, _, v = part.strip().partition("=")
            if k.lower() == "charset" and v:
                return v.strip('"')
        return "utf-8"


def redact_url(url: str, params: Mapping[str, Any] | None = None) -> str:
    """URL with any secret query params replaced by `***`, safe for logs and citations."""
    parts = urlsplit(url)
    query = [
        (k, "***" if k.lower() in SECRET_PARAMS else v)
        for k, v in httpx.QueryParams(parts.query).multi_items()
    ]
    query += [
        (k, "***" if k.lower() in SECRET_PARAMS else str(v)) for k, v in (params or {}).items()
    ]
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return f"{base}?{urlencode(query, safe='*')}" if query else base


def _cache_key(method: str, url: str, params: Mapping[str, Any] | None, body: Any) -> str:
    parts = urlsplit(url)
    query = sorted(
        (k, str(v))
        for k, v in [*httpx.QueryParams(parts.query).multi_items(), *(params or {}).items()]
        if k.lower() not in SECRET_PARAMS
    )
    material = json.dumps(
        [method.upper(), f"{parts.scheme}://{parts.netloc}{parts.path}", query, body],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode()).hexdigest()


@dataclass
class Http:
    """Shared HTTP client. Create one per run and close it (or use as a context manager)."""

    cache_dir: Path = field(default_factory=settings.http_cache_dir)
    default_ttl_seconds: int = field(default_factory=settings.http_cache_ttl_seconds)
    timeout_seconds: float = 30.0
    max_attempts: int = 4
    backoff_seconds: float = 1.0
    policies: dict[str, HostPolicy] = field(default_factory=dict)
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            transport=self.transport,
        )
        self._last_request: dict[str, float] = {}
        self.network_requests = 0

    def __enter__(self) -> Http:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def set_policy(self, host: str, policy: HostPolicy) -> None:
        self.policies[host.lower()] = policy

    # ---- public API -------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        ttl_seconds: int | None = None,
        refresh: bool = False,
    ) -> Response:
        return self.request(
            "GET", url, params=params, headers=headers, ttl_seconds=ttl_seconds, refresh=refresh
        )

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        ttl_seconds: int | None = None,
        refresh: bool = False,
    ) -> Response:
        """Send a request, serving from cache when a fresh entry exists.

        `ttl_seconds=0` disables caching for this call. Only 2xx responses are cached.
        """
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        key = _cache_key(method, url, params, json_body)
        host = urlsplit(url).netloc.lower()
        safe_url = redact_url(url, params)

        if ttl > 0 and not refresh:
            cached = self._cache_read(host, key, ttl)
            if cached is not None:
                log.debug("cache hit %s", safe_url)
                return cached

        response = self._send(method, url, host, safe_url, params, headers, json_body)
        if ttl > 0:
            self._cache_write(host, key, response)
        return response

    # ---- network ----------------------------------------------------------

    def _send(
        self,
        method: str,
        url: str,
        host: str,
        safe_url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        json_body: Any,
    ) -> Response:
        policy = self.policies.get(host, HostPolicy())
        last_error: str = ""
        for attempt in range(1, self.max_attempts + 1):
            self._consume_budget(host, policy)
            self._throttle(host, policy)
            try:
                r = self._client.request(
                    method, url, params=params, headers=headers, json=json_body
                )
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}"
                delay = self._backoff(attempt)
            else:
                self.network_requests += 1
                if r.is_success:
                    log.info("%s %s -> %d", method, safe_url, r.status_code)
                    return Response(
                        url=safe_url,
                        status=r.status_code,
                        content=r.content,
                        headers={k.lower(): v for k, v in r.headers.items()},
                        fetched_at=datetime.now(UTC),
                    )
                if r.status_code not in RETRY_STATUSES:
                    raise HttpError(f"{method} {safe_url} -> {r.status_code}", r.status_code)
                last_error = f"HTTP {r.status_code}"
                delay = self._retry_after(r) or self._backoff(attempt)
            if attempt < self.max_attempts:
                log.warning(
                    "%s %s failed (%s); retry in %.1fs", method, safe_url, last_error, delay
                )
                self.sleep(delay)
        raise HttpError(
            f"{method} {safe_url} failed after {self.max_attempts} attempts: {last_error}"
        )

    def _backoff(self, attempt: int) -> float:
        return self.backoff_seconds * 2 ** (attempt - 1)

    @staticmethod
    def _retry_after(r: httpx.Response) -> float | None:
        value = r.headers.get("retry-after")
        if value is None:
            return None
        try:
            return min(float(value), 120.0)
        except ValueError:
            return None

    def _throttle(self, host: str, policy: HostPolicy) -> None:
        if policy.min_interval_seconds <= 0:
            return
        last = self._last_request.get(host)
        now = self.clock()
        if last is not None:
            wait = policy.min_interval_seconds - (now - last)
            if wait > 0:
                self.sleep(wait)
                now = self.clock()
        self._last_request[host] = now

    # ---- request budget ---------------------------------------------------

    def _budget_path(self) -> Path:
        return self.cache_dir / "_budget.json"

    def _consume_budget(self, host: str, policy: HostPolicy) -> None:
        if policy.daily_budget is None:
            return
        path = self._budget_path()
        today = date.today().isoformat()
        data: dict[str, Any] = {}
        if path.is_file():
            data = json.loads(path.read_text())
        if data.get("date") != today:
            data = {"date": today, "hosts": {}}
        used = data["hosts"].get(host, 0)
        if used >= policy.daily_budget:
            raise RequestBudgetExceeded(
                f"{host}: daily request budget of {policy.daily_budget} used up"
            )
        data["hosts"][host] = used + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def budget_remaining(self, host: str) -> int | None:
        policy = self.policies.get(host.lower())
        if policy is None or policy.daily_budget is None:
            return None
        path = self._budget_path()
        if not path.is_file():
            return policy.daily_budget
        data = json.loads(path.read_text())
        if data.get("date") != date.today().isoformat():
            return policy.daily_budget
        return max(policy.daily_budget - data["hosts"].get(host.lower(), 0), 0)

    # ---- cache ------------------------------------------------------------

    def _cache_path(self, host: str, key: str) -> Path:
        safe_host = host.replace(":", "_") or "_"
        return self.cache_dir / safe_host / f"{key}.json"

    def _cache_read(self, host: str, key: str, ttl: int) -> Response | None:
        path = self._cache_path(host, key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text())
            fetched_at = datetime.fromisoformat(entry["fetched_at"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        if (datetime.now(UTC) - fetched_at).total_seconds() > ttl:
            return None
        return Response(
            url=entry["url"],
            status=entry["status"],
            content=base64.b64decode(entry["content_b64"]),
            headers=entry["headers"],
            fetched_at=fetched_at,
            from_cache=True,
        )

    def _cache_write(self, host: str, key: str, response: Response) -> None:
        path = self._cache_path(host, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            k: v
            for k, v in response.headers.items()
            if k in {"content-type", "etag", "last-modified"} and k not in SECRET_HEADERS
        }
        entry = {
            "url": response.url,
            "status": response.status,
            "headers": headers,
            "fetched_at": response.fetched_at.isoformat(),
            "content_b64": base64.b64encode(response.content).decode(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry))
        tmp.replace(path)
