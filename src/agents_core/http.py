"""All outbound HTTP: retries, per-host rate limiting, request budgets, on-disk cache.

Re-running an agent during development reads from `.cache/http` instead of
re-downloading. Secrets passed as query params or headers are never written to the
cache and never logged. Only 2xx responses are ever cached, and a cache key records
*which* secrets were sent (never their values), so a response fetched without an
API key is never served once the key is added.

Daily request budgets (`HostPolicy.daily_budget`, e.g. SAM.gov's ~10/day) count
only requests actually sent — cache hits and connections that never opened are
free — and reset at midnight UTC. On a budgeted host retries are off unless the
policy sets `max_attempts`, so one flaky 5xx can't burn a scarce quota.

`Http.download` streams a large file to disk with a conditional GET
(ETag/Last-Modified), bypassing the JSON cache entirely.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlencode, urlsplit

import httpx

from agents_core import settings, tracing

log = logging.getLogger(__name__)

USER_AGENT = "agents-hub/0.1 (+https://github.com/Kghaffari26/agents-core)"

# Query params and headers treated as secrets: excluded from cache keys, cache files, logs.
SECRET_PARAMS = frozenset({"api_key", "apikey", "api-key", "key", "token", "access_token"})
SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "cookie"})

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

# Transport errors raised before any bytes of the request left this machine: these
# never count toward a host's daily budget.
_NOT_SENT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.UnsupportedProtocol,
)

R = TypeVar("R")


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RequestBudgetExceeded(HttpError):
    """A host's request budget (e.g. SAM.gov's ~10/day) would be exceeded."""


class _Retryable(Exception):
    def __init__(self, reason: str, delay: float | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.delay = delay


@dataclass(frozen=True)
class HostPolicy:
    """Per-host limits.

    `daily_budget` caps requests actually sent per UTC day; cache hits are free.
    `max_attempts` overrides `Http.max_attempts` for this host. When it's None, a host
    with a `daily_budget` gets a single attempt (no retries) and any other host gets
    `Http.max_attempts`. Set it explicitly to allow retries on a budgeted host.
    """

    min_interval_seconds: float = 0.0
    daily_budget: int | None = None
    max_attempts: int | None = None

    def attempts(self, default: int) -> int:
        if self.max_attempts is not None:
            return max(1, self.max_attempts)
        return 1 if self.daily_budget is not None else default


@dataclass(frozen=True)
class DownloadResult:
    """Result of `Http.download`. `modified` is False when the server answered 304 and
    `path` still holds the previous download.

    `headers` are this response's headers (lower-cased names; `set-cookie` dropped), on a
    304 too — e.g. rate-limit headers or `Link` pagination; `links` parses `Link` into
    `{rel: url}`."""

    path: Path
    modified: bool
    status: int
    etag: str | None
    last_modified: str | None
    bytes: int
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def links(self) -> dict[str, str]:
        return parse_link_header(self.headers.get("link", ""))


_LINK = re.compile(r"<([^>]*)>((?:\s*;\s*[^;,]+)*)")
_LINK_REL = re.compile(r"""\brel\s*=\s*"?([^";]+)"?""", re.IGNORECASE)


def parse_link_header(value: str) -> dict[str, str]:
    """`<https://...?page=2>; rel="next", <...>; rel="last"` -> {"next": ..., "last": ...}."""
    links: dict[str, str] = {}
    for m in _LINK.finditer(value or ""):
        rel = _LINK_REL.search(m.group(2))
        if rel:
            for name in rel.group(1).split():
                links.setdefault(name.lower(), m.group(1))
    return links


def _response_headers(r: httpx.Response) -> dict[str, str]:
    return {k.lower(): v for k, v in r.headers.items() if k.lower() != "set-cookie"}


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


def _cache_key(
    method: str,
    url: str,
    params: Mapping[str, Any] | None,
    body: Any,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Secret values are excluded, but which secrets were present is part of the key:
    a response fetched without a key must not be served after one is added."""
    parts = urlsplit(url)
    items = [*httpx.QueryParams(parts.query).multi_items(), *(params or {}).items()]
    query = sorted((k, str(v)) for k, v in items if k.lower() not in SECRET_PARAMS)
    secrets_present = sorted(
        {f"param:{k.lower()}" for k, v in items if k.lower() in SECRET_PARAMS and v}
        | {
            f"header:{k.lower()}"
            for k, v in (headers or {}).items()
            if k.lower() in SECRET_HEADERS and v
        }
    )
    material = json.dumps(
        [method.upper(), f"{parts.scheme}://{parts.netloc}{parts.path}", query, body],
        sort_keys=True,
        default=str,
    )
    if secrets_present:
        material += json.dumps(secrets_present)
    return hashlib.sha256(material.encode()).hexdigest()


def _is_json(response: Response) -> bool:
    try:
        response.json()
    except ValueError:  # json.JSONDecodeError and UnicodeDecodeError
        return False
    return True


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
        """GET and parse JSON. A body that isn't JSON (e.g. an HTML error page served
        with a 200) raises and is never cached."""
        return self.request("GET", url, cache_if=_is_json, **kwargs).json()

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
        cache_if: Callable[[Response], bool] | None = None,
    ) -> Response:
        """Send a request, serving from cache when a fresh entry exists.

        `ttl_seconds=0` disables caching for this call. Only 2xx responses are cached
        (anything else raises `HttpError`), and only if `cache_if(response)` is true
        when given; a cached entry failing `cache_if` is treated as a miss.
        """
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        key = _cache_key(method, url, params, json_body, headers)
        host = urlsplit(url).netloc.lower()
        safe_url = redact_url(url, params)

        with tracing.span(
            "http", f"{method.upper()} {host}", method=method.upper(), url=safe_url
        ) as sp:
            if ttl > 0 and not refresh:
                cached = self._cache_read(host, key, ttl)
                if cached is not None and (cache_if is None or cache_if(cached)):
                    log.debug("cache hit %s", safe_url)
                    sp.set(from_cache=True, status=cached.status, retries=0)
                    return cached

            sp.set(from_cache=False)
            response = self._send(method, url, host, safe_url, params, headers, json_body)
            sp.set(status=response.status, bytes=len(response.content))
            if (
                ttl > 0
                and 200 <= response.status < 300
                and (cache_if is None or cache_if(response))
            ):
                self._cache_write(host, key, response)
            return response

    def download(
        self,
        url: str,
        dest: Path | str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        force: bool = False,
        chunk_size: int = 1 << 20,
    ) -> DownloadResult:
        """Stream `url` to `dest` with a conditional GET, for files too large for the cache.

        The previous download's ETag/Last-Modified are kept in a `<dest>.meta.json`
        sidecar and sent as If-None-Match/If-Modified-Since; a 304 leaves `dest`
        untouched and returns `modified=False`. `force=True` skips the conditional
        headers. The body is written to a temp file and moved into place only once
        complete. Rate limits, budgets and retries apply as for `request`.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        meta_path = dest.with_name(dest.name + ".meta.json")
        host = urlsplit(url).netloc.lower()
        safe_url = redact_url(url, params)

        prior: dict[str, Any] = {}
        if not force and dest.is_file() and meta_path.is_file():
            try:
                prior = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                prior = {}
        send_headers = dict(headers or {})
        if prior.get("etag"):
            send_headers["If-None-Match"] = prior["etag"]
        if prior.get("last_modified"):
            send_headers["If-Modified-Since"] = prior["last_modified"]

        tmp = dest.with_name(dest.name + ".part")

        def attempt() -> DownloadResult:
            with self._client.stream("GET", url, params=params, headers=send_headers or None) as r:
                if r.status_code == 304 and prior:
                    return DownloadResult(
                        path=dest,
                        modified=False,
                        status=304,
                        etag=prior.get("etag"),
                        last_modified=prior.get("last_modified"),
                        bytes=dest.stat().st_size,
                        headers=_response_headers(r),
                    )
                self._raise_for_status(r, "GET", safe_url)
                size = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size):
                        f.write(chunk)
                        size += len(chunk)
                tmp.replace(dest)
                etag = r.headers.get("etag")
                last_modified = r.headers.get("last-modified")
                meta_path.write_text(
                    json.dumps({"etag": etag, "last_modified": last_modified, "url": safe_url})
                )
                return DownloadResult(
                    path=dest,
                    modified=True,
                    status=r.status_code,
                    etag=etag,
                    last_modified=last_modified,
                    bytes=size,
                    headers=_response_headers(r),
                )

        with tracing.span("http", f"GET {host}", method="GET", url=safe_url, download=True) as sp:
            try:
                result = self._with_retries("GET", host, safe_url, attempt)
            finally:
                tmp.unlink(missing_ok=True)
            sp.set(status=result.status, modified=result.modified, bytes=result.bytes)
        log.info(
            "GET %s -> %d (%s)",
            safe_url,
            result.status,
            f"{result.bytes} bytes" if result.modified else "not modified",
        )
        return result

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
        def attempt() -> Response:
            r = self._client.request(method, url, params=params, headers=headers, json=json_body)
            self._raise_for_status(r, method, safe_url)
            log.info("%s %s -> %d", method, safe_url, r.status_code)
            return Response(
                url=safe_url,
                status=r.status_code,
                content=r.content,
                headers={k.lower(): v for k, v in r.headers.items()},
                fetched_at=datetime.now(UTC),
            )

        return self._with_retries(method, host, safe_url, attempt)

    def _raise_for_status(self, r: httpx.Response, method: str, safe_url: str) -> None:
        if r.is_success:
            return
        if r.status_code in RETRY_STATUSES:
            raise _Retryable(f"HTTP {r.status_code}", self._retry_after(r))
        raise HttpError(f"{method} {safe_url} -> {r.status_code}", r.status_code)

    def _with_retries(
        self, method: str, host: str, safe_url: str, attempt_fn: Callable[[], R]
    ) -> R:
        """Run `attempt_fn` (one network request) under the host's policy: budget check,
        throttle, retry with backoff. Counts each request actually sent."""
        policy = self.policies.get(host, HostPolicy())
        max_attempts = policy.attempts(self.max_attempts)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            self._check_budget(host, policy)
            self._throttle(host, policy)
            tracing.current_span().set(retries=attempt - 1)
            try:
                result = attempt_fn()
            except _Retryable as e:
                self._count_request(host, policy)
                last_error = e.reason
                delay = e.delay or self._backoff(attempt)
            except HttpError:
                self._count_request(host, policy)
                raise
            except httpx.TransportError as e:
                if not isinstance(e, _NOT_SENT_ERRORS):
                    self._count_request(host, policy)
                last_error = type(e).__name__
                delay = self._backoff(attempt)
            else:
                self._count_request(host, policy)
                return result
            if attempt < max_attempts:
                log.warning(
                    "%s %s failed (%s); retry in %.1fs", method, safe_url, last_error, delay
                )
                self.sleep(delay)
        raise HttpError(
            f"{method} {safe_url} failed after {max_attempts} attempt"
            f"{'s' if max_attempts != 1 else ''}: {last_error}"
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

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def _budget_used(self, host: str) -> int:
        path = self._budget_path()
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return 0
        if data.get("date") != self._today():
            return 0
        return int(data.get("hosts", {}).get(host, 0))

    def _check_budget(self, host: str, policy: HostPolicy) -> None:
        if policy.daily_budget is None:
            return
        if self._budget_used(host) >= policy.daily_budget:
            raise RequestBudgetExceeded(
                f"{host}: daily request budget of {policy.daily_budget} used up"
            )

    def _count_request(self, host: str, policy: HostPolicy) -> None:
        self.network_requests += 1
        if policy.daily_budget is None:
            return
        path = self._budget_path()
        today = self._today()
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}
        if data.get("date") != today:
            data = {"date": today, "hosts": {}}
        data["hosts"][host] = data["hosts"].get(host, 0) + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def budget_remaining(self, host: str) -> int | None:
        policy = self.policies.get(host.lower())
        if policy is None or policy.daily_budget is None:
            return None
        return max(policy.daily_budget - self._budget_used(host.lower()), 0)

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
