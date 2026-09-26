"""Nested spans for one run, published as a redacted, size-capped `trace.json`.

    tracer = Tracer(agent="macro", run_id=run_id)
    with tracing.use(tracer), tracing.span("run", "macro"):
        with tracing.span("custom", "score listings", n=40) as s:
            ...
            s.set(kept=12)

The runner does this for every run, and `agents_core.llm` (`llm_call`, `guard`),
`agents_core.http` (`http`) and `agents_core.agent_loop` (`agent_loop`, `tool_call`)
open their spans automatically, so an agent gets a full trace without calling
anything. `tracing.span(...)` is a cheap no-op when no tracer is active (tests, a
library used outside the runner).

Recorded attrs, by kind: `llm_call` — tier, model, purpose, input/output/cache
tokens, usd, estimated_usd, stop_reason; `http` — method, url (secrets already
redacted), status, retries, from_cache; `tool_call` — tool, input, output preview,
is_error, pending_approval; `guard` — outcome, attempts, unsupported;
`agent_loop` — steps, tool_calls, stop_reason, usd. Every span has a latency
(`duration_ms`), a status and, on failure, the exception.

`write_trace` redacts before writing: values under secret-looking keys (api_key,
token, authorization, password, cookie, ...), well-known credential formats
(Anthropic/GitHub/AWS/Slack keys, bearer tokens, JWTs, secret query params) and the
literal values of secret-looking environment variables all become `***`. It then
caps the file at `max_bytes` by shortening long strings, then dropping the latest
spans (`truncated`/`dropped_spans` say so). The summary is computed before any
truncation, so it's always exact.

Spans cross thread boundaries only through `contextvars`: code that starts threads
should run their work in `contextvars.copy_context()` (as `LLM` and `AgentLoop` do).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents_core import settings
from agents_core.publish import write_json
from agents_core.schema import SpanKind, Trace, TraceSpan, TraceSummary

TRACE_FILE = "trace.json"
DEFAULT_MAX_BYTES = 256_000
REDACTED = "***"

_current_tracer: ContextVar[Tracer | None] = ContextVar("agents_core_tracer", default=None)
_current_span: ContextVar[Span | None] = ContextVar("agents_core_span", default=None)


def _iso_ms(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class Span:
    """A timed operation. Use `set(**attrs)` to record values and `add(key, n)` to
    accumulate counters; both are safe to call after the span has ended."""

    def __init__(
        self,
        tracer: Tracer | None,
        span_id: str,
        parent_id: str | None,
        kind: SpanKind,
        name: str,
        attrs: dict[str, Any],
    ) -> None:
        self.tracer = tracer
        self.id = span_id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.attrs = dict(attrs)
        self.started_wall = time.time()
        self._t0 = tracer.clock() if tracer else 0.0
        self.duration_ms: float | None = None
        self.status = "ok"
        self.error: str | None = None

    def set(self, **attrs: Any) -> Span:
        self.attrs.update(attrs)
        return self

    def add(self, key: str, amount: float = 1) -> Span:
        self.attrs[key] = self.attrs.get(key, 0) + amount
        return self

    def fail(self, error: BaseException | str) -> None:
        self.status = "error"
        self.error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"

    def end(self) -> None:
        if self.duration_ms is None and self.tracer is not None:
            self.duration_ms = round((self.tracer.clock() - self._t0) * 1000, 3)

    def elapsed_ms(self) -> float:
        if self.duration_ms is not None:
            return self.duration_ms
        if self.tracer is None:
            return 0.0
        return round((self.tracer.clock() - self._t0) * 1000, 3)

    def to_model(self) -> TraceSpan:
        return TraceSpan(
            id=self.id,
            parent_id=self.parent_id,
            kind=self.kind,
            name=self.name,
            started_at=_iso_ms(self.started_wall),
            duration_ms=self.duration_ms,
            status=self.status,  # type: ignore[arg-type]
            error=self.error,
            attrs=_jsonable(self.attrs),
        )


class _NullSpan(Span):
    """What `span()` yields when no tracer is active: records nothing."""

    def __init__(self) -> None:
        super().__init__(None, "", None, "custom", "", {})

    def set(self, **attrs: Any) -> Span:
        return self

    def add(self, key: str, amount: float = 1) -> Span:
        return self


NULL_SPAN = _NullSpan()


class Tracer:
    """Collects one run's spans. Thread-safe."""

    def __init__(
        self, agent: str = "", run_id: str = "", *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.agent = agent
        self.run_id = run_id
        self.clock = clock
        self.spans: list[Span] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @contextmanager
    def span(self, kind: SpanKind, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current_span.get()
        parent_id = parent.id if parent is not None and parent.tracer is self else None
        with self._lock:
            s = Span(self, f"s{next(self._ids)}", parent_id, kind, name, attrs)
            self.spans.append(s)
        token = _current_span.set(s)
        try:
            yield s
        except BaseException as e:
            s.fail(e)
            raise
        finally:
            s.end()
            _current_span.reset(token)

    def summary(self) -> TraceSummary:
        with self._lock:
            spans = list(self.spans)
        roots = [s for s in spans if s.parent_id is None]
        return TraceSummary(
            steps=int(sum(s.attrs.get("steps", 0) for s in spans if s.kind == "agent_loop")),
            tool_calls=sum(1 for s in spans if s.kind == "tool_call"),
            llm_calls=sum(1 for s in spans if s.kind == "llm_call"),
            total_latency_ms=round(sum(s.elapsed_ms() for s in roots), 3),
            cost_usd=round(
                sum(float(s.attrs.get("usd", 0)) for s in spans if s.kind == "llm_call"), 6
            ),
            guard_retries=int(
                sum(max(int(s.attrs.get("attempts", 1)) - 1, 0) for s in spans if s.kind == "guard")
            ),
        )

    def to_trace(self) -> Trace:
        with self._lock:
            spans = [s.to_model() for s in self.spans]
        return Trace(agent=self.agent, run_id=self.run_id, summary=self.summary(), spans=spans)


# ---- module-level API used by the instrumented modules ---------------------------


def current_tracer() -> Tracer | None:
    return _current_tracer.get()


def current_span() -> Span:
    """The innermost open span, or a no-op span."""
    s = _current_span.get()
    return s if s is not None else NULL_SPAN


@contextmanager
def use(tracer: Tracer) -> Iterator[Tracer]:
    """Make `tracer` the active tracer for this context (and contexts copied from it)."""
    token = _current_tracer.set(tracer)
    span_token = _current_span.set(None)
    try:
        yield tracer
    finally:
        _current_span.reset(span_token)
        _current_tracer.reset(token)


@contextmanager
def span(kind: SpanKind, name: str, **attrs: Any) -> Iterator[Span]:
    """A span on the active tracer, or a no-op when there is none."""
    tracer = _current_tracer.get()
    if tracer is None:
        yield NULL_SPAN
        return
    with tracer.span(kind, name, **attrs) as s:
        yield s


# ---- redaction ------------------------------------------------------------------

# A key is secret when it *ends* in one of these words, so `input_tokens` and
# `max_tokens` are kept while `api_key`, `x-api-key` and `github_token` are not.
_SECRET_KEY = re.compile(
    r"(?:^|[_\-\s.])(?:api[_\-]?key|apikey|key|token|access[_\-]?token|secret|client[_\-]?secret"
    r"|password|passwd|pwd|authorization|auth|cookie|set[_\-]?cookie|credentials?"
    r"|private[_\-]?key|session[_\-]?id|signature)$|^(?:key|token|secret|password|auth)$",
    re.IGNORECASE,
)
_SECRET_VALUES = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(bearer|token|basic)\s+[A-Za-z0-9._~+/\-]{12,}=*"),
]
_SECRET_QUERY = re.compile(
    r"(?i)([?&](?:api[_\-]?key|apikey|key|token|access_token|secret|password)=)[^&\s\"'#]+"
)
_SECRET_ENV_NAME = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$")


def _env_secrets() -> list[str]:
    values = {
        v
        for k, v in os.environ.items()
        if _SECRET_ENV_NAME.search(k) and v and len(v) >= 8 and not v.isdigit()
    }
    return sorted(values, key=len, reverse=True)


def redact_text(text: str, env_secrets: list[str] | None = None) -> str:
    for secret in _env_secrets() if env_secrets is None else env_secrets:
        text = text.replace(secret, REDACTED)
    for pattern in _SECRET_VALUES:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return _SECRET_QUERY.sub(lambda m: m.group(1) + REDACTED, text)


def redact(obj: Any, env_secrets: list[str] | None = None) -> Any:
    """A copy of a JSON-like value with secrets replaced by `***`."""
    secrets_ = _env_secrets() if env_secrets is None else env_secrets
    if isinstance(obj, dict):
        return {
            k: (
                REDACTED
                if isinstance(k, str)
                and _SECRET_KEY.search(k)
                and v not in (None, "")
                and not isinstance(v, bool | int | float)
                else redact(v, secrets_)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list | tuple):
        return [redact(v, secrets_) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, secrets_)
    return obj


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


# ---- size cap -------------------------------------------------------------------


def _truncate_strings(obj: Any, limit: int) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"…[{len(obj) - limit} chars cut]"
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_strings(v, limit) for v in obj]
    return obj


def _size(data: dict[str, Any]) -> int:
    return len(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode())


def fit_trace(data: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Shrink a trace dict to at most `max_bytes` of compact JSON: shorten long
    strings in span attrs/errors first, then drop the latest spans."""
    if _size(data) <= max_bytes:
        return data
    out = dict(data)
    for limit in (2000, 500, 120):
        out["spans"] = [_truncate_strings(s, limit) for s in data["spans"]]
        out["truncated"] = True
        if _size(out) <= max_bytes:
            return out
    spans = out["spans"]
    lo, hi = 0, len(spans)
    while lo < hi:  # the largest prefix of spans that fits
        mid = (lo + hi + 1) // 2
        trial = {**out, "spans": spans[:mid], "dropped_spans": len(spans) - mid}
        if _size(trial) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return {**out, "spans": spans[:lo], "dropped_spans": len(spans) - lo}


def max_trace_bytes() -> int:
    value = os.environ.get("AGENTS_CORE_TRACE_MAX_BYTES")
    return int(value) if value else DEFAULT_MAX_BYTES


def trace_dict(tracer: Tracer, *, max_bytes: int | None = None) -> dict[str, Any]:
    """The redacted, size-capped trace, validated against `schema.Trace`."""
    data = tracer.to_trace().model_dump(mode="json")
    data = redact(data)
    data = fit_trace(data, max_trace_bytes() if max_bytes is None else max_bytes)
    return Trace.model_validate(data).model_dump(mode="json")


def write_trace(
    tracer: Tracer, publish_dir: Path | str | None = None, *, max_bytes: int | None = None
) -> Path:
    """Write `<publish_dir>/trace.json`. Returns its path."""
    base = Path(publish_dir) if publish_dir is not None else settings.publish_dir()
    path = base / TRACE_FILE
    write_json(path, trace_dict(tracer, max_bytes=max_bytes))
    return path
