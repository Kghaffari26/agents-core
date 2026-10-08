"""The only module that imports the Anthropic SDK.

Agents ask for a tier ("fast" or "smart"), never a model ID. Every call goes through
the run's CostTracker, which logs tokens/USD and enforces MAX_RUN_USD before and after
each call. Prompts are laid out for caching: stable `system` and `context` blocks first
(cache breakpoint on the last of them), per-run data in the user message after it.

Numbers come from data, never from the model: pass computed figures in the prompt and
ask for narrative only. Pass `guard=` (see core/guards.py) to check the narrative: a
failing output is retried once with the unsupported numbers named, then replaced by
`fallback()`. Guarded calls return `Guarded(value, narrative_source, ...)`.

Sampling: a tier may set `temperature` in models.toml, and every call takes a
`temperature=` override; it works for every call type (synchronous calls send it
in the request body via `extra_body`, batches in each request's params). Leave both
unset for models that reject sampling parameters. Synchronous calls made together
(`run_many`, `guard_batch` retries, a batch's `on_timeout="sync"` fallback) run on
up to `max_concurrency` threads (`[llm] max_concurrency` in models.toml, the
`LLM(max_concurrency=)` argument, or a per-call override); the default of 1 keeps
them sequential.

Multi-turn tool use: `converse()` sends one request of a conversation you manage
(messages + tool definitions) and returns a `Turn` with plain-dict content blocks;
`agents_core.agent_loop` is built on it. Every call opens an `llm_call` span (and
every guarded call a `guard` span) on the active tracer — see `agents_core.tracing`.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, overload

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from agents_core import settings, tracing
from agents_core.costs import CostTracker, SpendScope, Tier, Usage, usd_for
from agents_core.guards import RETRY_INSTRUCTION as RETRY_INSTRUCTION  # re-export (v0.3.1 API)
from agents_core.guards import GuardResult, retry_instruction
from agents_core.schema import NarrativeSource, iso_z

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V")
X = TypeVar("X")

# Rough chars-per-token used only for the pre-call budget estimate.
_CHARS_PER_TOKEN = 3.5
_SDK_MAX_RETRIES = 4


# Request parameters the installed SDK's `messages.create`/`messages.parse` don't take
# as keyword arguments (anthropic 1.8 has no `temperature=`: passing it raised
# TypeError before v0.3.1). They're sent in `extra_body`, which the SDK merges into
# the JSON request body unchanged, so the API sees the same request either way.
# Batch requests are plain JSON already and carry them as-is.
_BODY_ONLY_PARAMS = ("temperature",)


def _sdk_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """`params` as keyword arguments for `client.messages.create`/`parse`."""
    moved = {k: params[k] for k in _BODY_ONLY_PARAMS if k in params}
    if not moved:
        return params
    kwargs = {k: v for k, v in params.items() if k not in moved}
    kwargs["extra_body"] = {**(params.get("extra_body") or {}), **moved}
    return kwargs


class LLMError(RuntimeError):
    """The model returned something unusable (refusal, truncation, schema mismatch)."""


class GuardFailed(LLMError):
    """The number guard failed twice and no fallback was given."""


class BatchTimeout(LLMError):
    """A batch didn't end within `timeout_seconds` (it has been cancelled)."""


@dataclass(frozen=True)
class Guarded[R]:
    """Result of a guarded call. `narrative_source` is "template" when the fallback ran.

    `unsupported` lists the numbers that failed the last guard check (empty when the
    first attempt passed).
    """

    value: R
    narrative_source: NarrativeSource
    attempts: int
    unsupported: list[str] = field(default_factory=list)


def _plain(obj: Any) -> Any:
    """SDK objects (pydantic models) or test doubles -> plain JSON-like values."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            k: _plain(v) for k, v in vars(obj).items() if not k.startswith("_") and v is not None
        }
    return obj


@dataclass(frozen=True)
class Turn:
    """One `converse()` response. `content` holds the response's content blocks as
    plain dicts — append `{"role": "assistant", "content": turn.content}` to the
    conversation unchanged (thinking blocks included) before the next request."""

    content: list[dict[str, Any]]
    stop_reason: str | None
    usage: Usage
    usd: float
    model: str
    refusal_category: str | None = None

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


def _usage_attrs(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
    }


def _with_message_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of `messages` with a cache breakpoint on the last block, so each turn of
    a growing conversation reads the previous turns from the prompt cache."""
    if not messages:
        return messages
    last = dict(messages[-1])
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return messages
    # Thinking blocks and empty text blocks can't carry a breakpoint.
    eligible = [
        i
        for i, b in enumerate(content)
        if isinstance(b, dict)
        and b.get("type") not in ("thinking", "redacted_thinking")
        and not (b.get("type") == "text" and not b.get("text"))
    ]
    if not eligible:
        return messages
    blocks = [dict(b) if isinstance(b, dict) else b for b in content]
    blocks[eligible[-1]]["cache_control"] = {"type": "ephemeral"}
    last["content"] = blocks
    return [*messages[:-1], last]


@dataclass(frozen=True)
class TierConfig:
    model: str
    max_tokens: int
    effort: str | None = None
    thinking: str | None = None  # "adaptive" | "disabled" | None (model default)
    temperature: float | None = None  # None: not sent (the model's default)


def tier_config(tier: Tier) -> TierConfig:
    cfg = settings.load_config("models")["tiers"][tier]
    temperature = cfg.get("temperature")
    return TierConfig(
        model=cfg["model"],
        max_tokens=int(cfg["max_tokens"]),
        effort=cfg.get("effort"),
        thinking=cfg.get("thinking"),
        temperature=None if temperature is None else float(temperature),
    )


def default_max_concurrency() -> int:
    """`[llm] max_concurrency` from models.toml (default 1: sequential)."""
    value = settings.load_config("models").get("llm", {}).get("max_concurrency", 1)
    return max(1, int(value))


@dataclass(frozen=True)
class BatchItem:
    custom_id: str
    prompt: str
    max_tokens: int | None = None


@dataclass(frozen=True)
class BatchResult[M: BaseModel]:
    custom_id: str
    value: M | str | None = None
    error: str | None = None
    # "sync" when produced by run_many (including a batch's on_timeout="sync" fallback).
    via: Literal["batch", "sync"] = "batch"

    @property
    def ok(self) -> bool:
        return self.error is None


class LLM:
    def __init__(
        self,
        tracker: CostTracker,
        *,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        max_concurrency: int | None = None,
    ) -> None:
        self.tracker = tracker
        self._client = client
        self._sleep = sleep
        self.max_concurrency = max_concurrency
        self._log_lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key(), max_retries=_SDK_MAX_RETRIES
            )
        return self._client

    # ---- request building -------------------------------------------------

    @staticmethod
    def _system_blocks(system: str, context: str | None) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if context:
            blocks.append({"type": "text", "text": context})
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def _params(
        self,
        cfg: TierConfig,
        *,
        system: str,
        prompt: str,
        context: str | None,
        max_tokens: int | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": max_tokens or cfg.max_tokens,
            "system": self._system_blocks(system, context),
            "messages": [{"role": "user", "content": prompt}],
        }
        if cfg.effort:
            params["output_config"] = {"effort": cfg.effort}
        if cfg.thinking:
            params["thinking"] = {"type": cfg.thinking}
        temp = cfg.temperature if temperature is None else temperature
        if temp is not None:
            params["temperature"] = temp
        return params

    @staticmethod
    def _estimate_usd(params: dict[str, Any], *, batch: bool = False) -> float:
        """Worst-case cost: all input uncached, output runs to max_tokens."""
        chars = (
            len(json.dumps(params["system"]))
            + len(json.dumps(params["messages"]))
            + len(json.dumps(params.get("tools", [])))
        )
        worst = Usage(
            input_tokens=int(chars / _CHARS_PER_TOKEN), output_tokens=params["max_tokens"]
        )
        return usd_for(params["model"], worst, batch=batch)

    @staticmethod
    def _check_stop(message: Any, purpose: str) -> None:
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMError(f"{purpose}: model refused ({getattr(details, 'category', None)})")
        if message.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: output truncated at max_tokens; raise it for this call")

    @staticmethod
    def _text(message: Any) -> str:
        return "".join(b.text for b in message.content if b.type == "text").strip()

    # ---- single calls -----------------------------------------------------

    def _send(
        self, tier: Tier, params: dict[str, Any], output_model: type[T] | None, purpose: str
    ) -> Any:
        """One budget-checked, cost-logged call. Returns text, or the parsed model."""
        with tracing.span(
            "llm_call", purpose or tier, tier=tier, model=params["model"], purpose=purpose
        ) as sp:
            estimate = self._estimate_usd(params)
            sp.set(estimated_usd=round(estimate, 6))
            with self.tracker.reserve(estimate):
                if output_model is None:
                    message = self.client.messages.create(**_sdk_kwargs(params))
                else:
                    message = self.client.messages.parse(
                        output_format=output_model, **_sdk_kwargs(params)
                    )
                usage = Usage.from_api(message.usage)
                sp.set(
                    **_usage_attrs(usage),
                    usd=round(usd_for(params["model"], usage), 6),
                    stop_reason=message.stop_reason,
                )
                self.tracker.record(tier=tier, model=params["model"], usage=usage, purpose=purpose)
            self._check_stop(message, purpose)
        if output_model is None:
            return self._text(message)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"{purpose}: no parsed output returned")
        return parsed

    def estimate_usd(self, params: dict[str, Any]) -> float:
        """Worst-case USD for a request built by this class (all input uncached,
        output runs to max_tokens)."""
        return self._estimate_usd(params)

    def converse(
        self,
        tier: Tier,
        messages: list[dict[str, Any]],
        *,
        system: str,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        budget: SpendScope | None = None,
    ) -> Turn:
        """One request in a multi-turn conversation you manage (e.g. a tool-use loop).

        `tools` are Messages API tool definitions (`name`, `description`,
        `input_schema`). Cache breakpoints go on the system/context block, the last
        tool, and the last message. Before sending, the worst-case cost is checked
        against `budget` (a `SpendScope`, raising `ScopeBudgetExceeded`) and reserved
        against MAX_RUN_USD (raising `BudgetExceeded`); either way nothing is sent.
        Stop reasons are returned, not raised — the caller decides what a refusal or
        truncation means.
        """
        cfg = tier_config(tier)
        params = self._params(
            cfg,
            system=system,
            prompt="",
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        params["messages"] = _with_message_breakpoint(messages)
        if tools:
            tool_defs = [dict(t) for t in tools]
            tool_defs[-1]["cache_control"] = {"type": "ephemeral"}
            params["tools"] = tool_defs
        with tracing.span(
            "llm_call", purpose or tier, tier=tier, model=cfg.model, purpose=purpose
        ) as sp:
            estimate = self._estimate_usd(params)
            sp.set(estimated_usd=round(estimate, 6))
            if budget is not None:
                budget.check(estimate)
            with self.tracker.reserve(estimate):
                message = self.client.messages.create(**_sdk_kwargs(params))
                usage = Usage.from_api(message.usage)
                usd = usd_for(cfg.model, usage)
                sp.set(**_usage_attrs(usage), usd=round(usd, 6), stop_reason=message.stop_reason)
                self.tracker.record(tier=tier, model=cfg.model, usage=usage, purpose=purpose)
        details = getattr(message, "stop_details", None)
        return Turn(
            content=[_plain(b) for b in message.content],
            stop_reason=message.stop_reason,
            usage=usage,
            usd=usd,
            model=cfg.model,
            refusal_category=getattr(details, "category", None) if details else None,
        )

    @overload
    def complete(
        self,
        tier: Tier,
        prompt: str,
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: None = None,
        fallback: None = None,
    ) -> str: ...

    @overload
    def complete(
        self,
        tier: Tier,
        prompt: str,
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: Callable[[str], GuardResult],
        fallback: Callable[[], str] | None = None,
    ) -> Guarded[str]: ...

    def complete(
        self,
        tier: Tier,
        prompt: str,
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: Callable[[str], GuardResult] | None = None,
        fallback: Callable[[], str] | None = None,
    ) -> str | Guarded[str]:
        """Return plain text, or `Guarded[str]` when a guard is given.

        Use `structured` when the output feeds a schema.
        """
        cfg = tier_config(tier)
        params = self._params(
            cfg,
            system=system,
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = self._send(tier, params, None, purpose)
        if guard is None:
            return text
        return self._guarded(tier, params, text, guard, fallback, None, purpose)

    @overload
    def structured(
        self,
        tier: Tier,
        prompt: str,
        output_model: type[T],
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: None = None,
        fallback: None = None,
    ) -> T: ...

    @overload
    def structured(
        self,
        tier: Tier,
        prompt: str,
        output_model: type[T],
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: Callable[[T], GuardResult],
        fallback: Callable[[], T] | None = None,
    ) -> Guarded[T]: ...

    def structured(
        self,
        tier: Tier,
        prompt: str,
        output_model: type[T],
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
        temperature: float | None = None,
        guard: Callable[[T], GuardResult] | None = None,
        fallback: Callable[[], T] | None = None,
    ) -> T | Guarded[T]:
        """Return a validated `output_model` instance using structured outputs, or
        `Guarded[output_model]` when a guard is given. Build the guard with
        `agents_core.guards.fields_guard(facts, [...narrative fields...])`.
        """
        cfg = tier_config(tier)
        params = self._params(
            cfg,
            system=system,
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parsed = self._send(tier, params, output_model, purpose)
        if guard is None:
            return parsed
        return self._guarded(tier, params, parsed, guard, fallback, output_model, purpose)

    # ---- number guard -----------------------------------------------------

    def _guarded(
        self,
        tier: Tier,
        params: dict[str, Any],
        first: V,
        guard: Callable[[V], GuardResult],
        fallback: Callable[[], V] | None,
        output_model: type[T] | None,
        purpose: str,
    ) -> Guarded[V]:
        """Check `first`; on failure retry once in the same conversation, then fall back."""
        with tracing.span("guard", purpose or tier, purpose=purpose) as sp:
            guarded = self._guarded_inner(
                tier, params, first, guard, fallback, output_model, purpose
            )
            sp.set(
                attempts=guarded.attempts,
                outcome=(
                    "fallback"
                    if guarded.narrative_source == "template"
                    else ("pass" if guarded.attempts == 1 else "pass_after_retry")
                ),
                unsupported=guarded.unsupported,
            )
            return guarded

    def _guarded_inner(
        self,
        tier: Tier,
        params: dict[str, Any],
        first: V,
        guard: Callable[[V], GuardResult],
        fallback: Callable[[], V] | None,
        output_model: type[T] | None,
        purpose: str,
    ) -> Guarded[V]:
        result = guard(first)
        if result.ok:
            return Guarded(first, "llm", attempts=1)
        self._log_guard_failure(params, purpose, 1, first, result.unsupported)

        retry_params = {
            **params,
            "messages": [
                *params["messages"],
                {"role": "assistant", "content": self._as_text(first)},
                {
                    "role": "user",
                    "content": retry_instruction(result),
                },
            ],
        }
        try:
            second = self._send(tier, retry_params, output_model, f"{purpose}:guard-retry")
        except LLMError as e:
            # A refusal or truncation on the retry is treated as a second guard failure.
            self._log_guard_failure(params, purpose, 2, None, result.unsupported, error=str(e))
            unsupported = result.unsupported
        else:
            result = guard(second)
            if result.ok:
                return Guarded(second, "llm", attempts=2)
            self._log_guard_failure(params, purpose, 2, second, result.unsupported)
            unsupported = result.unsupported

        if fallback is None:
            tracing.current_span().set(attempts=2, outcome="failed", unsupported=unsupported)
            raise GuardFailed(f"{purpose}: unsupported numbers after retry: {unsupported}")
        log.warning("%s: number guard failed twice; using template fallback", purpose)
        return Guarded(fallback(), "template", attempts=2, unsupported=unsupported)

    @staticmethod
    def _as_text(value: Any) -> str:
        return value.model_dump_json() if isinstance(value, BaseModel) else str(value)

    @staticmethod
    def _prompt_hash(params: dict[str, Any]) -> str:
        material = json.dumps([params["system"], params["messages"][0]], sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def _log_guard_failure(
        self,
        params: dict[str, Any],
        purpose: str,
        attempt: int,
        output: Any,
        unsupported: list[str],
        *,
        error: str | None = None,
    ) -> None:
        entry = {
            "ts": iso_z(datetime.now(UTC)),
            "run_id": self.tracker.run_id,
            "agent": self.tracker.agent,
            "purpose": purpose,
            "attempt": attempt,
            "prompt_hash": self._prompt_hash(params),
            "output": None if output is None else self._as_text(output),
            "unsupported": unsupported,
        }
        if error:
            entry["error"] = error
        path = settings.guard_failures_path()
        with self._log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        log.warning("%s: number guard attempt %d failed: %s", purpose, attempt, unsupported)

    # ---- concurrency ------------------------------------------------------

    def _concurrency(self, override: int | None) -> int:
        if override is not None:
            return max(1, override)
        if self.max_concurrency is not None:
            return max(1, self.max_concurrency)
        return default_max_concurrency()

    def _map(
        self, fn: Callable[[X], V], items: Sequence[X], max_concurrency: int | None
    ) -> list[V]:
        """`[fn(i) for i in items]`, on up to `max_concurrency` threads, in order.
        The first exception (e.g. BudgetExceeded) cancels what hasn't started and
        propagates."""
        workers = min(self._concurrency(max_concurrency), len(items))
        if workers <= 1:
            return [fn(i) for i in items]
        _ = self.client  # create the SDK client once, before any thread needs it
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agents-core-llm")
        try:
            # Each item runs in a copy of this context, so its spans nest under ours.
            futures = [pool.submit(contextvars.copy_context().run, fn, i) for i in items]
            return [f.result() for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    # ---- many synchronous calls -------------------------------------------

    def run_many(
        self,
        tier: Tier,
        items: Sequence[BatchItem],
        *,
        system: str,
        context: str | None = None,
        output_model: type[T] | None = None,
        purpose: str = "",
        temperature: float | None = None,
        max_concurrency: int | None = None,
    ) -> dict[str, BatchResult[T]]:
        """Run independent prompts as synchronous calls (full price), up to
        `max_concurrency` at a time. Same inputs and result shape as `batch()` (results
        carry `via="sync"`), so it's a drop-in fallback when a batch times out.

        An unusable output (refusal, truncation, missing parse) becomes that item's
        `error`; BudgetExceeded and API errors fail the whole call.
        """
        if not items:
            return {}
        ids = [i.custom_id for i in items]
        if len(set(ids)) != len(ids):
            raise ValueError("custom_ids must be unique")
        cfg = tier_config(tier)

        def one(item: BatchItem) -> BatchResult[T]:
            params = self._params(
                cfg,
                system=system,
                prompt=item.prompt,
                context=context,
                max_tokens=item.max_tokens,
                temperature=temperature,
            )
            try:
                value = self._send(tier, params, output_model, f"{purpose}:{item.custom_id}")
            except LLMError as e:
                return BatchResult(item.custom_id, error=str(e), via="sync")
            return BatchResult(item.custom_id, value=value, via="sync")

        results = self._map(one, list(items), max_concurrency)
        return {r.custom_id: r for r in results}

    # ---- batch ------------------------------------------------------------

    def batch(
        self,
        tier: Tier,
        items: Sequence[BatchItem],
        *,
        system: str,
        context: str | None = None,
        output_model: type[T] | None = None,
        purpose: str = "",
        poll_seconds: float = 30.0,
        timeout_seconds: float = 3 * 3600,
        temperature: float | None = None,
        on_timeout: Literal["raise", "sync"] = "raise",
        max_concurrency: int | None = None,
    ) -> dict[str, BatchResult[T]]:
        """Run many independent prompts through the Batch API (50% price, async).

        Blocks until the batch ends. Returns results keyed by custom_id; failed items
        carry `error` instead of `value`. The whole batch's worst-case cost is checked
        against the budget before submitting.

        If the batch hasn't ended after `timeout_seconds` it's cancelled; then
        `on_timeout="raise"` (the default) raises `BatchTimeout`, and `"sync"` reruns
        every item through `run_many` (full price, `max_concurrency` at a time) and
        returns those results, marked `via="sync"`.
        """
        if not items:
            return {}
        ids = [i.custom_id for i in items]
        if len(set(ids)) != len(ids):
            raise ValueError("batch custom_ids must be unique")
        with tracing.span(
            "llm_call", purpose or f"{tier} batch", tier=tier, purpose=purpose, batch=True
        ) as sp:
            sp.set(items=len(items))
            return self._batch(
                tier,
                items,
                ids,
                system=system,
                context=context,
                output_model=output_model,
                purpose=purpose,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                on_timeout=on_timeout,
                max_concurrency=max_concurrency,
            )

    def _batch(
        self,
        tier: Tier,
        items: Sequence[BatchItem],
        ids: list[str],
        *,
        system: str,
        context: str | None,
        output_model: type[T] | None,
        purpose: str,
        poll_seconds: float,
        timeout_seconds: float,
        temperature: float | None,
        on_timeout: Literal["raise", "sync"],
        max_concurrency: int | None,
    ) -> dict[str, BatchResult[T]]:

        cfg = tier_config(tier)
        requests = []
        estimate = 0.0
        for item in items:
            params = self._params(
                cfg,
                system=system,
                prompt=item.prompt,
                context=context,
                max_tokens=item.max_tokens,
                temperature=temperature,
            )
            if output_model is not None:
                params["output_config"] = {
                    **params.get("output_config", {}),
                    "format": {
                        "type": "json_schema",
                        "schema": anthropic.transform_schema(output_model),
                    },
                }
            estimate += self._estimate_usd(params, batch=True)
            requests.append(
                Request(custom_id=item.custom_id, params=MessageCreateParamsNonStreaming(**params))
            )
        self.tracker.check(estimate)

        created = self.client.messages.batches.create(requests=requests)
        log.info("batch %s submitted: %d requests (%s)", created.id, len(requests), purpose)
        waited = 0.0
        while True:
            status = self.client.messages.batches.retrieve(created.id)
            if status.processing_status == "ended":
                break
            if waited >= timeout_seconds:
                self.client.messages.batches.cancel(created.id)
                if on_timeout == "sync":
                    log.warning(
                        "%s: batch %s timed out after %.0fs; cancelled, running %d items"
                        " synchronously",
                        purpose,
                        created.id,
                        waited,
                        len(items),
                    )
                    return self.run_many(
                        tier,
                        items,
                        system=system,
                        context=context,
                        output_model=output_model,
                        purpose=purpose,
                        temperature=temperature,
                        max_concurrency=max_concurrency,
                    )
                raise BatchTimeout(f"{purpose}: batch {created.id} timed out after {waited:.0f}s")
            self._sleep(poll_seconds)
            waited += poll_seconds

        results: dict[str, BatchResult[T]] = {}
        for entry in self.client.messages.batches.results(created.id):
            results[entry.custom_id] = self._batch_result(
                entry, tier, cfg.model, output_model, purpose
            )
        for custom_id in ids:
            results.setdefault(custom_id, BatchResult(custom_id, error="missing from results"))
        return results

    def _batch_result(
        self,
        entry: Any,
        tier: Tier,
        model: str,
        output_model: type[T] | None,
        purpose: str,
    ) -> BatchResult[T]:
        cid = entry.custom_id
        if entry.result.type != "succeeded":
            detail = getattr(getattr(entry.result, "error", None), "type", entry.result.type)
            return BatchResult(cid, error=str(detail))
        message = entry.result.message
        usage = Usage.from_api(message.usage)
        sp = tracing.current_span().set(model=model)
        for key, value in _usage_attrs(usage).items():
            sp.add(key, value)
        sp.add("usd", round(usd_for(model, usage, batch=True), 6))
        # Record before validating: the tokens were billed either way.
        self.tracker.record(
            tier=tier,
            model=model,
            usage=usage,
            batch=True,
            purpose=f"{purpose}:{cid}",
        )
        try:
            self._check_stop(message, cid)
        except LLMError as e:
            return BatchResult(cid, error=str(e))
        text = self._text(message)
        if output_model is None:
            return BatchResult(cid, value=text)
        try:
            return BatchResult(cid, value=output_model.model_validate_json(text))
        except ValidationError as e:
            return BatchResult(cid, error=f"schema mismatch: {e.error_count()} errors")

    def guard_batch(
        self,
        tier: Tier,
        items: Sequence[BatchItem],
        results: dict[str, BatchResult[Any]],
        *,
        system: str,
        guard: Callable[[str, Any], GuardResult],
        fallback: Callable[[str], Any],
        context: str | None = None,
        output_model: type[T] | None = None,
        purpose: str = "",
        temperature: float | None = None,
        max_concurrency: int | None = None,
    ) -> dict[str, Guarded[Any]]:
        """Apply the number guard to `batch()` (or `run_many()`) results, keyed by custom_id.

        `guard(custom_id, value)` and `fallback(custom_id)` take the item's ID so each
        item can be checked against its own facts. A failing item is retried once
        synchronously (full price, counted toward MAX_RUN_USD) in the same conversation,
        then falls back; items are checked `max_concurrency` at a time, so `guard` and
        `fallback` must be safe to call from several threads. Items that errored in the
        batch go straight to the fallback. Pass the same `system`, `context`,
        `output_model` and `temperature` as the batch call.
        """
        cfg = tier_config(tier)

        def one(item: BatchItem) -> tuple[str, Guarded[Any]]:
            cid = item.custom_id
            res = results.get(cid)
            if res is None or not res.ok:
                log.warning(
                    "%s:%s: batch item failed (%s); using fallback",
                    purpose,
                    cid,
                    None if res is None else res.error,
                )
                return cid, Guarded(fallback(cid), "template", attempts=1)
            params = self._params(
                cfg,
                system=system,
                prompt=item.prompt,
                context=context,
                max_tokens=item.max_tokens,
                temperature=temperature,
            )
            return cid, self._guarded(
                tier,
                params,
                res.value,
                lambda value: guard(cid, value),
                lambda: fallback(cid),
                output_model,
                f"{purpose}:{cid}",
            )

        return dict(self._map(one, list(items), max_concurrency))
