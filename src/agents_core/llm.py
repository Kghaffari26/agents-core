"""The only module that imports the Anthropic SDK.

Agents ask for a tier ("fast" or "smart"), never a model ID. Every call goes through
the run's CostTracker, which logs tokens/USD and enforces MAX_RUN_USD before and after
each call. Prompts are laid out for caching: stable `system` and `context` blocks first
(cache breakpoint on the last of them), per-run data in the user message after it.

Numbers come from data, never from the model: pass computed figures in the prompt and
ask for narrative only. Pass `guard=` (see core/guards.py) to check the narrative: a
failing output is retried once with the unsupported numbers named, then replaced by
`fallback()`. Guarded calls return `Guarded(value, narrative_source, ...)`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar, overload

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from agents_core import settings
from agents_core.costs import CostTracker, Tier, Usage, usd_for
from agents_core.guards import GuardResult
from agents_core.schema import NarrativeSource, iso_z

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V")

# Rough chars-per-token used only for the pre-call budget estimate.
_CHARS_PER_TOKEN = 3.5
_SDK_MAX_RETRIES = 4


class LLMError(RuntimeError):
    """The model returned something unusable (refusal, truncation, schema mismatch)."""


class GuardFailed(LLMError):
    """The number guard failed twice and no fallback was given."""


RETRY_INSTRUCTION = (
    "These numbers are not in the input: [{tokens}]. Rewrite using only numbers provided."
)


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


@dataclass(frozen=True)
class TierConfig:
    model: str
    max_tokens: int
    effort: str | None = None
    thinking: str | None = None  # "adaptive" | "disabled" | None (model default)


def tier_config(tier: Tier) -> TierConfig:
    cfg = settings.load_config("models")["tiers"][tier]
    return TierConfig(
        model=cfg["model"],
        max_tokens=int(cfg["max_tokens"]),
        effort=cfg.get("effort"),
        thinking=cfg.get("thinking"),
    )


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
    ) -> None:
        self.tracker = tracker
        self._client = client
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            settings.require_env("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(max_retries=_SDK_MAX_RETRIES)
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
        return params

    @staticmethod
    def _estimate_usd(params: dict[str, Any], *, batch: bool = False) -> float:
        """Worst-case cost: all input uncached, output runs to max_tokens."""
        chars = len(json.dumps(params["system"])) + len(json.dumps(params["messages"]))
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
        self.tracker.check(self._estimate_usd(params))
        if output_model is None:
            message = self.client.messages.create(**params)
        else:
            message = self.client.messages.parse(output_format=output_model, **params)
        self.tracker.record(
            tier=tier, model=params["model"], usage=Usage.from_api(message.usage), purpose=purpose
        )
        self._check_stop(message, purpose)
        if output_model is None:
            return self._text(message)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"{purpose}: no parsed output returned")
        return parsed

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
        guard: Callable[[str], GuardResult] | None = None,
        fallback: Callable[[], str] | None = None,
    ) -> str | Guarded[str]:
        """Return plain text, or `Guarded[str]` when a guard is given.

        Use `structured` when the output feeds a schema.
        """
        cfg = tier_config(tier)
        params = self._params(
            cfg, system=system, prompt=prompt, context=context, max_tokens=max_tokens
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
        guard: Callable[[T], GuardResult] | None = None,
        fallback: Callable[[], T] | None = None,
    ) -> T | Guarded[T]:
        """Return a validated `output_model` instance using structured outputs, or
        `Guarded[output_model]` when a guard is given. Build the guard with
        `agents_core.guards.fields_guard(facts, [...narrative fields...])`.
        """
        cfg = tier_config(tier)
        params = self._params(
            cfg, system=system, prompt=prompt, context=context, max_tokens=max_tokens
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
                    "content": RETRY_INSTRUCTION.format(tokens=", ".join(result.unsupported)),
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        log.warning("%s: number guard attempt %d failed: %s", purpose, attempt, unsupported)

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
    ) -> dict[str, BatchResult[T]]:
        """Run many independent prompts through the Batch API (50% price, async).

        Blocks until the batch ends. Returns results keyed by custom_id; failed items
        carry `error` instead of `value`. The whole batch's worst-case cost is checked
        against the budget before submitting.
        """
        if not items:
            return {}
        ids = [i.custom_id for i in items]
        if len(set(ids)) != len(ids):
            raise ValueError("batch custom_ids must be unique")

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
                raise LLMError(f"{purpose}: batch {created.id} timed out after {waited:.0f}s")
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
        # Record before validating: the tokens were billed either way.
        self.tracker.record(
            tier=tier,
            model=model,
            usage=Usage.from_api(message.usage),
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
    ) -> dict[str, Guarded[Any]]:
        """Apply the number guard to `batch()` results, keyed by custom_id.

        `guard(custom_id, value)` and `fallback(custom_id)` take the item's ID so each
        item can be checked against its own facts. A failing item is retried once
        synchronously (full price, counted toward MAX_RUN_USD) in the same conversation,
        then falls back. Items that errored in the batch go straight to the fallback.
        Pass the same `system`, `context` and `output_model` as the batch call.
        """
        cfg = tier_config(tier)
        guarded: dict[str, Guarded[Any]] = {}
        for item in items:
            cid = item.custom_id
            res = results.get(cid)
            if res is None or not res.ok:
                log.warning(
                    "%s:%s: batch item failed (%s); using fallback",
                    purpose,
                    cid,
                    None if res is None else res.error,
                )
                guarded[cid] = Guarded(fallback(cid), "template", attempts=1)
                continue
            params = self._params(
                cfg, system=system, prompt=item.prompt, context=context, max_tokens=item.max_tokens
            )
            guarded[cid] = self._guarded(
                tier,
                params,
                res.value,
                lambda value, cid=cid: guard(cid, value),
                lambda cid=cid: fallback(cid),
                output_model,
                f"{purpose}:{cid}",
            )
        return guarded
