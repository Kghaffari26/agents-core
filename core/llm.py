"""The only module that imports the Anthropic SDK.

Agents ask for a tier ("fast" or "smart"), never a model ID. Every call goes through
the run's CostTracker, which logs tokens/USD and enforces MAX_RUN_USD before and after
each call. Prompts are laid out for caching: stable `system` and `context` blocks first
(cache breakpoint on the last of them), per-run data in the user message after it.

Numbers come from data, never from the model: pass computed figures in the prompt and
ask for narrative only. Validate narrative against those figures with the evals.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from core import settings
from core.costs import CostTracker, Tier, Usage, usd_for

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Rough chars-per-token used only for the pre-call budget estimate.
_CHARS_PER_TOKEN = 3.5
_SDK_MAX_RETRIES = 4


class LLMError(RuntimeError):
    """The model returned something unusable (refusal, truncation, schema mismatch)."""


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

    def complete(
        self,
        tier: Tier,
        prompt: str,
        *,
        system: str,
        context: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
    ) -> str:
        """Return plain text. Use `structured` when the output feeds a schema."""
        cfg = tier_config(tier)
        params = self._params(
            cfg, system=system, prompt=prompt, context=context, max_tokens=max_tokens
        )
        self.tracker.check(self._estimate_usd(params))
        message = self.client.messages.create(**params)
        self.tracker.record(
            tier=tier, model=cfg.model, usage=Usage.from_api(message.usage), purpose=purpose
        )
        self._check_stop(message, purpose)
        return self._text(message)

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
    ) -> T:
        """Return a validated `output_model` instance using structured outputs."""
        cfg = tier_config(tier)
        params = self._params(
            cfg, system=system, prompt=prompt, context=context, max_tokens=max_tokens
        )
        self.tracker.check(self._estimate_usd(params))
        message = self.client.messages.parse(output_format=output_model, **params)
        self.tracker.record(
            tier=tier, model=cfg.model, usage=Usage.from_api(message.usage), purpose=purpose
        )
        self._check_stop(message, purpose)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"{purpose}: no parsed output returned")
        return parsed

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
