"""Token and USD accounting, the per-run budget cap, and the site's cost summary."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from core import settings
from core.schema import (
    AgentCost,
    AvgRunCost,
    CostSummary,
    DailyCost,
    ModelUsage,
    TierUsage,
    iso_z,
)

log = logging.getLogger(__name__)

Tier = Literal["fast", "smart"]


class BudgetExceeded(RuntimeError):
    """Raised when a run's LLM spend would pass `MAX_RUN_USD`. Fails the run."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @classmethod
    def from_api(cls, usage: Any) -> Usage:
        """Build from an SDK `usage` object; cache fields may be missing or None."""
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


def _pricing_table() -> tuple[dict[str, Price], float]:
    cfg = settings.load_config("models")["pricing"]
    prices = {model: Price(**values) for model, values in cfg["models"].items()}
    return prices, float(cfg.get("batch_discount", 0.5))


def price_for(model: str) -> Price:
    prices, _ = _pricing_table()
    if model not in prices:
        # Fail closed: an unpriced model would make the budget cap meaningless.
        raise KeyError(f"No pricing for model {model!r}; add it to config/models.toml [pricing]")
    return prices[model]


def usd_for(model: str, usage: Usage, *, batch: bool = False) -> float:
    p = price_for(model)
    usd = (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_write_tokens * p.cache_write
        + usage.cache_read_tokens * p.cache_read
    ) / 1_000_000
    if batch:
        usd *= _pricing_table()[1]
    return usd


@dataclass
class CostTracker:
    """Accumulates one run's spend, appends each call to `costs.jsonl`, enforces the cap."""

    agent: str
    run_id: str
    max_usd: float = field(default_factory=settings.max_run_usd)
    path: Path = field(default_factory=settings.costs_path)
    total_usd: float = 0.0
    calls: int = 0
    _by_tier: dict[str, TierUsage] = field(default_factory=dict)

    def check(self, estimated_usd: float = 0.0) -> None:
        """Raise before a call if spend so far plus `estimated_usd` would pass the cap."""
        if self.total_usd + estimated_usd > self.max_usd:
            raise BudgetExceeded(
                f"{self.agent}: run spend ${self.total_usd:.4f} + estimated ${estimated_usd:.4f}"
                f" would exceed MAX_RUN_USD ${self.max_usd:.2f}"
            )

    def record(
        self, *, tier: Tier, model: str, usage: Usage, batch: bool = False, purpose: str = ""
    ) -> float:
        """Log one call and add it to the run total. Raises BudgetExceeded once over the cap."""
        usd = usd_for(model, usage, batch=batch)
        self.total_usd += usd
        self.calls += 1
        tier_usage = self._by_tier.setdefault(tier, TierUsage())
        tier_usage.input_tokens += usage.total_input_tokens
        tier_usage.output_tokens += usage.output_tokens

        entry = {
            "ts": iso_z(datetime.now(UTC)),
            "agent": self.agent,
            "run_id": self.run_id,
            "tier": tier,
            "model": model,
            "purpose": purpose,
            "batch": batch,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "usd": round(usd, 6),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info("llm %s/%s %s: $%.4f (run total $%.4f)", tier, model, purpose, usd, self.total_usd)

        if self.total_usd > self.max_usd:
            raise BudgetExceeded(
                f"{self.agent}: run spend ${self.total_usd:.4f} exceeded MAX_RUN_USD"
                f" ${self.max_usd:.2f}"
            )
        return usd

    def record_run(self, status: str) -> None:
        """Append one run-level line so runs with no LLM calls still count in the summary."""
        entry = {
            "ts": iso_z(datetime.now(UTC)),
            "kind": "run",
            "agent": self.agent,
            "run_id": self.run_id,
            "status": status,
            "calls": self.calls,
            "run_usd": round(self.total_usd, 6),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def model_usage(self) -> ModelUsage:
        return ModelUsage(
            fast=self._by_tier.get("fast", TierUsage()).model_copy(),
            smart=self._by_tier.get("smart", TierUsage()).model_copy(),
        )


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or settings.costs_path()
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed line in %s", path)
    return entries


def summarize(entries: list[dict[str, Any]], now: datetime | None = None) -> CostSummary:
    """Aggregate the call log into the site's `costs/summary.json` (current UTC month)."""
    now = now or datetime.now(UTC)
    month = now.strftime("%Y-%m")

    all_time = 0.0
    month_by_agent: dict[str, float] = defaultdict(float)
    month_runs: dict[str, set[str]] = defaultdict(set)
    daily: dict[str, float] = defaultdict(float)
    all_by_agent: dict[str, float] = defaultdict(float)
    all_runs: dict[str, set[str]] = defaultdict(set)

    for e in entries:
        # Call lines carry `usd`; run lines (kind=run) carry none, so they only count runs.
        usd = float(e.get("usd", 0))
        agent, run_id, ts = e.get("agent", "?"), e.get("run_id", ""), e.get("ts", "")
        all_time += usd
        all_by_agent[agent] += usd
        all_runs[agent].add(run_id)
        if ts.startswith(month):
            month_by_agent[agent] += usd
            month_runs[agent].add(run_id)
            daily[ts[:10]] += usd

    return CostSummary(
        month=month,
        total_usd=round(sum(month_by_agent.values()), 4),
        by_agent=[
            AgentCost(agent=a, usd=round(month_by_agent[a], 4), runs=len(month_runs[a]))
            for a in sorted(month_by_agent)
        ],
        daily=[DailyCost(date=d, usd=round(daily[d], 4)) for d in sorted(daily)],
        all_time_usd=round(all_time, 4),
        avg_cost_per_run=[
            AvgRunCost(agent=a, usd=round(all_by_agent[a] / len(all_runs[a]), 4))
            for a in sorted(all_by_agent)
        ],
    )
