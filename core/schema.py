"""Shared pydantic models for every agent's published JSON.

These shapes are the contract with the website (docs/specs/SPEC_WEBSITE.md §3).
Change them only together with the site's generated types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, PlainSerializer

RunStatus = Literal["ok", "stale", "failed"]
GoodDirection = Literal["up", "down", "neutral"]
StatFormat = Literal[
    "currency_compact",
    "currency",
    "percent",
    "percent_signed",
    "pp_signed",
    "count",
    "count_signed",
    "count_signed_thousands",
    "decimal1",
    "days",
    "ratio",
]


def iso_z(dt: datetime) -> str:
    """Format an aware datetime as `2026-09-23T14:00:05Z` (UTC, second precision)."""
    return dt.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


# Timezone-aware datetime published as `YYYY-MM-DDTHH:MM:SSZ`. Use it for every timestamp.
Timestamp = Annotated[AwareDatetime, PlainSerializer(iso_z, return_type=str, when_used="json")]


class Model(BaseModel):
    """Base for published models: unknown fields are a validation error."""

    # Defaulted fields are always present in published JSON, so the exported schema marks
    # them required and the site's generated TypeScript types are not optional.
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Source(Model):
    """A dataset or page an agent pulled from, shown in the site's source list."""

    name: str
    url: HttpUrl
    retrieved_at: Timestamp


class Citation(Model):
    """Backs one narrative claim. Every claim in an AI brief carries at least one."""

    source: str = Field(description="Source name, e.g. 'FRED'")
    url: HttpUrl
    note: str | None = Field(default=None, description="What the citation supports")


class TierUsage(Model):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelUsage(Model):
    fast: TierUsage = Field(default_factory=TierUsage)
    smart: TierUsage = Field(default_factory=TierUsage)


class RunMeta(Model):
    """The `meta` block at the top of every agent's `latest.json`."""

    agent: str
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    run_id: str
    started_at: Timestamp
    finished_at: Timestamp
    status: RunStatus
    data_changed: bool
    cost_usd: float = Field(ge=0)
    model_usage: ModelUsage
    sources: list[Source]


class AgentOutput(Model):
    """Base for each agent's `latest.json` model. Agents subclass and add their fields."""

    meta: RunMeta


class KeyStat(Model):
    label: str
    value: float | None
    format: StatFormat
    delta: float | None = None
    delta_format: StatFormat | None = None
    good_direction: GoodDirection = "neutral"


class ManifestEntry(Model):
    id: str
    name: str
    route: str
    status: RunStatus
    last_run_at: Timestamp
    last_data_change_at: Timestamp | None
    expected_interval_hours: int = Field(gt=0)
    next_run_hint: str
    headline: str
    key_stats: list[KeyStat] = Field(max_length=4)
    run_cost_usd: float = Field(ge=0)
    items_count: int | None = None


class Manifest(Model):
    generated_at: Timestamp
    agents: list[ManifestEntry]


class AgentCost(Model):
    agent: str
    usd: float
    runs: int


class DailyCost(Model):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    usd: float


class AvgRunCost(Model):
    agent: str
    usd: float


class CostSummary(Model):
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    total_usd: float
    by_agent: list[AgentCost]
    daily: list[DailyCost]
    all_time_usd: float
    avg_cost_per_run: list[AvgRunCost]
