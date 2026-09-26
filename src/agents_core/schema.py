"""Shared pydantic models for every agent's published JSON.

`RunMeta`/`AgentOutput` are the base of `latest.json`. `ManifestEntry` and
`CostsSummary` are the single-agent `manifest-entry.json` / `costs-summary.json`
files each agent publishes at the root of its own data branch — see the README's
data-branch contract. There is no cross-agent manifest here: assembling one across
agents is the consuming website's job, not this package's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, PlainSerializer

RunStatus = Literal["ok", "stale", "failed"]
# "llm" when the narrative passed the number guard. "template" for every other
# narrative: the guard's fallback, and any deterministic, non-LLM text an agent
# builds itself (e.g. a --dry-run brief, or text reused because the model is skipped).
NarrativeSource = Literal["llm", "template"]

# Version of the shared `meta` block's own shape (independent of each agent's
# `schema_version`, which versions the agent's whole latest.json). 1.1.0 added
# `warnings` and `meta_schema_version`, and allowed agent-specific meta subclasses.
META_SCHEMA_VERSION = "1.1.0"
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
    """The `meta` block at the top of every agent's `latest.json`.

    Agent-specific meta fields: subclass `RunMeta` (fields need defaults), declare
    `meta: MyMeta` on your `AgentOutput` subclass, and return the values in
    `AgentResult.meta_fields`. The runner merges them into `meta` before validating,
    so they're published in latest.json and appear in schema.json.
    """

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
    # Non-fatal problems in an otherwise usable run ("ok with a warning"), in plain
    # language. From `AgentResult.warnings`.
    warnings: list[str] = Field(default_factory=list)
    meta_schema_version: str = Field(default=META_SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")


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


class DailyCost(Model):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    usd: float


class CostsSummary(Model):
    """Published as `costs-summary.json`: this agent's own spend only."""

    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    total_usd: float
    runs: int
    daily: list[DailyCost]
    all_time_usd: float
