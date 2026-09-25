"""The contract every agent implements, and the context the runner hands it.

An agent repo registers a module-level `Agent` instance under the
`agents_core.agents` entry-point group (see `agents_core.registry`) and depends on
this package to get `agents-run <name>` for free. The runner calls
fetch -> transform -> analyze; the agent never writes files itself.

    fetch(ctx)            download source data through ctx.http (cached, rate-limited)
    transform(ctx, raw)   pure Python: compute every number that gets published. No LLM.
    analyze(ctx, data)    LLM narrative via ctx.llm, returns an AgentResult

`--dry-run` stops after transform, so fetch and transform must not call the LLM.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from agents_core import settings
from agents_core.costs import CostTracker
from agents_core.http import Http
from agents_core.llm import LLM
from agents_core.schema import AgentOutput, KeyStat, Source


@dataclass
class RunContext:
    agent_id: str
    run_id: str
    started_at: datetime
    http: Http
    llm: LLM
    costs: CostTracker
    dry_run: bool = False
    apply: bool = False
    # Unrecognized `agents-run` CLI arguments, forwarded verbatim so an agent can
    # define flags of its own (e.g. "--force-briefs") without this package knowing
    # about them. Ignored unless the agent chooses to read it.
    extra_args: list[str] = field(default_factory=list)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("agent"))

    def previous_latest(self) -> dict[str, Any] | None:
        """The previous `latest.json` as a dict, or None if there isn't one locally
        (a fresh CI checkout normally won't have one unless the workflow first checks
        out the agent's `data` branch into the publish dir). Use it to reuse narrative
        when source data hasn't changed (and set `data_changed=False`)."""
        path = settings.publish_dir() / "latest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None


@dataclass
class AgentResult:
    """What `analyze` returns. The runner adds `meta` and validates against output_model.

    `body` holds every top-level field of `latest.json` except `meta`.
    `files` maps extra paths under the publish dir (e.g. "metros/austin-tx.json",
    "all.json") to validated models; they are written before `latest.json`.
    `headline` and `key_stats` feed an overview card and must be built
    deterministically from computed data, not by the LLM.
    """

    body: dict[str, Any] | BaseModel
    sources: list[Source]
    headline: str
    key_stats: list[KeyStat] = field(default_factory=list)
    data_changed: bool = True
    items_count: int | None = None
    files: dict[str, BaseModel] = field(default_factory=dict)
    status: Literal["ok", "stale"] = "ok"


class Agent(ABC):
    id: ClassVar[str]
    name: ClassVar[str]
    route: ClassVar[str]
    schema_version: ClassVar[str]
    expected_interval_hours: ClassVar[int]
    next_run_hint: ClassVar[str]
    # How many history/YYYY-MM-DD.json files to keep: 52 for weekly agents, 90 for daily.
    history_keep: ClassVar[int]
    output_model: ClassVar[type[AgentOutput]]

    def configure_http(self, http: Http) -> None:  # noqa: B027 - optional hook
        """Register per-host rate limits and request budgets, e.g. SAM.gov's daily cap."""

    @abstractmethod
    def fetch(self, ctx: RunContext) -> Any: ...

    @abstractmethod
    def transform(self, ctx: RunContext, raw: Any) -> Any: ...

    @abstractmethod
    def analyze(self, ctx: RunContext, data: Any) -> AgentResult: ...

    def summarize_dry_run(self, data: Any) -> str:
        """One-line description of transform output, printed by --dry-run."""
        return f"transform returned {type(data).__name__}"
