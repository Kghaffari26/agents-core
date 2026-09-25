"""A minimal agent used to exercise the runner end to end."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agents_core.agent import Agent, AgentResult, RunContext
from agents_core.schema import AgentOutput, KeyStat, Model, Source


class Point(Model):
    date: str
    value: float


class Detail(Model):
    slug: str
    series: list[Point]


class FakeOutput(AgentOutput):
    headline_value: float
    brief: str


class FakeAgent(Agent):
    id = "macro"
    name = "Macro & Fed Agent"
    route = "/macro"
    schema_version = "1.0.0"
    expected_interval_hours = 24
    next_run_hint = "Weekdays 07:00 PT"
    history_keep = 3
    output_model = FakeOutput

    def __init__(
        self,
        *,
        body: dict[str, Any] | None = None,
        fail_in: str | None = None,
        data_changed: bool = True,
        use_llm: bool = True,
    ) -> None:
        self.body = body
        self.fail_in = fail_in
        self.data_changed = data_changed
        self.use_llm = use_llm
        self.calls: list[str] = []

    def fetch(self, ctx: RunContext) -> Any:
        self.calls.append("fetch")
        if self.fail_in == "fetch":
            raise RuntimeError("source down")
        return {"values": [1.0, 2.0, 4.0]}

    def transform(self, ctx: RunContext, raw: Any) -> Any:
        self.calls.append("transform")
        return {"latest": raw["values"][-1], "change": raw["values"][-1] - raw["values"][-2]}

    def analyze(self, ctx: RunContext, data: Any) -> AgentResult:
        self.calls.append("analyze")
        if self.fail_in == "analyze":
            raise RuntimeError("boom")
        brief = (
            ctx.llm.complete("smart", f"latest={data['latest']}", system="brief")
            if self.use_llm
            else "reused"
        )
        body = (
            self.body
            if self.body is not None
            else {
                "headline_value": data["latest"],
                "brief": brief,
            }
        )
        return AgentResult(
            body=body,
            sources=[
                Source(
                    name="FRED", url="https://fred.stlouisfed.org/", retrieved_at=datetime.now(UTC)
                )
            ],
            headline=f"Index at {data['latest']:.1f}.",
            key_stats=[
                KeyStat(
                    label="Index",
                    value=data["latest"],
                    format="decimal1",
                    delta=data["change"],
                    delta_format="count_signed",
                    good_direction="up",
                )
            ],
            data_changed=self.data_changed,
            items_count=1,
            files={"series/us.json": Detail(slug="us", series=[Point(date="2026-08", value=4.0)])},
        )


# A default instance, usable directly (as in most tests) or as an
# `agents_core.agents` entry-point target for registry/CLI/install tests.
AGENT = FakeAgent()
