"""Builders for fake Messages API responses with tool use, shared by the loop, eval
and tracing tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from agents_core.agent_loop import tool


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def use(tool_id: str, name: str, input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input)


def resp(
    *blocks: SimpleNamespace,
    stop_reason: str = "tool_use",
    input_tokens: int = 1000,
    output_tokens: int = 200,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def finish(tool_id: str = "f1", **result: Any) -> SimpleNamespace:
    return resp(use(tool_id, "finish", result))


class SeriesQuery(BaseModel):
    series_id: str


class Brief(BaseModel):
    summary: str
    series_used: list[str] = []


SERIES = {"CPI": 3.1, "UNRATE": 4.2}


@tool
def get_series(args: SeriesQuery) -> dict[str, Any]:
    """Latest value of one data series."""
    if args.series_id not in SERIES:
        raise KeyError(args.series_id)
    return {"series_id": args.series_id, "value": SERIES[args.series_id]}


class Comment(BaseModel):
    issue: int
    body: str


POSTED: list[Comment] = []


@tool(requires_approval=True)
def post_comment(args: Comment) -> str:
    """Post a comment on a GitHub issue."""
    POSTED.append(args)
    return "posted"
