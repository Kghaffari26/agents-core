"""The README's "complete example agent" is executed here, end to end through the
runner (mocked FRED + a fake Anthropic client), so the documented code can't rot."""

import json
import re
import sys
import types
from pathlib import Path

import httpx
import pytest

from agents_core import runner
from agents_core.http import Http
from tests.conftest import FakeClient
from tests.loop_helpers import finish, resp, text, use

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture
def example(monkeypatch):
    section = README.read_text().split("## A complete example agent", 1)[1].split("\n## ", 1)[0]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    module = types.ModuleType("fed_agent_example")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(code, "README.md#example", "exec"), module.__dict__)
    return module


def fred(request: httpx.Request) -> httpx.Response:
    sid = request.url.params["series_id"]
    values = [103.1, *[101.0] * 11, 100.0] if sid == "CPIAUCSL" else [4.2] * 13
    return httpx.Response(200, json={"observations": [{"value": str(v)} for v in values]})


def run_example(example, responses, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-secret-123456")
    client = FakeClient(responses)
    http = Http(transport=httpx.MockTransport(fred))
    code = runner.run(example.AGENT, http=http, llm_client=client)
    return code, client


def test_example_agent_publishes_a_guarded_loop_brief(example, isolated_paths, monkeypatch):
    summary = "Consumer prices are up 3.1% on the year. Unemployment is 4.2%."
    code, client = run_example(
        example,
        [
            resp(
                text("Looking up both series."), use("t1", "get_series", {"series_id": "CPI_YOY"})
            ),
            resp(use("t2", "get_series", {"series_id": "UNRATE"})),
            finish(summary=summary, series_used=["CPI_YOY", "UNRATE"]),
        ],
        monkeypatch,
    )
    assert code == 0
    root = isolated_paths / "public_data"
    latest = json.loads((root / "latest.json").read_text())
    assert latest["cpi_yoy"] == 3.1 and latest["unemployment"] == 4.2
    assert latest["brief"] == summary and latest["narrative_source"] == "llm"
    assert latest["meta"]["warnings"] == []

    entry = json.loads((root / "manifest-entry.json").read_text())
    assert entry["trace_summary"]["steps"] == 3
    assert entry["trace_summary"]["tool_calls"] == 2
    assert entry["trace_summary"]["llm_calls"] == 3

    trace_text = (root / "trace.json").read_text()
    assert "fred-secret-123456" not in trace_text
    kinds = [s["kind"] for s in json.loads(trace_text)["spans"]]
    assert kinds.count("http") == 2 and "agent_loop" in kinds and "guard" in kinds


def test_example_agent_falls_back_to_the_template(example, isolated_paths, monkeypatch):
    code, _ = run_example(
        example,
        [
            finish(summary="Prices rose 5%.", series_used=["CPI_YOY"]),
            finish("f2", summary="Prices rose 6%.", series_used=["CPI_YOY"]),
        ],
        monkeypatch,
    )
    assert code == 0
    latest = json.loads((isolated_paths / "public_data" / "latest.json").read_text())
    assert latest["narrative_source"] == "template"
    assert latest["brief"].startswith("Consumer prices are up 3.1% from a year ago.")


def test_example_agent_warns_when_the_loop_fails(example, isolated_paths, monkeypatch):
    code, _ = run_example(example, [resp(text("Done."), stop_reason="end_turn")], monkeypatch)
    assert code == 0
    latest = json.loads((isolated_paths / "public_data" / "latest.json").read_text())
    assert latest["narrative_source"] == "template"
    assert latest["meta"]["warnings"] == [
        "brief loop stopped (end_turn_without_finish); published the template"
    ]
