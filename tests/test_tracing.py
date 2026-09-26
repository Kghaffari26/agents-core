import json
import threading

import httpx
import pytest

from agents_core import export_schemas, runner, tracing
from agents_core.costs import CostTracker
from agents_core.guards import text_guard
from agents_core.http import Http
from agents_core.llm import LLM, BatchItem
from agents_core.publish import publish_output
from agents_core.schema import Trace, TraceSummary
from tests.conftest import FakeClient, fake_message
from tests.fake_agent import FakeAgent, FakeOutput


def fake_clock(step=0.01):
    t = [0.0]

    def clock():
        t[0] += step
        return t[0]

    return clock


# ---- spans ------------------------------------------------------------------------


def test_spans_nest_and_record_status():
    tracer = tracing.Tracer(agent="a", run_id="r")
    with tracing.use(tracer), tracing.span("run", "a") as root:
        with tracing.span("custom", "step", n=1) as child:
            child.set(kept=2).add("count").add("count")
        with pytest.raises(ValueError), tracing.span("custom", "boom"):
            raise ValueError("bad input")
    assert root.parent_id is None
    step, boom = tracer.spans[1:]
    assert step.parent_id == root.id and step.attrs == {"n": 1, "kept": 2, "count": 2}
    assert boom.status == "error" and boom.error == "ValueError: bad input"
    assert all(s.duration_ms is not None for s in tracer.spans)


def test_span_is_a_noop_without_a_tracer():
    with tracing.span("custom", "x") as s:
        s.set(a=1)
    assert s is tracing.NULL_SPAN and s.attrs == {}
    assert tracing.current_tracer() is None


def test_summary_rolls_up_steps_calls_latency_cost_and_guard_retries():
    tracer = tracing.Tracer(clock=fake_clock())
    with tracer.span("run", "a"):
        with tracer.span("agent_loop", "l") as loop:
            loop.set(steps=3)
        for usd in (0.01, 0.02):
            with tracer.span("llm_call", "c", usd=usd):
                pass
        with tracer.span("tool_call", "t"):
            pass
        with tracer.span("guard", "g", attempts=2):
            pass
        with tracer.span("guard", "g", attempts=1):
            pass
    s = tracer.summary()
    assert s == TraceSummary(
        steps=3,
        tool_calls=1,
        llm_calls=2,
        total_latency_ms=s.total_latency_ms,
        cost_usd=0.03,
        guard_retries=1,
    )
    assert s.total_latency_ms > 0


# ---- automatic instrumentation ------------------------------------------------------


def test_llm_calls_and_guard_retries_are_traced():
    tracer = tracing.Tracer()
    llm = LLM(
        CostTracker(agent="a", run_id="r"),
        client=FakeClient([fake_message("Rose 9.9%."), fake_message("Rose 2.5%.")]),
    )
    with tracing.use(tracer):
        out = llm.complete("smart", "p", system="s", purpose="brief", guard=text_guard([2.5]))
    assert out.attempts == 2
    first, guard, retry = tracer.spans
    assert guard.kind == "guard" and guard.attrs["outcome"] == "pass_after_retry"
    assert guard.attrs["attempts"] == 2
    assert first.kind == retry.kind == "llm_call"
    assert retry.parent_id == guard.id  # the retry is part of the guard's work
    assert first.attrs["model"] == "claude-sonnet-5" and first.attrs["input_tokens"] == 1000
    assert first.attrs["usd"] == pytest.approx(0.004)
    assert first.attrs["estimated_usd"] > first.attrs["usd"]
    assert retry.name == "brief:guard-retry"
    assert tracer.summary().guard_retries == 1


def test_concurrent_calls_nest_under_the_calling_span():
    tracer = tracing.Tracer()
    llm = LLM(
        CostTracker(agent="a", run_id="r"),
        client=FakeClient([fake_message(str(i)) for i in range(4)]),
        max_concurrency=4,
    )
    lock = threading.Lock()
    original = llm.client.messages.create

    def create(**params):
        with lock:
            return original(**params)

    llm.client.messages.create = create
    with tracing.use(tracer), tracing.span("custom", "fan-out") as parent:
        llm.run_many("fast", [BatchItem(str(i), "p") for i in range(4)], system="s")
    calls = [s for s in tracer.spans if s.kind == "llm_call"]
    assert len(calls) == 4 and all(s.parent_id == parent.id for s in calls)


def test_http_requests_are_traced_with_retries_and_cache_hits():
    statuses = [503, 200]

    def handler(request):
        return httpx.Response(statuses.pop(0) if statuses else 200, json={"ok": True})

    tracer = tracing.Tracer()
    http = Http(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with tracing.use(tracer):
        http.get("https://api.example.gov/x", params={"api_key": "SECRETVALUE"})
        http.get("https://api.example.gov/x", params={"api_key": "SECRETVALUE"})
    first, second = tracer.spans
    assert first.kind == "http" and first.name == "GET api.example.gov"
    assert first.attrs["retries"] == 1 and first.attrs["status"] == 200
    assert first.attrs["from_cache"] is False and second.attrs["from_cache"] is True
    assert "SECRETVALUE" not in json.dumps(first.attrs)


# ---- redaction and size cap -----------------------------------------------------------


def test_redaction_of_keys_formats_env_values_and_query_params(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fredsecret12345")
    data = {
        "api_key": "abc",
        "headers": {"Authorization": "Bearer abcdefghijklmnop", "x-api-key": "zzz"},
        "input_tokens": 1200,
        "max_tokens": 4000,
        "github_token": "whatever",
        "note": "called with sk-ant-api03-AAAAAAAAAAAAAAAA and ghp_" + "b" * 36,
        "url": "https://x.test/a?series=CPI&api_key=live123&token=t0k",
        "echo": "the key is fredsecret12345!",
        "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "list": [{"password": "hunter22"}],
    }
    out = tracing.redact(data)
    text = json.dumps(out)
    for secret in (
        'abc"',
        "abcdefghijklmnop",
        "zzz",
        "whatever",
        "sk-ant-api03",
        "bbbbbbbb",
        "live123",
        "t0k",
        "fredsecret12345",
        "eyJhbGci",
        "AKIAIOSFODNN7EXAMPLE",
        "hunter22",
    ):
        assert secret not in text, secret
    assert out["input_tokens"] == 1200 and out["max_tokens"] == 4000
    assert out["url"] == "https://x.test/a?series=CPI&api_key=***&token=***"
    assert out["echo"] == "the key is ***!"


def written_size(data):
    # how publish.write_json serializes it
    return len(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode())


def test_fit_trace_shortens_strings_then_drops_latest_spans():
    tracer = tracing.Tracer()
    for i in range(50):
        with tracer.span("tool_call", f"t{i}", output="y" * 5000):
            pass
    small = tracing.trace_dict(tracer, max_bytes=20_000)
    assert written_size(small) <= 20_000
    assert small["truncated"] is True
    assert "chars cut" in small["spans"][0]["attrs"]["output"]

    tiny = tracing.trace_dict(tracer, max_bytes=3_000)
    assert written_size(tiny) <= 3_000
    assert tiny["dropped_spans"] > 0
    assert len(tiny["spans"]) + tiny["dropped_spans"] == 50
    assert tiny["spans"][0]["name"] == "t0"  # the earliest spans are kept
    assert tiny["summary"]["tool_calls"] == 50  # the summary is exact regardless


def test_write_trace_is_redacted_capped_and_schema_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live-key-that-must-not-leak")
    tracer = tracing.Tracer(agent="macro", run_id="r1")
    with tracer.span("tool_call", "t", input={"q": "sk-live-key-that-must-not-leak"}):
        pass
    path = tracing.write_trace(tracer, tmp_path, max_bytes=10_000)
    raw = path.read_text()
    assert path.stat().st_size <= 10_000
    assert "sk-live-key" not in raw
    trace = Trace.model_validate_json(raw)
    assert trace.agent == "macro" and trace.spans[0].attrs["input"] == {"q": "***"}


def test_trace_schema_export(tmp_path):
    path = export_schemas.write_trace_schema(tmp_path)
    schema = json.loads(path.read_text())
    assert path.name == "trace.schema.json"
    assert {"spans", "summary", "agent", "run_id"} <= set(schema["properties"])
    assert export_schemas.main(["--trace"]) == 0


@pytest.mark.parametrize("name", ["trace.json", "trace.schema.json"])
def test_trace_files_are_reserved_names(name):
    with pytest.raises(ValueError, match="reserved"):
        publish_output(FakeOutput.model_construct(), files={name: TraceSummary()}, history_keep=1)


# ---- the runner publishes trace.json + trace_summary -------------------------------------


def test_run_publishes_trace_and_manifest_trace_summary(isolated_paths, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-supersecretvalue")
    assert runner.run(FakeAgent(), llm_client=FakeClient([fake_message("Index rose.")])) == 0
    root = isolated_paths / "public_data"
    trace = json.loads((root / "trace.json").read_text())
    Trace.model_validate(trace)
    kinds = [s["kind"] for s in trace["spans"]]
    assert kinds[0] == "run" and kinds.count("phase") == 4 and "llm_call" in kinds
    run_span = trace["spans"][0]
    assert run_span["attrs"]["status"] == "ok" and run_span["parent_id"] is None
    phases = [s["name"] for s in trace["spans"] if s["kind"] == "phase"]
    assert phases == ["fetch", "transform", "analyze", "publish"]
    assert "supersecretvalue" not in (root / "trace.json").read_text()
    assert (root / "trace.schema.json").is_file()

    entry = json.loads((root / "manifest-entry.json").read_text())
    ts = entry["trace_summary"]
    assert ts["llm_calls"] == 1 and ts["cost_usd"] > 0 and ts["total_latency_ms"] > 0
    assert ts["steps"] == 0 and ts["tool_calls"] == 0 and ts["guard_retries"] == 0


def test_failed_run_still_publishes_its_trace(isolated_paths):
    assert runner.run(FakeAgent(fail_in="analyze"), llm_client=FakeClient([])) == 1
    root = isolated_paths / "public_data"
    trace = json.loads((root / "trace.json").read_text())
    run_span = trace["spans"][0]
    assert run_span["status"] == "error" and "boom" in run_span["error"]
    analyze = next(s for s in trace["spans"] if s["name"] == "analyze")
    assert analyze["status"] == "error"
    entry = json.loads((root / "manifest-entry.json").read_text())
    assert entry["status"] == "failed" and entry["trace_summary"]["llm_calls"] == 0


def test_dry_run_writes_no_trace(isolated_paths):
    assert runner.run(FakeAgent(), dry_run=True) == 0
    assert not (isolated_paths / "public_data" / "trace.json").exists()


def test_old_manifest_entries_without_trace_summary_still_validate():
    from agents_core.schema import ManifestEntry

    old = {
        "id": "macro",
        "name": "M",
        "route": "/m",
        "status": "ok",
        "last_run_at": "2026-09-01T00:00:00Z",
        "last_data_change_at": None,
        "expected_interval_hours": 24,
        "next_run_hint": "daily",
        "headline": "h",
        "key_stats": [],
        "run_cost_usd": 0.01,
        "items_count": None,
    }
    assert ManifestEntry.model_validate(old).trace_summary is None
