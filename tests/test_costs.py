import json
from datetime import UTC, datetime

import pytest

from core import costs
from core.costs import BudgetExceeded, CostTracker, Usage


def test_usd_for_includes_cache_and_batch_discount():
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    # sonnet-5: 2.00 + 1.00 + 2.50 + 0.20
    assert costs.usd_for("claude-sonnet-5", usage) == pytest.approx(5.70)
    assert costs.usd_for("claude-sonnet-5", usage, batch=True) == pytest.approx(2.85)


def test_unpriced_model_fails_closed():
    with pytest.raises(KeyError, match="No pricing"):
        costs.usd_for("claude-unknown", Usage(input_tokens=1))


def test_tracker_logs_and_aggregates_by_tier(isolated_paths):
    t = CostTracker(agent="macro", run_id="r1")
    t.record(
        tier="smart",
        model="claude-sonnet-5",
        usage=Usage(input_tokens=100, output_tokens=50, cache_read_tokens=900),
        purpose="brief",
    )
    t.record(
        tier="fast",
        model="claude-haiku-4-5-20251001",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    lines = [json.loads(x) for x in (isolated_paths / "costs.jsonl").read_text().splitlines()]
    assert [line["tier"] for line in lines] == ["smart", "fast"]
    assert lines[0]["purpose"] == "brief"
    usage = t.model_usage()
    assert usage.smart.input_tokens == 1000  # uncached + cache reads
    assert usage.smart.output_tokens == 50
    assert usage.fast.input_tokens == 10
    assert t.calls == 2


def test_tracker_raises_when_over_cap():
    t = CostTracker(agent="grants", run_id="r1", max_usd=0.01)
    t.check(0.005)
    with pytest.raises(BudgetExceeded):
        t.check(0.02)
    with pytest.raises(BudgetExceeded):
        t.record(tier="smart", model="claude-sonnet-5", usage=Usage(output_tokens=10_000))
    # The spend is still logged even though the run is aborted.
    assert t.total_usd == pytest.approx(0.1)


def test_max_run_usd_from_env(monkeypatch):
    monkeypatch.setenv("MAX_RUN_USD", "0.25")
    assert CostTracker(agent="a", run_id="r").max_usd == 0.25


def test_summarize_month_and_runs():
    entries = [
        {"ts": "2026-08-31T10:00:00Z", "agent": "macro", "run_id": "a", "usd": 1.0},
        {"ts": "2026-09-01T10:00:00Z", "agent": "macro", "run_id": "b", "usd": 0.10},
        {"ts": "2026-09-01T10:00:01Z", "agent": "macro", "run_id": "b", "usd": 0.05},
        {"ts": "2026-09-02T10:00:00Z", "agent": "grants", "run_id": "c", "usd": 0.20},
        {
            "ts": "2026-09-03T10:00:00Z",
            "kind": "run",
            "agent": "grants",
            "run_id": "d",
            "run_usd": 0,
        },
    ]
    s = costs.summarize(entries, now=datetime(2026, 9, 15, tzinfo=UTC))
    assert s.month == "2026-09"
    assert s.total_usd == pytest.approx(0.35)
    assert {a.agent: (a.usd, a.runs) for a in s.by_agent} == {
        "grants": (0.2, 2),
        "macro": (0.15, 1),
    }
    assert [d.date for d in s.daily] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert s.all_time_usd == pytest.approx(1.35)
    assert {a.agent: a.usd for a in s.avg_cost_per_run} == {"grants": 0.1, "macro": 0.575}
