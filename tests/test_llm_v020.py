"""Temperature, max_concurrency, run_many and the batch-timeout fallback (v0.2.0)."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents_core import settings
from agents_core.costs import BudgetExceeded, CostTracker
from agents_core.guards import fields_guard
from agents_core.llm import LLM, BatchItem, BatchTimeout, LLMError, tier_config
from tests.conftest import FakeBatches, FakeClient, fake_message


class Brief(BaseModel):
    summary: str


@pytest.fixture
def models_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a repo-level config/models.toml override in a temp CWD."""

    def write(text: str) -> None:
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / "config" / "models.toml").write_text(text)
        settings.load_config.cache_clear()

    monkeypatch.chdir(tmp_path)
    yield write
    settings.load_config.cache_clear()


class PromptClient:
    """Thread-safe fake: answers by prompt, tracks peak concurrency."""

    def __init__(self, delay: float = 0.02, refuse: frozenset[str] = frozenset()):
        self.delay = delay
        self.refuse = refuse
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls: list[dict] = []
        self.messages = self
        self.batches = FakeBatches([], polls_before_end=10**6)

    def _answer(self, params):
        with self.lock:
            self.calls.append(params)
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        prompt = params["messages"][0]["content"]
        if prompt in self.refuse:
            return fake_message("", stop_reason="refusal")
        return fake_message(f"re:{prompt}", parsed=Brief(summary=f"re:{prompt}"))

    def create(self, **params):
        return self._answer(params)

    def parse(self, **params):
        return self._answer(params)


def make_llm(client, max_usd=5.0, **kw):
    tracker = CostTracker(agent="macro", run_id="r1", max_usd=max_usd)
    return LLM(tracker, client=client, sleep=lambda s: None, **kw), tracker


# ---- temperature ----------------------------------------------------------


def test_no_temperature_sent_by_default():
    client = FakeClient([fake_message("x")])
    llm, _ = make_llm(client)
    llm.complete("smart", "p", system="s")
    assert "temperature" not in client.messages.calls[0]
    assert "extra_body" not in client.messages.calls[0]
    assert tier_config("smart").temperature is None


def test_tier_temperature_from_models_toml(models_toml):
    models_toml("[tiers.fast]\ntemperature = 0\n[tiers.smart]\ntemperature = 0.3\n")
    assert tier_config("fast").temperature == 0.0
    client = FakeClient([fake_message("x"), fake_message("y")])
    llm, _ = make_llm(client)
    llm.complete("fast", "p", system="s")
    llm.complete("smart", "p", system="s")
    assert [c["extra_body"]["temperature"] for c in client.messages.calls] == [0.0, 0.3]
    # The shipped defaults are still merged underneath.
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5-20251001"


def test_per_call_temperature_overrides_tier(models_toml):
    models_toml("[tiers.fast]\ntemperature = 0.7\n")
    client = FakeClient([fake_message("x", parsed=Brief(summary="x"))])
    llm, _ = make_llm(client)
    llm.structured("fast", "p", Brief, system="s", temperature=0)
    assert client.messages.calls[0]["extra_body"]["temperature"] == 0


def test_guard_retry_keeps_temperature():
    client = FakeClient([fake_message("12 bids"), fake_message("5 bids")])
    llm, _ = make_llm(client)
    from agents_core.guards import text_guard

    out = llm.complete("fast", "p", system="s", temperature=0.1, guard=text_guard({"n": 5}))
    assert out.value == "5 bids"
    assert [c["extra_body"]["temperature"] for c in client.messages.calls] == [0.1, 0.1]


def test_batch_requests_carry_temperature():
    batches = FakeBatches([], polls_before_end=0)
    llm, _ = make_llm(FakeClient(batches=batches))
    llm.batch("fast", [BatchItem("a", "x")], system="s", temperature=0)
    assert batches.created[0][0]["params"]["temperature"] == 0


# ---- run_many / max_concurrency --------------------------------------------


def test_run_many_is_sequential_by_default():
    client = PromptClient()
    llm, tracker = make_llm(client)
    items = [BatchItem(str(i), f"p{i}") for i in range(4)]
    results = llm.run_many("fast", items, system="s")
    assert client.peak == 1
    assert [r.value for r in results.values()] == ["re:p0", "re:p1", "re:p2", "re:p3"]
    assert all(r.via == "sync" and r.ok for r in results.values())
    assert tracker.calls == 4


def test_run_many_respects_per_call_max_concurrency():
    client = PromptClient(delay=0.05)
    llm, _ = make_llm(client)
    items = [BatchItem(str(i), f"p{i}") for i in range(10)]
    results = llm.run_many("fast", items, system="s", output_model=Brief, max_concurrency=3)
    assert 1 < client.peak <= 3
    assert list(results) == [str(i) for i in range(10)]  # input order kept
    assert results["7"].value == Brief(summary="re:p7")


def test_max_concurrency_from_constructor_and_models_toml(models_toml):
    models_toml("[llm]\nmax_concurrency = 2\n")
    client = PromptClient(delay=0.05)
    llm, _ = make_llm(client)
    llm.run_many("fast", [BatchItem(str(i), f"p{i}") for i in range(6)], system="s")
    assert client.peak == 2

    client = PromptClient(delay=0.05)
    llm, _ = make_llm(client, max_concurrency=4)
    llm.run_many("fast", [BatchItem(str(i), f"p{i}") for i in range(8)], system="s")
    assert 2 < client.peak <= 4


def test_run_many_turns_unusable_output_into_item_error():
    client = PromptClient(refuse=frozenset({"p1"}))
    llm, _ = make_llm(client)
    items = [BatchItem("0", "p0"), BatchItem("1", "p1")]
    results = llm.run_many("fast", items, system="s", max_concurrency=2)
    assert results["0"].ok
    assert "refused" in results["1"].error


def test_run_many_rejects_duplicate_ids():
    llm, _ = make_llm(PromptClient())
    with pytest.raises(ValueError):
        llm.run_many("fast", [BatchItem("a", "x"), BatchItem("a", "y")], system="s")


def test_concurrent_calls_cannot_jointly_overshoot_budget():
    # Each fast call's worst case is ~$0.02 (4096 output tokens at $5/M); with a $0.05
    # cap only two can be reserved at once, so the third must fail before sending.
    client = PromptClient(delay=0.2)
    llm, tracker = make_llm(client, max_usd=0.05)
    items = [BatchItem(str(i), f"p{i}") for i in range(3)]
    with pytest.raises(BudgetExceeded):
        llm.run_many("fast", items, system="s", max_concurrency=3)
    assert len(client.calls) == 2
    assert tracker.total_usd <= tracker.max_usd


def test_cost_tracker_reserve_releases_on_exit():
    tracker = CostTracker(agent="a", run_id="r", max_usd=1.0)
    with tracker.reserve(0.6), pytest.raises(BudgetExceeded):
        tracker.check(0.5)
    tracker.check(0.9)  # released


# ---- batch timeout ---------------------------------------------------------


def test_batch_timeout_raises_batch_timeout_by_default():
    client = PromptClient()
    llm, _ = make_llm(client)
    with pytest.raises(BatchTimeout, match="timed out"):
        llm.batch("fast", [BatchItem("a", "x")], system="s", poll_seconds=10, timeout_seconds=30)
    assert issubclass(BatchTimeout, LLMError)
    assert client.batches.cancelled == ["batch_1"]
    assert client.calls == []


def test_batch_timeout_sync_fallback_runs_concurrently():
    client = PromptClient(delay=0.05)
    llm, tracker = make_llm(client)
    items = [BatchItem(str(i), f"p{i}") for i in range(6)]
    results = llm.batch(
        "fast",
        items,
        system="s",
        output_model=Brief,
        poll_seconds=10,
        timeout_seconds=30,
        on_timeout="sync",
        max_concurrency=5,
        temperature=0,
    )
    assert client.batches.cancelled == ["batch_1"]
    assert 1 < client.peak <= 5
    assert all(r.via == "sync" for r in results.values())
    assert results["5"].value == Brief(summary="re:p5")
    assert all(c["extra_body"]["temperature"] == 0 for c in client.calls)
    assert tracker.calls == 6


def test_guard_batch_runs_retries_concurrently(isolated_paths):
    client = PromptClient(delay=0.05)
    llm, _ = make_llm(client)
    items = [BatchItem(str(i), f"{i} bids") for i in range(4)]
    # Every first answer cites a number not in the facts; the retry answers "re:<prompt>".
    results = {
        i.custom_id: SimpleNamespace(ok=True, value=Brief(summary="99 bids"), error=None)
        for i in items
    }
    guarded = llm.guard_batch(
        "fast",
        items,
        results,
        system="s",
        output_model=Brief,
        guard=lambda cid, v: fields_guard({"n": 2}, ["summary"])(v),
        fallback=lambda cid: Brief(summary="template"),
        max_concurrency=4,
    )
    assert client.peak > 1
    # Only "re:2 bids" cites a supported number (2); the rest fall back.
    assert guarded["2"].narrative_source == "llm" and guarded["2"].attempts == 2
    assert guarded["1"].narrative_source == "template"
    assert list(guarded) == ["0", "1", "2", "3"]
