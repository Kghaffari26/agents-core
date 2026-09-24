import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from core.costs import BudgetExceeded, CostTracker
from core.llm import LLM, BatchItem, LLMError
from tests.conftest import FakeBatches, FakeClient, fake_message


class Brief(BaseModel):
    summary: str
    bullets: list[str]


def make_llm(responses=None, batches=None, max_usd=0.50):
    tracker = CostTracker(agent="macro", run_id="r1", max_usd=max_usd)
    client = FakeClient(responses, batches)
    return LLM(tracker, client=client, sleep=lambda s: None), client, tracker


def test_complete_uses_tier_model_and_caches_stable_prefix():
    llm, client, tracker = make_llm([fake_message("Rates held.")])
    text = llm.complete(
        "smart",
        "numbers: {...}",
        system="You write briefs.",
        context="Glossary...",
        purpose="brief",
    )
    assert text == "Rates held."
    params = client.messages.calls[0]
    assert params["model"] == "claude-sonnet-5"
    assert params["output_config"] == {"effort": "medium"}
    system = params["system"]
    assert [b["text"] for b in system] == ["You write briefs.", "Glossary..."]
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"] == [{"role": "user", "content": "numbers: {...}"}]
    assert tracker.calls == 1
    assert tracker.model_usage().smart.output_tokens == 200


def test_fast_tier_sends_no_effort():
    llm, client, _ = make_llm([fake_message("x")])
    llm.complete("fast", "classify", system="s")
    params = client.messages.calls[0]
    assert params["model"] == "claude-haiku-4-5-20251001"
    assert "output_config" not in params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_unusable_stop_reasons_raise_but_still_bill(stop):
    llm, _, tracker = make_llm([fake_message("partial", stop_reason=stop)])
    with pytest.raises(LLMError):
        llm.complete("smart", "p", system="s")
    assert tracker.calls == 1


def test_structured_returns_parsed_model():
    brief = Brief(summary="s", bullets=["a"])
    llm, client, _ = make_llm([fake_message("{}", parsed=brief)])
    assert llm.structured("smart", "p", Brief, system="s") == brief
    assert client.messages.calls[0]["output_format"] is Brief


def test_structured_without_parsed_output_raises():
    llm, _, _ = make_llm([fake_message("{}", parsed=None)])
    with pytest.raises(LLMError):
        llm.structured("smart", "p", Brief, system="s")


def test_budget_precheck_blocks_call_before_sending():
    llm, client, _ = make_llm([fake_message()], max_usd=0.01)
    with pytest.raises(BudgetExceeded):
        llm.complete("smart", "p", system="s")  # worst case 8000 output tokens = $0.08
    assert client.messages.calls == []


def _batch_entry(cid, text=None, error=None, stop="end_turn"):
    if error:
        return SimpleNamespace(
            custom_id=cid, result=SimpleNamespace(type="errored", error=SimpleNamespace(type=error))
        )
    return SimpleNamespace(
        custom_id=cid,
        result=SimpleNamespace(type="succeeded", message=fake_message(text, stop_reason=stop)),
    )


def test_batch_structured_results_keyed_by_id():
    good = json.dumps({"summary": "fit", "bullets": ["x"]})
    batches = FakeBatches(
        [
            _batch_entry("b", good),
            _batch_entry("a", "not json"),
            _batch_entry("c", error="invalid_request"),
        ],
        polls_before_end=2,
    )
    llm, client, tracker = make_llm(batches=batches)
    items = [BatchItem("a", "p1"), BatchItem("b", "p2"), BatchItem("c", "p3"), BatchItem("d", "p4")]
    results = llm.batch("fast", items, system="score", output_model=Brief, purpose="score")

    assert results["b"].ok and results["b"].value == Brief(summary="fit", bullets=["x"])
    assert "schema mismatch" in results["a"].error
    assert results["c"].error == "invalid_request"
    assert results["d"].error == "missing from results"
    assert tracker.calls == 2  # the two succeeded messages were billed

    requests = batches.created[0]
    fmt = requests[0]["params"]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["type"] == "object"
    entries = [json.loads(x) for x in tracker.path.read_text().splitlines()]
    assert all(e["batch"] for e in entries)


def test_batch_rejects_duplicate_ids():
    llm, _, _ = make_llm()
    with pytest.raises(ValueError):
        llm.batch("fast", [BatchItem("a", "x"), BatchItem("a", "y")], system="s")


def test_batch_times_out_and_cancels():
    batches = FakeBatches([], polls_before_end=10**6)
    llm, _, _ = make_llm(batches=batches)
    with pytest.raises(LLMError, match="timed out"):
        llm.batch("fast", [BatchItem("a", "x")], system="s", poll_seconds=10, timeout_seconds=30)
    assert batches.cancelled == ["batch_1"]


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLM(CostTracker(agent="a", run_id="r"))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.complete("fast", "p", system="s")
