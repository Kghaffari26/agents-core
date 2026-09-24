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


# ---- number guard -------------------------------------------------------------

from core.guards import fields_guard, text_guard  # noqa: E402
from core.llm import Guarded, GuardFailed  # noqa: E402

FACTS = {"cpi": 2.9, "payrolls": 142_000}


def guard_log(isolated_paths):
    path = isolated_paths / "guard_failures.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_guard_pass_first_try(isolated_paths):
    llm, client, _ = make_llm([fake_message("CPI held at 2.9%.")])
    out = llm.complete("smart", "p", system="s", guard=text_guard(FACTS))
    assert out == Guarded("CPI held at 2.9%.", "llm", attempts=1)
    assert len(client.messages.calls) == 1
    assert guard_log(isolated_paths) == []


def test_guard_retry_succeeds(isolated_paths):
    llm, client, tracker = make_llm(
        [
            fake_message("CPI rose to 3.1%."),
            fake_message("CPI held at 2.9%."),
        ]
    )
    out = llm.complete("smart", "cpi=2.9", system="s", guard=text_guard(FACTS), purpose="brief")
    assert out.value == "CPI held at 2.9%."
    assert (out.narrative_source, out.attempts) == ("llm", 2)
    retry = client.messages.calls[1]["messages"]
    assert retry[0] == {"role": "user", "content": "cpi=2.9"}
    assert retry[1] == {"role": "assistant", "content": "CPI rose to 3.1%."}
    assert retry[2]["content"] == (
        "These numbers are not in the input: [3.1%]. Rewrite using only numbers provided."
    )
    assert tracker.calls == 2  # the retry is billed and budgeted
    (entry,) = guard_log(isolated_paths)
    assert entry["attempt"] == 1 and entry["unsupported"] == ["3.1%"]
    assert entry["agent"] == "macro" and entry["run_id"] == "r1"
    assert entry["purpose"] == "brief" and len(entry["prompt_hash"]) == 16
    assert entry["output"] == "CPI rose to 3.1%."


def test_guard_falls_back_after_second_failure(isolated_paths):
    llm, _, _ = make_llm([fake_message("CPI 3.1%."), fake_message("CPI 3.2%.")])
    out = llm.complete(
        "smart", "p", system="s", guard=text_guard(FACTS), fallback=lambda: "CPI was 2.9%."
    )
    assert out == Guarded("CPI was 2.9%.", "template", attempts=2, unsupported=["3.2%"])
    assert [e["attempt"] for e in guard_log(isolated_paths)] == [1, 2]


def test_guard_without_fallback_raises():
    llm, _, _ = make_llm([fake_message("CPI 3.1%."), fake_message("CPI 3.2%.")])
    with pytest.raises(GuardFailed):
        llm.complete("smart", "p", system="s", guard=text_guard(FACTS))


def test_guard_refusal_on_retry_falls_back(isolated_paths):
    llm, _, _ = make_llm([fake_message("CPI 3.1%."), fake_message("", stop_reason="refusal")])
    out = llm.complete(
        "smart", "p", system="s", guard=text_guard(FACTS), fallback=lambda: "template"
    )
    assert out.narrative_source == "template"
    second = guard_log(isolated_paths)[1]
    assert second["attempt"] == 2 and second["output"] is None
    assert "refused" in second["error"]


def test_guard_retry_respects_budget():
    # First call fits the cap; the retry's worst-case estimate does not.
    llm, client, _ = make_llm([fake_message("CPI 3.1%.", output_tokens=5000)], max_usd=0.1)
    with pytest.raises(BudgetExceeded):
        llm.complete("smart", "p", system="s", guard=text_guard(FACTS), fallback=lambda: "t")
    assert len(client.messages.calls) == 1


def test_structured_guard_checks_named_fields(isolated_paths):
    bad = Brief(summary="CPI 3.1%", bullets=["ok"])
    good = Brief(summary="CPI 2.9%", bullets=["payrolls +142K"])
    llm, client, _ = make_llm([fake_message("{}", parsed=bad), fake_message("{}", parsed=good)])
    out = llm.structured(
        "smart",
        "p",
        Brief,
        system="s",
        guard=fields_guard(FACTS, ["summary", "bullets"]),
        fallback=lambda: Brief(summary="t", bullets=[]),
    )
    assert out == Guarded(good, "llm", attempts=2)
    assert client.messages.calls[1]["output_format"] is Brief
    assert client.messages.calls[1]["messages"][1]["content"] == bad.model_dump_json()


def test_guard_batch_retries_failures_synchronously(isolated_paths):
    facts = {"a": [5], "b": [7], "c": [9]}
    batches = FakeBatches(
        [
            _batch_entry("a", json.dumps({"summary": "5 bids", "bullets": []})),
            _batch_entry("b", json.dumps({"summary": "8 bids", "bullets": []})),
            _batch_entry("c", error="overloaded"),
        ],
        polls_before_end=0,
    )
    retry_ok = Brief(summary="7 bids", bullets=[])
    llm, client, tracker = make_llm([fake_message("{}", parsed=retry_ok)], batches=batches)
    items = [BatchItem("a", "pa"), BatchItem("b", "pb"), BatchItem("c", "pc")]
    results = llm.batch("fast", items, system="score", output_model=Brief)

    guarded = llm.guard_batch(
        "fast",
        items,
        results,
        system="score",
        output_model=Brief,
        guard=lambda cid, v: fields_guard(facts[cid], ["summary"])(v),
        fallback=lambda cid: Brief(summary=f"template {cid}", bullets=[]),
        purpose="score",
    )
    assert guarded["a"] == Guarded(Brief(summary="5 bids", bullets=[]), "llm", attempts=1)
    assert guarded["b"] == Guarded(retry_ok, "llm", attempts=2)
    assert guarded["c"].narrative_source == "template"
    assert guarded["c"].value.summary == "template c"
    (retry,) = client.messages.calls  # only b was retried, synchronously
    assert retry["messages"][0]["content"] == "pb"
    assert "[8]" in retry["messages"][2]["content"]
    assert guard_log(isolated_paths)[0]["purpose"] == "score:b"
    entries = [json.loads(x) for x in tracker.path.read_text().splitlines()]
    assert [e["batch"] for e in entries] == [True, True, False]
