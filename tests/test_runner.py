import json
import re

import pytest

from agents_core import runner
from agents_core.costs import BudgetExceeded
from tests.conftest import FakeClient, fake_message
from tests.fake_agent import FakeAgent


def publish_root(root):
    return root / "public_data"


def run(agent, **kw):
    kw.setdefault("llm_client", FakeClient([fake_message("Index rose.")]))
    return runner.run(agent, **kw)


def test_successful_run_publishes_everything(isolated_paths):
    agent = FakeAgent()
    assert run(agent) == 0
    root = publish_root(isolated_paths)
    latest = json.loads((root / "latest.json").read_text())
    assert latest["headline_value"] == 4.0
    assert latest["brief"] == "Index rose."
    meta = latest["meta"]
    assert meta["agent"] == "macro" and meta["status"] == "ok" and meta["data_changed"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z-[0-9a-f]{6}", meta["run_id"])
    assert meta["cost_usd"] > 0
    assert meta["model_usage"]["smart"]["output_tokens"] == 200
    assert meta["started_at"].endswith("Z")

    history = list((root / "history").iterdir())
    assert len(history) == 1
    assert json.loads((root / "series" / "us.json").read_text())["slug"] == "us"

    entry = json.loads((root / "manifest-entry.json").read_text())
    assert entry["status"] == "ok" and entry["headline"] == "Index at 4.0."
    assert entry["last_data_change_at"] == meta["finished_at"]
    assert entry["items_count"] == 1

    summary = json.loads((root / "costs-summary.json").read_text())
    assert summary["total_usd"] > 0 and summary["runs"] == 1

    schema = json.loads((root / "schema.json").read_text())
    assert schema["required"] == ["meta", "headline_value", "brief"]


def test_unchanged_data_keeps_previous_change_time(isolated_paths):
    assert run(FakeAgent()) == 0
    entry_path = publish_root(isolated_paths) / "manifest-entry.json"
    first = json.loads(entry_path.read_text())
    assert run(FakeAgent(data_changed=False, use_llm=False), llm_client=FakeClient()) == 0
    second = json.loads(entry_path.read_text())
    assert second["last_data_change_at"] == first["last_data_change_at"]
    assert second["run_cost_usd"] == 0


def test_invalid_output_fails_run_and_keeps_previous_latest(isolated_paths):
    assert run(FakeAgent()) == 0
    latest_path = publish_root(isolated_paths) / "latest.json"
    before = latest_path.read_text()

    bad = FakeAgent(body={"headline_value": "not a number", "brief": "x"})
    assert run(bad) == 1
    assert latest_path.read_text() == before
    entry = json.loads((publish_root(isolated_paths) / "manifest-entry.json").read_text())
    assert entry["status"] == "failed"
    assert entry["headline"] == "Index at 4.0."  # last good headline kept


def test_failure_with_no_previous_run(isolated_paths):
    assert run(FakeAgent(fail_in="fetch")) == 1
    root = publish_root(isolated_paths)
    assert not (root / "latest.json").exists()
    entry = json.loads((root / "manifest-entry.json").read_text())
    assert entry["status"] == "failed" and entry["last_data_change_at"] is None
    # Failed runs still publish a cost summary and a schema, since neither depends on
    # the run having succeeded.
    assert (root / "costs-summary.json").is_file()
    assert (root / "schema.json").is_file()


def test_budget_exceeded_fails_run(isolated_paths, monkeypatch):
    monkeypatch.setenv("AGENTS_CORE_MAX_RUN_USD", "0.001")
    assert run(FakeAgent()) == 1
    assert not (publish_root(isolated_paths) / "latest.json").exists()


def test_dry_run_skips_llm_and_publish(isolated_paths):
    agent = FakeAgent()
    client = FakeClient()
    assert runner.run(agent, dry_run=True, llm_client=client) == 0
    assert agent.calls == ["fetch", "transform"]
    assert client.messages.calls == []
    assert not publish_root(isolated_paths).exists()
    assert not (isolated_paths / "costs.jsonl").exists()


def test_extra_meta_in_body_is_ignored_and_replaced(isolated_paths):
    body = {"headline_value": 1.0, "brief": "b", "meta": {"agent": "spoofed"}}
    assert run(FakeAgent(body=body)) == 0
    latest = json.loads((publish_root(isolated_paths) / "latest.json").read_text())
    assert latest["meta"]["agent"] == "macro"


def test_cli_rejects_unregistered_agent(capsys):
    assert runner.main(["not-a-registered-agent"]) == 2


def test_cli_lists_agents_with_no_positional_arg(monkeypatch, capsys):
    monkeypatch.setattr(runner.registry, "discover_agents", lambda: {"macro": "pkg.agent:AGENT"})
    assert runner.main([]) == 0
    assert "macro -> pkg.agent:AGENT" in capsys.readouterr().out


def test_cli_list_flag(monkeypatch, capsys):
    monkeypatch.setattr(runner.registry, "discover_agents", lambda: {"macro": "pkg.agent:AGENT"})
    assert runner.main(["--list"]) == 0
    assert "macro -> pkg.agent:AGENT" in capsys.readouterr().out


def test_budget_exceeded_is_a_runtime_error():
    assert issubclass(BudgetExceeded, RuntimeError)


def test_history_is_trimmed(isolated_paths):
    h = publish_root(isolated_paths) / "history"
    h.mkdir(parents=True)
    for d in ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]:
        (h / f"{d}.json").write_text("{}")
    assert run(FakeAgent()) == 0
    assert len(list(h.iterdir())) == FakeAgent.history_keep


@pytest.mark.parametrize("flag", [True, False])
def test_apply_flag_reaches_context(flag, isolated_paths):
    seen = {}

    class Probe(FakeAgent):
        def fetch(self, ctx):
            seen["apply"] = ctx.apply
            return super().fetch(ctx)

    run(Probe(), apply=flag)
    assert seen["apply"] is flag


def test_extra_args_reach_context(isolated_paths):
    seen = {}

    class Probe(FakeAgent):
        def fetch(self, ctx):
            seen["extra_args"] = ctx.extra_args
            return super().fetch(ctx)

    run(Probe(), extra_args=["--force-briefs", "x"])
    assert seen["extra_args"] == ["--force-briefs", "x"]


def test_extra_args_default_to_empty_list(isolated_paths):
    seen = {}

    class Probe(FakeAgent):
        def fetch(self, ctx):
            seen["extra_args"] = ctx.extra_args
            return super().fetch(ctx)

    run(Probe())
    assert seen["extra_args"] == []
