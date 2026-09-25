"""End-to-end: install agents-core into a fresh temp project (as a local path
dependency) alongside a dummy agent that registers itself under the
`agents_core.agents` entry-point group, run it through the real `agents-run`
console script, and check every file the data-branch contract promises.

This is a real (not mocked) integration test: it shells out to `uv sync` and
`uv run`, proving the packaging works end to end, not just the Python API.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest


def _clean_env() -> dict[str, str]:
    """`os.environ` without the AGENTS_CORE_* overrides the autouse `isolated_paths`
    fixture sets for *this* process — those must not leak into the subprocess under
    test, or it would (wrongly) publish into this test's own tmp_path instead of the
    dummy project's."""
    return {k: v for k, v in os.environ.items() if not k.startswith("AGENTS_CORE_")}


AGENTS_CORE_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

DUMMY_AGENT_PY = """\
from datetime import UTC, datetime

from agents_core.agent import Agent, AgentResult, RunContext
from agents_core.schema import AgentOutput, KeyStat, Source


class DummyOutput(AgentOutput):
    value: float


class DummyAgent(Agent):
    id = "dummy"
    name = "Dummy Agent"
    route = "/dummy"
    schema_version = "1.0.0"
    expected_interval_hours = 24
    next_run_hint = "daily"
    history_keep = 5
    output_model = DummyOutput

    def fetch(self, ctx: RunContext):
        return {"value": 42}

    def transform(self, ctx: RunContext, raw):
        return raw

    def analyze(self, ctx: RunContext, data) -> AgentResult:
        return AgentResult(
            body={"value": data["value"]},
            sources=[
                Source(name="Test", url="https://example.com/", retrieved_at=datetime.now(UTC))
            ],
            headline=f"Value is {data['value']}.",
            key_stats=[KeyStat(label="Value", value=data["value"], format="count")],
            items_count=1,
        )


AGENT = DummyAgent()
"""

PYPROJECT_TOML = """\
[project]
name = "dummy-agent-repo"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["agents-core"]

[project.entry-points."agents_core.agents"]
dummy = "dummy_pkg.agent:AGENT"

[tool.uv.sources]
agents-core = {{ path = "{root}" }}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["dummy_pkg"]
"""


@pytest.fixture
def dummy_project(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")

    project = tmp_path / "dummy-agent-repo"
    pkg_dir = project / "dummy_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "agent.py").write_text(DUMMY_AGENT_PY)
    (project / "pyproject.toml").write_text(PYPROJECT_TOML.format(root=AGENTS_CORE_ROOT.as_posix()))

    sync = subprocess.run(
        [uv, "sync"], cwd=project, capture_output=True, text=True, timeout=300, env=_clean_env()
    )
    assert sync.returncode == 0, f"uv sync failed:\nstdout={sync.stdout}\nstderr={sync.stderr}"
    return uv, project


def test_list_shows_registered_agent(dummy_project):
    uv, project = dummy_project
    result = subprocess.run(
        [uv, "run", "agents-run", "--list"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "dummy -> dummy_pkg.agent:AGENT" in result.stdout


def test_dry_run_makes_no_llm_calls_and_publishes_nothing(dummy_project):
    uv, project = dummy_project
    result = subprocess.run(
        [uv, "run", "agents-run", "dummy", "--dry-run"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not (project / "public-data").exists()
    assert not (project / "data").exists()


def test_real_run_publishes_the_full_data_branch_contract(dummy_project):
    uv, project = dummy_project
    result = subprocess.run(
        [uv, "run", "agents-run", "dummy"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    out = project / "public-data"
    latest = json.loads((out / "latest.json").read_text())
    assert latest["value"] == 42
    assert latest["meta"]["agent"] == "dummy"
    assert latest["meta"]["status"] == "ok"

    history_files = list((out / "history").glob("*.json"))
    assert len(history_files) == 1

    entry = json.loads((out / "manifest-entry.json").read_text())
    assert entry["id"] == "dummy"
    assert entry["headline"] == "Value is 42."
    assert entry["items_count"] == 1

    costs_summary = json.loads((out / "costs-summary.json").read_text())
    assert costs_summary["total_usd"] == 0  # DummyAgent never calls the LLM
    assert costs_summary["runs"] == 1

    schema = json.loads((out / "schema.json").read_text())
    assert schema["required"] == ["meta", "value"]

    assert (project / "data" / "costs.jsonl").is_file()

    # A second run without new data still updates the state files.
    result2 = subprocess.run(
        [uv, "run", "agents-run", "dummy"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert result2.returncode == 0
    entry2 = json.loads((out / "manifest-entry.json").read_text())
    assert entry2["last_run_at"] != entry["last_run_at"]
