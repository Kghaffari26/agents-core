"""run-evals.yml's contract, plus its change-detection script run for real in a
temporary git repo."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "run-evals.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def triggers(wf):
    return wf.get("on", wf.get(True))


def steps(wf):
    return {s.get("name", s.get("uses", s.get("run"))): s for s in wf["jobs"]["evals"]["steps"]}


def test_workflow_call_only(wf):
    assert set(triggers(wf)) == {"workflow_call"}


def test_declares_no_permissions(wf):
    assert "permissions" not in wf
    assert "permissions" not in wf["jobs"]["evals"]


def test_inputs(wf):
    inputs = triggers(wf)["workflow_call"]["inputs"]
    assert inputs["eval_command"]["required"] is True
    assert inputs["max_usd"]["default"] == "1.00"
    assert inputs["regression_threshold"]["default"] == "0.05"
    assert triggers(wf)["workflow_call"]["secrets"]["ANTHROPIC_API_KEY"] == {"required": False}


def test_inputs_are_not_interpolated_into_scripts(wf):
    for step in wf["jobs"]["evals"]["steps"]:
        assert "${{ inputs." not in step.get("run", ""), step.get("name")


def test_eval_and_compare_steps(wf):
    run = steps(wf)["Run evals"]
    assert run["env"]["AGENTS_CORE_EVAL_MAX_USD"] == "${{ inputs.max_usd }}"
    assert run["env"]["EVAL_COMMAND"] == "${{ inputs.eval_command }}"
    compare = steps(wf)["Compare with previous results"]["run"]
    assert "agents-evals compare" in compare and '--markdown "$GITHUB_STEP_SUMMARY"' in compare
    assert '--threshold "$THRESHOLD"' in compare


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def pr_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("x")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def run_changes(repo, base, wf, *, patterns=None, event="pull_request"):
    script = steps(wf)["Check changed paths"]["run"]
    out = repo.parent / "gh_output"
    summary = repo.parent / "gh_summary"
    out.unlink(missing_ok=True)
    env = {
        **os.environ,
        "EVENT_NAME": event,
        "BASE_SHA": base,
        "PATTERNS": (
            triggers(wf)["workflow_call"]["inputs"]["paths"]["default"]
            if patterns is None
            else patterns
        ),
        "GITHUB_OUTPUT": str(out),
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    subprocess.run(["bash", "-c", script], cwd=repo, env=env, check=True)
    return out.read_text().strip()


def test_docs_only_pr_skips_evals(pr_repo, wf):
    repo, base = pr_repo
    (repo / "README.md").write_text("y")
    git(repo, "commit", "-qam", "docs")
    assert run_changes(repo, base, wf) == "run=false"
    assert "skipping evals" in (repo.parent / "gh_summary").read_text()


def test_prompt_or_code_change_runs_evals(pr_repo, wf):
    repo, base = pr_repo
    (repo / "prompts").mkdir()
    (repo / "prompts" / "brief.md").write_text("new prompt")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "prompt")
    assert run_changes(repo, base, wf) == "run=true"
    assert run_changes(repo, base, wf, patterns="src/*") == "run=false"


def test_non_pr_events_and_empty_patterns_always_run(pr_repo, wf):
    repo, base = pr_repo
    assert run_changes(repo, base, wf, event="workflow_dispatch") == "run=true"
    assert run_changes(repo, base, wf, patterns="") == "run=true"
