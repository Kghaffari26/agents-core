"""The reusable workflow's contract, plus its restore/publish scripts run for real
against a local bare git remote (no GitHub needed)."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "run-agent.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def steps(wf):
    return {s.get("name", s.get("uses", s.get("run"))): s for s in wf["jobs"]["run"]["steps"]}


def triggers(wf):
    return wf.get("on", wf.get(True))  # PyYAML parses a bare `on:` key as True


def test_workflow_call_only(wf):
    assert set(triggers(wf)) == {"workflow_call"}


def test_declares_no_permissions_so_callers_decide(wf):
    # A called workflow can't exceed its caller's grant, and GitHub refuses to start
    # one whose jobs request more — so the job inherits exactly what the caller grants.
    assert "permissions" not in wf
    assert "permissions" not in wf["jobs"]["run"]
    header = WORKFLOW.read_text().split("\nname:")[0]
    assert "contents: write, issues: write, pull-requests: read, checks: read" in header


def test_apply_changes_input(wf):
    spec = triggers(wf)["workflow_call"]["inputs"]["apply_changes"]
    assert spec == {**spec, "type": "boolean", "default": False, "required": False}


def test_agent_step_env(wf):
    env = steps(wf)["Run ${{ inputs.agent }}"]["env"]
    assert env["GITHUB_TOKEN"] == "${{ github.token }}"
    assert env["REPO_MAINT_TOKEN"] == "${{ secrets.REPO_MAINT_TOKEN }}"
    assert env["APPLY_CHANGES"] == "${{ inputs.apply_changes }}"
    for key in ("ANTHROPIC_API_KEY", "FRED_API_KEY", "SAM_API_KEY", "CENSUS_API_KEY"):
        assert env[key] == f"${{{{ secrets.{key} }}}}"


def test_all_referenced_secrets_are_declared_optional(wf):
    declared = triggers(wf)["workflow_call"]["secrets"]
    text = WORKFLOW.read_text()
    for name in ("ANTHROPIC_API_KEY", "REPO_MAINT_TOKEN", "SITE_DISPATCH_TOKEN"):
        assert f"secrets.{name}" in text
        assert declared[name] == {"required": False}


def test_inputs_are_not_interpolated_into_scripts(wf):
    for step in wf["jobs"]["run"]["steps"]:
        assert "${{ inputs." not in step.get("run", ""), step.get("name")


def test_restore_runs_before_the_agent(wf):
    names = [s.get("name") for s in wf["jobs"]["run"]["steps"]]
    assert names.index("Restore previous data branch") < names.index("Run ${{ inputs.agent }}")


# ---- the restore and publish scripts, executed ------------------------------


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def bash(script, cwd, env):
    subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=cwd,
        check=True,
        env={**os.environ, **env},
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))

    def checkout(name):
        work = tmp_path / name
        git(tmp_path, "clone", "-q", str(remote), str(work))
        git(work, "config", "user.email", "t@example.com")
        git(work, "config", "user.name", "t")
        return work

    first = checkout("seed")
    (first / ".gitignore").write_text("public-data/\n.cache/\n")
    (first / "app.py").write_text("print('agent')\n")
    git(first, "add", ".")
    git(first, "commit", "-qm", "init")
    git(first, "push", "-q", "origin", "HEAD:main")
    return remote, checkout


def test_restore_and_publish_accumulate_history_in_one_orphan_commit(wf, repo):
    remote, checkout = repo
    restore = steps(wf)["Restore previous data branch"]["run"]
    publish = steps(wf)["Publish data branch"]["run"]
    env = {"PUBLISH_DIR": "public-data", "AGENT": "macro"}

    for day, name in [("2026-09-25", "run1"), ("2026-09-26", "run2")]:
        work = checkout(name)  # a fresh CI checkout each run
        bash(restore, work, env)
        pub = work / "public-data"
        (pub / "history").mkdir(parents=True, exist_ok=True)
        (pub / "history" / f"{day}.json").write_text("{}")
        (pub / "latest.json").write_text(f'{{"day": "{day}"}}')
        (work / ".cache").mkdir(exist_ok=True)
        (work / ".cache" / "junk").write_text("local only")
        bash(publish, work, env)
        assert git(work, "status", "--porcelain") == ""  # default-branch checkout untouched

    files = git(remote, "ls-tree", "-r", "--name-only", "data").splitlines()
    assert files == ["history/2026-09-25.json", "history/2026-09-26.json", "latest.json"]
    assert git(remote, "rev-list", "--count", "data") == "1"
    assert '"2026-09-26"' in git(remote, "show", "data:latest.json")


def test_restore_without_data_branch_is_a_no_op(wf, repo):
    _, checkout = repo
    work = checkout("fresh")
    bash(steps(wf)["Restore previous data branch"]["run"], work, {"PUBLISH_DIR": "public-data"})
    assert not (work / "public-data").exists()
