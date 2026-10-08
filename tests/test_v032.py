"""v0.3.2: the items the agent repos reported against v0.3.1 that were deferred —
the guard's no-multiples check, DownloadResult headers, the judge's input selector,
and marking evals run on uncommitted changes."""

import json
import subprocess

import httpx
import pytest

from agents_core import evals
from agents_core.agent_loop import AgentLoop
from agents_core.costs import CostTracker
from agents_core.evals import (
    EvalCase,
    EvalContext,
    EvalOutput,
    EvalSuite,
    JudgeVerdict,
    LabeledExample,
    LLMJudge,
    exact,
    run_suite,
    run_suites,
)
from agents_core.guards import (
    RETRY_INSTRUCTION,
    fields_guard,
    find_derived,
    retry_instruction,
    text_guard,
    verify_numbers,
)
from agents_core.http import Http, parse_link_header
from agents_core.llm import LLM
from agents_core.llm import RETRY_INSTRUCTION as LLM_RETRY_INSTRUCTION
from tests.conftest import FakeClient, fake_message
from tests.loop_helpers import Brief, finish

# ---- number guard: no multiples/ratios (real-estate-agent) ---------------------------


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("Prices are 4.3 times income.", "4.3 times"),
        ("Rents rose 3x.", "3x"),
        ("A 2-fold rise.", "2-fold"),
        ("Two-fold, in fact.", "Two-fold"),
        ("Three times the national rate.", "Three times"),
        ("Sales doubled.", "doubled"),
        ("Twice as fast.", "Twice"),
        ("Half as many listings.", "Half as"),
        ("A 3:1 ratio of buyers.", "3:1"),
        ("A 3-to-1 margin.", "3-to-1"),
        ("A ratio of 2.5 to income.", "ratio of 2.5"),
        ("Up 2× on the year.", "2×"),
    ],
)
def test_find_derived_flags_multiples_and_ratios(text, phrase):
    assert find_derived(text) == [phrase]


@pytest.mark.parametrize(
    "text",
    [
        "At 8:30 a.m. the index was 4.3.",
        "Times were hard.",
        "It rose from 3 to 10.",
        "Version 1.2.3 of a 3x4 grid.",
        "Fell 50% over 2 years.",
        "The price-to-income ratio is 4.3.",
    ],
)
def test_find_derived_ignores_ordinary_text(text):
    assert find_derived(text) == []


def test_a_multiple_passes_by_value_but_fails_with_no_multiples():
    facts = {"price_to_income": 4.3, "median": 412_000}
    text = "The median home costs 4.3 times income."
    assert verify_numbers(text, facts).ok  # 4.3 is a fact, so values alone pass
    result = verify_numbers(text, facts, no_multiples=True)
    assert not result.ok
    assert result.unsupported == result.derived == ["4.3 times"]


def test_no_multiples_combines_with_unsupported_numbers_and_allow():
    result = verify_numbers("Up 9.9%, twice the rate.", [2.0], no_multiples=True)
    assert result.unsupported == ["9.9%", "twice"] and result.derived == ["twice"]
    assert verify_numbers("Up twice.", [], no_multiples=True, allow=["twice"]).ok


def test_retry_instruction_names_numbers_and_multiples():
    plain = verify_numbers("Up 9.9%.", [1.0])
    assert retry_instruction(plain) == RETRY_INSTRUCTION.format(tokens="9.9%")
    mixed = verify_numbers("Up 9.9%, 3x the rate.", [3.0], no_multiples=True)
    msg = retry_instruction(mixed)
    assert "[9.9%]" in msg and "Don't compute multiples or ratios ([3x])" in msg
    only = verify_numbers("3x the rate.", [3.0], no_multiples=True)
    assert retry_instruction(only).startswith("Don't compute multiples")
    assert LLM_RETRY_INSTRUCTION == RETRY_INSTRUCTION  # still importable from llm


def test_guard_factories_take_no_multiples():
    assert not text_guard([4.3], no_multiples=True)("4.3 times").ok
    assert text_guard([4.3])("4.3 times").ok
    guard = fields_guard({"x": 4.3}, ["summary"], no_multiples=True)
    assert guard(Brief(summary="4.3 times income")).derived == ["4.3 times"]


def test_llm_guard_retry_explains_the_multiple():
    llm = LLM(
        CostTracker(agent="a", run_id="r"),
        client=FakeClient([fake_message("4.3 times income."), fake_message("4.3 and 1.0.")]),
    )
    out = llm.complete("smart", "p", system="s", guard=text_guard([4.3, 1.0], no_multiples=True))
    assert out.attempts == 2 and out.value == "4.3 and 1.0."
    retry = llm.client.messages.calls[1]["messages"][-1]["content"]
    assert "Don't compute multiples or ratios ([4.3 times])" in retry


def test_agent_loop_finish_guard_rejects_multiples():
    llm = LLM(
        CostTracker(agent="a", run_id="r"),
        client=FakeClient(
            [finish(summary="Prices are 4.3 times income."), finish("f2", summary="It is 4.3.")]
        ),
    )
    loop = AgentLoop(
        llm,
        tools=[],
        result_model=Brief,
        system="s",
        max_tokens=1000,
        guard=fields_guard({"pti": 4.3}, ["summary"], no_multiples=True),
    )
    result = loop.run("x")
    assert result.ok and result.result.summary == "It is 4.3." and result.guard_attempts == 2
    retry = llm.client.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert "state both figures instead" in retry


# ---- DownloadResult.headers (repo-maintain-agent) -----------------------------------


LINK = (
    '<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=5>; rel="last"'
)


def test_download_returns_response_headers_and_links(tmp_path):
    def handler(request):
        headers = {
            "etag": '"v1"',
            "x-ratelimit-remaining": "4999",
            "link": LINK,
            "set-cookie": "session=secret",
        }
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304, headers={**headers, "x-ratelimit-remaining": "4998"})
        return httpx.Response(200, content=b"[]", headers=headers)

    http = Http(transport=httpx.MockTransport(handler))
    first = http.download("https://api.github.com/x", tmp_path / "x.json")
    assert first.headers["x-ratelimit-remaining"] == "4999"
    assert first.links == {
        "next": "https://api.github.com/x?page=2",
        "last": "https://api.github.com/x?page=5",
    }
    assert "set-cookie" not in first.headers
    second = http.download("https://api.github.com/x", tmp_path / "x.json")
    assert not second.modified and second.headers["x-ratelimit-remaining"] == "4998"
    assert second.links["next"].endswith("page=2")


def test_parse_link_header():
    assert parse_link_header("") == {}
    assert parse_link_header('<a>; rel="prev first", <b>; title="x"; rel=next') == {
        "prev": "a",
        "first": "a",
        "next": "b",
    }


# ---- LLMJudge input= (fed-agent) -------------------------------------------------------


def judge_llm(*scores):
    return LLM(
        CostTracker(agent="t", run_id="r"),
        client=FakeClient(
            [fake_message(parsed=JudgeVerdict(reasoning="r", score=s)) for s in scores]
        ),
    )


def prompt_of(llm, i=0):
    return llm.client.messages.calls[i]["messages"][0]["content"]


CASE = EvalCase(id="c", input={"fixture": "cpi.json", "facts": {"cpi": 3.1}, "noise": 1})


def test_judge_input_selector_narrows_case_input():
    llm = judge_llm(5)
    ctx = EvalContext(llm=llm, costs=llm.tracker, case=CASE)
    LLMJudge("r", input="facts")(CASE, EvalOutput("CPI is 3.1%."), ctx)
    assert '<input>\n{\n  "cpi": 3.1\n}\n</input>' in prompt_of(llm)
    assert "fixture" not in prompt_of(llm)


def test_judge_shows_what_the_task_gave_the_model():
    llm = judge_llm(5, 5)
    ctx = EvalContext(llm=llm, costs=llm.tracker, case=CASE)
    seen = {"facts": {"cpi": 3.2}, "as_of": "2026-09"}
    LLMJudge("r")(CASE, EvalOutput("CPI is 3.2%.", input=seen), ctx)
    assert '"cpi": 3.2' in prompt_of(llm) and "fixture" not in prompt_of(llm)
    LLMJudge("r", input=lambda i: i["facts"])(CASE, EvalOutput("x", input=seen), ctx)
    assert '<input>\n{\n  "cpi": 3.2\n}\n</input>' in prompt_of(llm, 1)


def test_task_can_return_the_input_it_used(isolated_paths):
    def task(case, ectx):
        rebuilt = {"cpi": case.input["facts"]["cpi"] + 0.1}
        return EvalOutput("CPI is 3.2%.", input=rebuilt)

    client = FakeClient([fake_message(parsed=JudgeVerdict(reasoning="ok", score=5))])
    suite = EvalSuite(name="judged", cases=[CASE], task=task, scorers=[LLMJudge("r")])
    report = run_suite(suite, llm_client=client, write=False)
    assert report.pass_rate == 1.0
    assert '"cpi": 3.2' in client.messages.calls[0]["messages"][0]["content"]


def test_calibration_uses_the_example_input():
    llm = judge_llm(5)
    LLMJudge("r").calibrate(
        llm, [LabeledExample(case=CASE, output="o", human_score=1.0, input={"seen": 7})]
    )
    assert '"seen": 7' in prompt_of(llm)


# ---- evals on uncommitted changes (sam-agent) -----------------------------------------


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "agent-repo"
    (root / "prompts").mkdir(parents=True)
    (root / "evals").mkdir()
    (root / "data").mkdir()
    (root / "prompts" / "brief.md").write_text("v1")
    (root / "evals" / "history.jsonl").write_text("")
    (root / "data" / "costs.jsonl").write_text("")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "add", ".")
    git(root, "commit", "-qm", "init")
    monkeypatch.chdir(root)
    monkeypatch.setenv("AGENTS_CORE_EVALS_DIR", "evals")
    monkeypatch.setenv("AGENTS_CORE_DATA_DIR", "data")
    return root


def suite():
    return EvalSuite(
        name="s",
        cases=[EvalCase(id="c", input=1, expected=1)],
        task=lambda case, ectx: case.input,
        scorers=[exact()],
    )


def test_clean_tree_is_not_dirty(repo):
    assert evals.git_dirty() is False
    (repo / "scratch.txt").write_text("untracked files don't count")
    assert evals.git_dirty() is False


def test_modified_or_staged_tracked_file_is_dirty(repo):
    (repo / "prompts" / "brief.md").write_text("v2")
    assert evals.git_dirty() is True
    git(repo, "add", "prompts/brief.md")
    assert evals.git_dirty() is True


def test_eval_and_data_dirs_dont_count(repo):
    (repo / "evals" / "history.jsonl").write_text("{}\n")
    (repo / "data" / "costs.jsonl").write_text("{}\n")
    assert evals.git_dirty() is True
    assert evals.git_dirty(exclude=["evals", "data"]) is False


def test_renamed_file_is_dirty(repo):
    git(repo, "mv", "prompts/brief.md", "prompts/brief2.md")
    assert evals.git_dirty(exclude=["evals"]) is True


def test_env_override_and_no_git(repo, monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTS_CORE_GIT_DIRTY", "true")
    assert evals.git_dirty() is True
    monkeypatch.setenv("AGENTS_CORE_GIT_DIRTY", "false")
    (repo / "prompts" / "brief.md").write_text("v2")
    assert evals.git_dirty() is False
    monkeypatch.delenv("AGENTS_CORE_GIT_DIRTY")
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert evals.git_dirty() is None


def test_run_on_uncommitted_changes_is_recorded_and_flagged(repo):
    (repo / "prompts" / "brief.md").write_text("v2")
    report = run_suite(suite())
    assert report.dirty is True
    entry = json.loads((repo / "evals" / "history.jsonl").read_text().splitlines()[-1])
    assert entry["dirty"] is True
    md = evals.comparisons_markdown(evals.compare())
    assert "uncommitted changes" in md


def test_a_runs_own_output_doesnt_make_the_next_suite_dirty(repo):
    # suite 1 writes evals/ and data/ (both tracked here); suite 2 must still be clean
    first, second = run_suites([suite(), suite()])
    assert first.dirty is False and second.dirty is False
    assert evals.git_dirty() is True  # the results really are uncommitted now
    assert run_suite(suite()).dirty is False  # the eval/data dirs are excluded


def test_old_history_entries_without_dirty_still_compare(tmp_path):
    history = tmp_path / "h.jsonl"
    old = {"suite": "a", "git_sha": "s1", "scores": {"x": 1.0}, "pass_rate": 1.0, "n_scored": 1}
    history.write_text(json.dumps(old) + "\n" + json.dumps({**old, "git_sha": "s2"}) + "\n")
    (c,) = evals.compare(history)
    assert c.regressions == [] and "uncommitted" not in evals.comparisons_markdown([c])
