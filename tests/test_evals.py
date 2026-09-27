import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agents_core import evals
from agents_core.agent_loop import AgentLoop, LoopBudget
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
    forbidden_tools_not_called,
    max_steps,
    numeric,
    required_tools_called,
    run_suite,
    set_overlap,
    stop_reason,
)
from agents_core.llm import LLM
from tests.conftest import FakeClient, fake_message
from tests.loop_helpers import Brief, finish, get_series, post_comment, resp, use

CASE = EvalCase(id="c1", input="x", expected={"rate": 3.1, "tags": ["a", "b"], "label": "Up"})


def ctx():
    return EvalContext(
        llm=LLM(CostTracker(agent="t", run_id="r"), client=FakeClient()),
        costs=CostTracker(agent="t", run_id="r"),
        case=CASE,
    )


# ---- output scorers ----------------------------------------------------------------


def test_exact_with_paths_and_normalize():
    out = EvalOutput({"label": "up"})
    s = exact(output="label", expected="label", normalize=str.lower)(CASE, out, ctx())
    assert s.passed and s.value == 1.0 and s.name == "exact"
    s = exact(output="label", expected="label")(CASE, out, ctx())
    assert not s.passed and "'up' != 'Up'" in s.detail


def test_numeric_tolerance():
    close = EvalOutput(SimpleNamespace(rate=3.14))
    assert numeric(tolerance=0.05, output="rate", expected="rate")(CASE, close, ctx()).passed
    assert not numeric(tolerance=0.01, output="rate", expected="rate")(CASE, close, ctx()).passed
    assert numeric(rel_tolerance=0.02, output="rate", expected="rate")(CASE, close, ctx()).passed
    bad = numeric(output="rate", expected="rate")(CASE, EvalOutput({"rate": "n/a"}), ctx())
    assert not bad.passed and "not numeric" in bad.detail


def test_set_overlap_is_jaccard():
    s = set_overlap(threshold=0.5, output="tags", expected="tags")(
        CASE, EvalOutput({"tags": ["a", "c"]}), ctx()
    )
    assert s.value == pytest.approx(1 / 3) and not s.passed
    s = set_overlap(output="tags", expected="tags")(CASE, EvalOutput({"tags": ["b", "a"]}), ctx())
    assert s.value == 1.0 and s.passed


# ---- trajectory scorers -------------------------------------------------------------


def run_loop(responses, **kw):
    llm = LLM(CostTracker(agent="t", run_id="r"), client=FakeClient(responses))
    loop = AgentLoop(
        llm,
        tools=[get_series, post_comment],
        result_model=Brief,
        system="s",
        max_tokens=1000,
        **kw,
    )
    return loop.run("x")


def test_trajectory_scorers():
    result = run_loop(
        [
            resp(use("t1", "get_series", {"series_id": "CPI"})),
            resp(use("t2", "post_comment", {"issue": 1, "body": "b"})),
            finish(summary="s"),
        ]
    )
    out = EvalOutput(result.result, loop=result)
    c = ctx()
    assert required_tools_called(["get_series"])(CASE, out, c).passed
    missing = required_tools_called(["get_series", "search"])(CASE, out, c)
    assert missing.value == 0.5 and not missing.passed
    forbidden = forbidden_tools_not_called(["post_comment"])(CASE, out, c)
    assert not forbidden.passed and "post_comment" in forbidden.detail  # queued counts
    assert max_steps(3)(CASE, out, c).passed and not max_steps(2)(CASE, out, c).passed
    assert stop_reason()(CASE, out, c).passed
    assert stop_reason("max_steps")(CASE, out, c).detail == "finished"
    no_loop = required_tools_called(["x"])(CASE, EvalOutput("text"), c)
    assert not no_loop.passed and "no trajectory" in no_loop.detail


# ---- LLM judge + calibration ---------------------------------------------------------


def judge_llm(scores):
    return LLM(
        CostTracker(agent="t", run_id="r"),
        client=FakeClient(
            [fake_message(parsed=JudgeVerdict(reasoning="r", score=s)) for s in scores]
        ),
    )


def test_llm_judge_scores_against_a_rubric():
    llm = judge_llm([5, 2])
    c = EvalContext(llm=llm, costs=llm.tracker, case=CASE)
    judge = LLMJudge("Is it accurate?", expected="label")
    good = judge(CASE, EvalOutput("Rates rose."), c)
    assert good.value == 1.0 and good.passed and good.detail == "r"
    bad = judge(CASE, EvalOutput("Rates fell."), c)
    assert bad.value == 0.25 and not bad.passed
    prompt = llm.client.messages.calls[0]["messages"][0]["content"]
    assert "Is it accurate?" in prompt and "<output>\nRates rose.\n</output>" in prompt
    assert "<reference>\nUp\n</reference>" in prompt
    assert llm.client.messages.calls[0]["model"] == "claude-haiku-4-5-20251001"


def test_judge_calibration_against_human_labels():
    judge = LLMJudge("rubric")
    examples = [
        LabeledExample(case=CASE, output="a", human_score=1.0),
        LabeledExample(case=CASE, output="b", human_score=0.5),
        LabeledExample(case=CASE, output="c", human_score=0.0),
        LabeledExample(case=CASE, output="d", human_score=0.75),
    ]
    report = judge.calibrate(judge_llm([5, 4, 2, 5]), examples)  # 1.0, .75, .25, 1.0
    assert report.n == 4
    assert report.agreement == 0.75  # "b" is a false pass
    assert report.bias == pytest.approx(0.1875)
    assert report.mean_abs_error == pytest.approx(0.1875)
    assert report.correlation == pytest.approx(0.9661, abs=1e-3)
    assert not report.ok(min_agreement=0.8) and report.ok(min_agreement=0.7)

    adjusted = LLMJudge("rubric", adjust=report.offset_adjust())
    c = ctx()
    c.llm = judge_llm([5])
    assert adjusted(CASE, EvalOutput("x"), c).value == pytest.approx(0.8125)


# ---- suites -------------------------------------------------------------------------


def fixed_now():
    return datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def simple_suite(outputs, **kw):
    cases = [EvalCase(id=f"c{i}", input=i, expected=i) for i in range(len(outputs))]
    return EvalSuite(
        name="demo",
        prompt_version="p1",
        cases=cases,
        task=lambda case, ectx: outputs[case.input],
        scorers=[exact()],
        **kw,
    )


def test_run_suite_writes_results_and_history(isolated_paths, monkeypatch):
    monkeypatch.setenv("AGENTS_CORE_GIT_SHA", "abc123")
    report = run_suite(simple_suite([0, 1, 5]), now=fixed_now)
    assert report.scores == {"exact": pytest.approx(2 / 3, abs=1e-6)}
    assert report.pass_rate == pytest.approx(2 / 3, abs=1e-6)
    assert report.git_sha == "abc123" and report.model == "none"
    assert report.n_cases == report.n_scored == 3 and not report.budget_exhausted

    evals_dir = isolated_paths / "evals"
    day = json.loads((evals_dir / "results" / "2026-09-26.json").read_text())
    assert day["date"] == "2026-09-26"
    assert day["suites"]["demo"]["cases"][2]["scores"][0]["passed"] is False
    (line,) = (evals_dir / "history.jsonl").read_text().splitlines()
    entry = json.loads(line)
    assert entry["suite"] == "demo" and entry["prompt_version"] == "p1"
    assert entry["git_sha"] == "abc123" and entry["model"] == "none"
    assert entry["scores"] == report.scores and entry["ts"] == "2026-09-26T12:00:00Z"

    # a second suite the same day shares the day's results file
    other = simple_suite([0])
    other.name = "other"
    run_suite(other, now=fixed_now)
    day = json.loads((evals_dir / "results" / "2026-09-26.json").read_text())
    assert set(day["suites"]) == {"demo", "other"}


def test_task_and_scorer_errors_score_zero(isolated_paths):
    def task(case, ectx):
        if case.input == 1:
            raise RuntimeError("tool crashed")
        return case.input

    def flaky(case, out, ectx):
        raise ValueError("bad scorer")

    flaky.name = "flaky"
    suite = EvalSuite(
        name="err",
        cases=[EvalCase(id="ok", input=0, expected=0), EvalCase(id="bad", input=1, expected=1)],
        task=task,
        scorers=[exact(), flaky],
    )
    report = run_suite(suite, write=False)
    ok, bad = report.cases
    assert bad.error == "RuntimeError: tool crashed" and [s.value for s in bad.scores] == [0, 0]
    assert ok.scores[1].detail == "scorer error: bad scorer"
    assert report.scores == {"exact": 0.5, "flaky": 0.0} and report.pass_rate == 0.0


def loop_task(case, ectx):
    loop = AgentLoop(
        ectx.llm,
        tools=[get_series],
        result_model=Brief,
        system="s",
        max_tokens=1000,
        budget=LoopBudget(max_steps=4),
    )
    return loop.run(case.input)


def test_loop_tasks_feed_trajectory_scorers_and_record_the_model(isolated_paths):
    responses = [
        resp(use("t1", "get_series", {"series_id": "CPI"})),
        finish(summary="CPI is 3.1.", series_used=["CPI"]),
    ]
    suite = EvalSuite(
        name="loop",
        cases=[EvalCase(id="c", input="brief", expected={"series_used": ["CPI"]})],
        task=loop_task,
        scorers=[
            required_tools_called(["get_series"]),
            stop_reason(),
            set_overlap(output="series_used", expected="series_used"),
        ],
    )
    report = run_suite(suite, llm_client=FakeClient(responses), write=False)
    assert report.pass_rate == 1.0
    assert report.model == "claude-sonnet-5"
    assert report.usd > 0 and report.cases[0].usd == report.usd


def test_spend_cap_skips_remaining_cases(isolated_paths):
    # each case makes one $0.004 call; worst case is ~$0.0015 + $0.08 on max_tokens 8000,
    # so give the calls small max_tokens and a cap that fits exactly two of them
    def task(case, ectx):
        return ectx.llm.complete("smart", "p", system="s", max_tokens=100)

    suite = EvalSuite(
        name="capped",
        cases=[EvalCase(id=f"c{i}", expected="hello") for i in range(5)],
        task=task,
        scorers=[exact()],
    )
    client = FakeClient([fake_message("hello") for _ in range(5)])
    report = run_suite(suite, llm_client=client, max_usd=0.0085, write=False)
    assert report.budget_exhausted
    assert [c.skipped for c in report.cases] == [False, False, True, True, True]
    assert report.n_scored == 2 and report.pass_rate == 1.0
    assert len(client.messages.calls) == 2 and report.usd <= 0.0085


def test_eval_max_usd_env_default(isolated_paths, monkeypatch):
    monkeypatch.setenv("AGENTS_CORE_EVAL_MAX_USD", "0.25")
    assert run_suite(simple_suite([0]), write=False).max_usd == 0.25


def llm_suite(name, n):
    def task(case, ectx):
        return ectx.llm.complete("smart", "p", system="s", max_tokens=100)

    return EvalSuite(
        name=name,
        cases=[EvalCase(id=f"{name}{i}", expected="hello") for i in range(n)],
        task=task,
        scorers=[exact()],
    )


def test_total_cap_spans_suites(isolated_paths):
    # each call costs $0.004 (worst case ~$0.001 up front); $0.0125 fits three calls
    client = FakeClient([fake_message("hello") for _ in range(6)])
    suites = [llm_suite("a", 2), llm_suite("b", 2), llm_suite("c", 2)]
    reports = evals.run_suites(suites, total_max_usd=0.0125, llm_client=client, write=False)
    assert [r.suite for r in reports] == ["a", "b", "c"]
    assert [r.n_scored for r in reports] == [2, 1, 0]
    assert [r.budget_exhausted for r in reports] == [False, True, True]
    assert reports[0].max_usd == 0.0125
    assert reports[1].max_usd == pytest.approx(0.0045)
    assert len(client.messages.calls) == 3
    assert sum(r.usd for r in reports) <= 0.0125


def test_total_cap_never_raises_a_suite_cap(isolated_paths, monkeypatch):
    monkeypatch.setenv("AGENTS_CORE_EVAL_TOTAL_MAX_USD", "5")
    reports = evals.run_suites([simple_suite([0], max_usd=0.3), simple_suite([0])], write=False)
    assert [r.max_usd for r in reports] == [0.3, 1.0]
    monkeypatch.delenv("AGENTS_CORE_EVAL_TOTAL_MAX_USD")
    reports = evals.run_suites([simple_suite([0])], max_usd=2.0, write=False)
    assert reports[0].max_usd == 2.0  # no total cap by default


def test_run_cli_total_max_usd(isolated_paths, monkeypatch, capsys):
    client = FakeClient([fake_message("hello") for _ in range(4)])
    monkeypatch.setattr(
        evals, "LLM", lambda tracker, client=None, _c=client: LLM(tracker, client=_c)
    )
    monkeypatch.setitem(
        sys.modules, "cap_evals", SimpleNamespace(A=llm_suite("a", 2), B=llm_suite("b", 2))
    )
    argv = ["run", "cap_evals:A", "cap_evals:B", "--total-max-usd", "0.0085"]
    assert evals.main(argv) == 0
    out = capsys.readouterr().out
    assert "a: pass_rate=1.000" in out and "b: pass_rate=0.000" in out
    assert "(spend cap reached)" in out.splitlines()[1]
    assert len(client.messages.calls) == 2


def test_llm_judge_temperature_and_max_tokens():
    llm = judge_llm([4])
    c = EvalContext(llm=llm, costs=llm.tracker, case=CASE)
    LLMJudge("ok?", temperature=0, max_tokens=300)(CASE, EvalOutput("x"), c)
    call = llm.client.messages.calls[0]
    assert call["extra_body"] == {"temperature": 0} and call["max_tokens"] == 300


# ---- compare --------------------------------------------------------------------------


def write_history(path, *entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def entry(suite, sha, pass_rate, **scores):
    return {
        "suite": suite,
        "git_sha": sha,
        "prompt_version": sha,
        "model": "m",
        "usd": 0.01,
        "scores": scores,
        "pass_rate": pass_rate,
        "n_cases": 4,
        "n_scored": 4,
        "budget_exhausted": False,
    }


def test_compare_flags_drops_beyond_the_threshold(tmp_path):
    history = tmp_path / "history.jsonl"
    write_history(
        history,
        entry("a", "s1", 1.0, exact=1.0, judge=0.8),
        entry("b", "s1", 1.0, exact=1.0),
        entry("a", "s2", 0.75, exact=0.96, judge=0.7),
    )
    (c,) = evals.compare(history, threshold=0.05)  # only suite "a" ran at s2
    assert c.suite == "a" and c.previous["git_sha"] == "s1"
    by_name = {d.name: d for d in c.deltas}
    assert by_name["exact"].delta == pytest.approx(-0.04) and not by_name["exact"].regression
    assert by_name["judge"].regression and by_name["pass_rate"].regression
    assert [d.name for d in c.regressions] == ["judge", "pass_rate"]

    md = evals.comparisons_markdown([c])
    assert "| judge | 0.800 | 0.700 | -0.100 | ❌ regression |" in md
    assert "**2 regressions.**" in md


def test_compare_first_run_has_no_baseline(tmp_path):
    history = tmp_path / "history.jsonl"
    write_history(history, entry("a", "s1", 1.0, exact=1.0))
    (c,) = evals.compare(history)
    assert c.previous is None and c.regressions == []
    assert "No previous entry" in evals.comparisons_markdown([c])
    assert evals.compare(tmp_path / "missing.jsonl") == []


def test_compare_ignores_pass_rate_of_a_suite_that_never_ran(tmp_path):
    history = tmp_path / "history.jsonl"
    starved = {**entry("a", "s2", 0.0), "n_scored": 0, "budget_exhausted": True}
    write_history(history, entry("a", "s1", 1.0, exact=1.0), starved)
    (c,) = evals.compare(history)
    assert c.regressions == []
    assert "Spend cap reached: only 0/4" in evals.comparisons_markdown([c])


def test_compare_cli_exit_code_and_markdown_summary(tmp_path, capsys):
    history = tmp_path / "history.jsonl"
    summary = tmp_path / "summary.md"
    write_history(history, entry("a", "s1", 1.0, exact=1.0), entry("a", "s2", 1.0, exact=1.0))
    assert evals.main(["compare", "--history", str(history), "--markdown", str(summary)]) == 0
    write_history(history, entry("a", "s1", 1.0, exact=1.0), entry("a", "s2", 0.5, exact=0.5))
    assert evals.main(["compare", "--history", str(history), "--threshold", "0.6"]) == 0
    assert evals.main(["compare", "--history", str(history), "--markdown", str(summary)]) == 1
    text = summary.read_text()
    assert text.count("## Evals") == 2 and "❌ regression" in text


SUITE = simple_suite([0, 1])


def test_run_cli_loads_a_suite_by_module_path(isolated_paths, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "my_evals", SimpleNamespace(SUITE=SUITE))
    assert evals.main(["run", "my_evals:SUITE"]) == 0
    assert "demo: pass_rate=1.000 exact=1.000" in capsys.readouterr().out
    assert (isolated_paths / "evals" / "history.jsonl").is_file()


def test_load_cases_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "input": {"q": 1}, "expected": 2}\n\n{"id": "b"}\n')
    cases = evals.load_cases(path)
    assert [c.id for c in cases] == ["a", "b"] and cases[0].input == {"q": 1}
