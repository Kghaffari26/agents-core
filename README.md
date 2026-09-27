# agents-core

Shared framework for a family of scheduled data agents: HTTP caching and rate
limiting, a per-run LLM budget cap, a number guard that keeps LLM narrative
honest to the numbers you computed, tiered/cached Claude calls, a budgeted
tool-use loop, run tracing, an eval harness, a JSON publisher with dated
history, and agent registration via Python entry points.

It ships **no agents of its own** — it's a dependency, not a standalone tool.
Four repos depend on it: `real-estate-agent`, `fed-agent`, `sam-agent`, and
`repo-maintain-agent` (all under `Kghaffari26`).

> **Tag note:** the install line and the reusable workflows' `uses:` lines
> below pin `@v0.3.1`. That tag needs to exist on this repo (`git tag
> v0.3.1 && git push origin v0.3.1`, done by a human, not by an agent) before
> they will resolve — until then, pin to `@v0.2.0` or a commit SHA. Upgrading?
> See [Migrating from v0.2.0](#migrating-from-v020),
> [Migrating from v0.1.0](#migrating-from-v010) and
> [CHANGELOG.md](CHANGELOG.md).

## Why this design

These agents run unattended on a schedule and publish numbers that people
read and act on. Every design choice follows from that:

- **Numbers come from code, never from the model.** `transform` computes every
  published figure in plain Python; the model only writes narrative *about*
  numbers it's given. A model that does arithmetic or recalls a statistic will
  eventually be confidently wrong, and a scheduled job has no one watching it.
- **Guards, not trust.** The number guard checks every number in model-written
  text against the facts you computed (allowing for rounding, K/M/B, %, bp),
  retries once with the offending numbers named, then falls back to a
  deterministic template — and labels which one was published
  (`narrative_source`). The agent loop applies the same guard to its final
  result, and wraps every tool output in untrusted-content delimiters so data
  a tool fetched can't steer the model.
- **Budgets everywhere.** A per-run USD cap (`MAX_RUN_USD`) is checked *before*
  each call with a worst-case estimate, so a call that could overshoot is
  never sent. Agent loops add their own step, USD and wall-clock budgets and
  stop gracefully with a partial result; HTTP hosts get daily request budgets;
  evals get a spend cap. A run that hits a budget fails or degrades — it never
  silently spends.
- **Caching at both ends.** HTTP responses are cached on disk (2xx only, never
  with secrets) so development re-runs are free; prompts are laid out stable-
  first with cache breakpoints on the system prompt, tool list and latest
  message, so repeated and multi-turn calls read from the prompt cache.
- **Contracts, not conventions.** Every published file is a pydantic model with
  an exported JSON Schema (`schema.json`, `trace.schema.json`); the data branch
  has a fixed layout; the shared `meta` block only grows additively. Four repos
  and a website depend on these shapes, so they're versioned and tested here.
- **Tracing by default.** Every run writes a redacted, size-capped `trace.json`
  (nested spans for the run, each LLM call, tool call, HTTP request and guard
  check, with tokens, cost, latency, retries and stop reasons) and a
  `trace_summary` in `manifest-entry.json` — without the agent calling
  anything — so a bad run can be diagnosed from its published output alone.
- **Evals as a gate.** Prompt and agent-code changes are scored against cases
  (exact/numeric/set scorers, trajectory scorers, a calibrated LLM judge), the
  scores are kept in a history with the prompt version, git SHA and model, and a
  reusable workflow fails a PR that regresses them. Recorded trajectories
  replay deterministically, so loop behaviour is unit-testable offline.

## Using agents-core in an agent repo

### Install

```toml
# pyproject.toml
[project]
dependencies = ["agents-core"]

[tool.uv.sources]
agents-core = { git = "https://github.com/Kghaffari26/agents-core", tag = "v0.3.1" }
```

or from the command line:

```bash
uv add "agents-core @ git+https://github.com/Kghaffari26/agents-core@v0.3.1"
```

### Register your agent

Expose a module-level `agents_core.agent.Agent` instance and register it
under the `agents_core.agents` entry-point group in your own
`pyproject.toml`:

```toml
[project.entry-points."agents_core.agents"]
real_estate = "real_estate_agent.agent:AGENT"
```

```python
# real_estate_agent/agent.py
from agents_core.agent import Agent, AgentResult, RunContext
from agents_core.schema import AgentOutput, KeyStat, Source


class RealEstateOutput(AgentOutput):
    median_price: float
    # ... every field your latest.json needs, besides `meta`


class RealEstateAgent(Agent):
    id = "real_estate"
    name = "Real Estate Market Agent"
    route = "/real-estate"
    schema_version = "1.0.0"
    expected_interval_hours = 168  # weekly
    next_run_hint = "Fridays 08:00 PT"
    history_keep = 52  # 52 weekly snapshots, or 90 for a daily agent
    output_model = RealEstateOutput

    def configure_http(self, http):
        # Optional: per-host rate limits and daily request budgets. A budgeted host
        # gets no retries unless you set max_attempts.
        http.set_policy("api.sam.gov", HostPolicy(daily_budget=10))

    def fetch(self, ctx: RunContext):
        # Download source data through ctx.http (cached, rate-limited). No LLM calls.
        return ctx.http.get_json("https://api.example.gov/series")

    def transform(self, ctx: RunContext, raw):
        # Pure Python: compute every number your output publishes. No LLM.
        return {"median_price": raw["median"]}

    def analyze(self, ctx: RunContext, data) -> AgentResult:
        # LLM narrative via ctx.llm, built from numbers `transform` already computed.
        brief = ctx.llm.complete("smart", f"median={data['median_price']}", system="...")
        return AgentResult(
            body={"median_price": data["median_price"], "brief": brief},
            sources=[Source(name="Redfin", url="https://redfin.com/...", retrieved_at=...)],
            headline=f"Median price ${data['median_price']:,.0f}.",
            key_stats=[
                KeyStat(label="Median price", value=data["median_price"], format="currency")
            ],
        )


AGENT = RealEstateAgent()
```

Once installed, `agents-run` (a console script this package provides) can
run it:

```bash
agents-run real_estate                # fetch -> transform -> analyze -> publish
agents-run real_estate --dry-run      # fetch + transform only; no LLM calls, nothing published
agents-run --list                     # show every registered agent
```

`--apply` is passed through to your agent (via `ctx.apply`) for agents that
default to read-only, e.g. a repo-maintenance bot that only labels or
comments when explicitly told to. Any argument `agents-run` doesn't
recognize is forwarded verbatim as `ctx.extra_args`, so your agent can define
flags of its own without this package knowing about them.

### The number guard

Numbers come from data, never from the model: compute every figure in
`transform`, and only pass computed numbers into your prompts. Guard every
LLM call whose output reaches your published narrative — `agents_core.guards`
checks that each number in the text matches a fact you computed, allowing for
the rounding shown and for K/M/B, %, bp, and pp forms.

```python
from agents_core.guards import fields_guard, text_guard

# Plain text (ctx.llm.complete):
result = ctx.llm.complete(
    "smart",
    prompt,
    system="...",
    guard=text_guard(facts),
    fallback=lambda: deterministic_template_text(facts),
)

# Structured output (ctx.llm.structured) — list only the narrative fields;
# numeric fields should be copied from data in code, not guarded as text:
result = ctx.llm.structured(
    "smart",
    prompt,
    Brief,
    system="...",
    guard=fields_guard(facts, ["summary", "bullets"]),
    fallback=lambda: Brief(summary=template_summary(facts), bullets=[]),
)
```

- `facts` is whatever data you put in the prompt — a dict, list, or pydantic
  model; every int/float in it counts, recursively. Terms that look like
  numbers but aren't facts (`"S&P 500"`, a fixed `"2%"` target) go in
  `allow=`.
- A failing output is retried once, with the unsupported numbers named in the
  retry. A second failure calls your `fallback()` and returns its result.
  Every failure is logged to `data/guard_failures.jsonl`, and the retry
  counts toward `MAX_RUN_USD` like any other call.
- Both `complete`/`structured` return `Guarded(value, narrative_source,
  attempts, unsupported)` when a guard is given. Publish `narrative_source`
  (`"llm"` or `"template"`) next to the narrative so a consuming site can
  label template text.
- `narrative_source` has exactly two values. `"llm"` means model-written text
  that passed the guard. **Any deterministic, non-LLM text counts as
  `"template"`** — the guard's fallback, and equally text your agent builds
  itself without calling the model (a brief skipped because the budget or a
  flag said so, a canned "no change this week" line). There is no third value.
- The fallback must be deterministic, built from the same facts, never raise,
  and never call the LLM.

### Sampling and concurrency

A tier can set `temperature` in `config/models.toml`, and every call
(`complete`, `structured`, `converse`, `batch`, `run_many`, `guard_batch`, and
`AgentLoop(temperature=)`) takes a `temperature=` override. Neither is sent
unless set — leave both unset for models that reject sampling parameters
(Sonnet 5 answers 400 "deprecated for this model"). Synchronous calls send it in
the request body through the SDK's `extra_body` (the pinned anthropic SDK has no
`temperature=` keyword), batch requests in each request's params; either way the
API receives the same `"temperature"` field. A fake client in your tests sees
`extra_body={"temperature": ...}`, not a `temperature` keyword.

```toml
[tiers.fast]
temperature = 0      # e.g. for a scoring rubric that must be stable

[llm]
max_concurrency = 5  # synchronous calls made together; default 1 (sequential)
```

`ctx.llm.run_many(tier, items, ...)` runs `BatchItem`s as ordinary synchronous
calls, up to `max_concurrency` at a time (per-call `max_concurrency=`, else
`LLM(max_concurrency=)`, else `[llm] max_concurrency`), and returns the same
`dict[custom_id, BatchResult]` as `batch()` (with `via="sync"`). Concurrent
calls reserve their worst-case cost against `MAX_RUN_USD` before sending, so
they can't jointly overshoot it. `guard_batch` retries use the same limit.

### Bulk work: the Batch API

```python
from agents_core.llm import BatchItem

items = [BatchItem(custom_id=listing.id, prompt=score_prompt(listing)) for listing in listings]
results = ctx.llm.batch("fast", items, system="...", output_model=Score)
```

`ctx.llm.batch` runs many independent prompts through Anthropic's Batch API
(50% of standard price) and blocks until it ends, returning results keyed by
`custom_id`. If it hasn't ended after `timeout_seconds` it's cancelled and
raises `BatchTimeout` (an `LLMError`) — or, with `on_timeout="sync"`, reruns
every item through `run_many` at full price, `max_concurrency` at a time:

```python
results = ctx.llm.batch(
    "fast", items, system="...", output_model=Score, on_timeout="sync", max_concurrency=5
)
if any(r.via == "sync" for r in results.values()):
    ctx.warn("batch timed out; scores ran synchronously")
```

To guard batch results, retry failures synchronously, and fall
back per item:

```python
guarded = ctx.llm.guard_batch(
    "fast",
    items,
    results,
    system="...",
    output_model=Score,
    guard=lambda cid, value: fields_guard(facts_by_id[cid], ["summary"])(value),
    fallback=lambda cid: Score(summary=template_summary(facts_by_id[cid])),
)
```

### Warnings and agent-specific meta fields

`meta.warnings` (a list of strings, always present, usually empty) is where an
"ok with a warning" run says what went wrong. Add to it with `ctx.warn("...")`
anywhere in fetch/transform/analyze, or return `AgentResult(warnings=[...])`;
both are published (result warnings first, duplicates dropped).

For fields of your own inside `meta`, subclass `RunMeta` (every added field
needs a default), use it as your output model's `meta` type, and return the
values in `AgentResult.meta_fields`. The runner merges them into `meta` before
validation, so they're published in `latest.json` and appear in `schema.json`;
an undeclared key fails validation, and `meta_fields` may not override a shared
`RunMeta` field.

```python
from agents_core.schema import AgentOutput, RunMeta


class GrantsMeta(RunMeta):
    sam_budget_exhausted: bool = False
    sam_requests_used: int = 0


class GrantsOutput(AgentOutput):
    meta: GrantsMeta
    ...


# in analyze():
return AgentResult(..., meta_fields={"sam_budget_exhausted": True, "sam_requests_used": 3})
```

`meta.meta_schema_version` versions the shared `meta` block itself (`"1.1.0"`
since agents-core v0.2.0: `warnings` and meta subclasses); your agent's own
`schema_version` still versions its whole `latest.json`.

### Ops alerts

For a problem a human has to fix that shouldn't fail the run (a rejected API
key, an extraction that fell back to last week's data):

```python
ctx.alert("SAM.gov API key rejected", "SAM returned 403 for the opportunities search ...")
```

It opens one GitHub issue per title, labelled `ops-alert`; a later alert with
the same title comments on the open issue instead, and each title alerts at
most once per 7 days (tracked in `data/ops_alerts.json`, and by the open
issue's last update). It's active only when `GITHUB_TOKEN` and
`GITHUB_REPOSITORY` are both set — as in the reusable workflow, where the
calling job must grant `issues: write` — and otherwise just logs. It never
raises; it returns `"created"`, `"commented"`, `"skipped_recent"`,
`"disabled"` or `"failed"`. (`agents_core.alerts.ops_alert` is the same thing
without a `ctx`.)

### HTTP: caching, budgets, large downloads

- Only 2xx responses are cached; `get_json` also never caches a body that
  isn't JSON (an HTML error page served with a 200). Cache keys exclude secret
  values but record which secrets were sent, so a response fetched without an
  API key is never served once one is added.
- `HostPolicy(daily_budget=N)` counts only requests actually sent — not cache
  hits, not connections that never opened — per UTC day. A budgeted host gets
  a single attempt per request unless the policy sets `max_attempts`, so one
  flaky 5xx can't burn a scarce quota; retries, when allowed, stop at the
  budget.
- `ctx.http.download(url, dest)` streams a large file to disk with a
  conditional GET (ETag/Last-Modified kept in `<dest>.meta.json`), bypassing
  the JSON cache. It returns `DownloadResult(path, modified, status, etag,
  last_modified, bytes)`; `modified=False` means a 304 and `dest` still holds
  the previous download.

### The agent loop

For the parts of an agent that genuinely need the model to decide what to look
at next, `agents_core.agent_loop.AgentLoop` runs a tool-use loop on the
Messages API. Tools are plain functions taking one pydantic model (their input
schema); the loop needs a `result_model` and ends when the model calls the
built-in `finish` tool with input that validates as it.

```python
from typing import Literal

from pydantic import BaseModel

from agents_core.agent_loop import AgentLoop, LoopBudget, tool
from agents_core.guards import fields_guard


class SeriesQuery(BaseModel):
    series_id: Literal["CPI_YOY", "UNRATE"]


class Brief(BaseModel):
    summary: str
    series_used: list[str]


def build_loop(llm, data):
    @tool(timeout_seconds=10)
    def get_series(args: SeriesQuery) -> dict:
        """Latest computed value of one series, in percent."""
        return {"series_id": args.series_id, "value": data[args.series_id]}

    return AgentLoop(
        llm,
        tools=[get_series],
        result_model=Brief,
        system="You write a two-sentence US macro brief. Use only numbers the tools return.",
        tier="smart",
        max_tokens=2000,  # per response; sets the worst case each call is checked at
        budget=LoopBudget(max_steps=6, max_usd=0.10, max_seconds=120),
        guard=fields_guard(data, ["summary"]),  # the number guard, on finish
    )


result = build_loop(ctx.llm, data).run("Write today's brief.")
if result.ok:
    brief, source = result.result, result.narrative_source
else:
    ctx.warn(f"brief loop stopped ({result.stop_reason}); using the template")
```

- **Stop conditions** (`result.stop_reason`): `"finished"` (the only `ok`
  outcome); `"end_turn_without_finish"` — the model stopped without calling
  `finish`, treated as a failure; `"max_steps"`, `"max_usd"`, `"max_seconds"`
  or `"run_budget"` (MAX_RUN_USD) — a graceful partial result: `ok=False`,
  `partial=True`, with the tool calls made so far and the model's last text,
  never an exception; `"approval_required"`, `"guard_failed"`, `"refusal"`,
  `"max_tokens"`. `result.require()` returns the result or raises `LoopFailed`.
- **Budgets.** `LoopBudget(max_steps, max_usd, max_seconds)`. `max_usd` is
  enforced through `agents_core.costs.SpendScope`: before every model call the
  worst-case cost (all input uncached, output to `max_tokens`) is checked
  against what's left, so the loop never sends a call that could overshoot.
  The run-wide MAX_RUN_USD still applies on top. Size `max_tokens` to the
  job: with the smart tier's default 8000 output tokens, one call's worst case
  is about $0.08, so a `max_usd` below that stops at `"max_usd"` before the
  first call.
- **Tools.** `@tool` takes the docstring as the description and the argument's
  pydantic model as the input schema; `timeout_seconds` (default 30) bounds each
  call (a timed-out call is abandoned, not killed — keep tools idempotent).
  Exceptions, timeouts and invalid inputs go back to the model as `is_error`
  tool results; raise `ToolError("...")` for a clean message. Outputs over
  `max_tool_output_chars` are truncated.
- **Untrusted content.** Every tool output (and tool error) is wrapped in
  `<untrusted-tool-output tool="...">` delimiters — any delimiter inside the
  output is defused first — and the system prompt tells the model to treat
  what's inside as data, never instructions.
- **Allowlist.** `allowed_tools=[...]` limits which of the loop's tools are
  offered in this run; a call to anything else is refused with an error result.
- **Approval gating.** `@tool(requires_approval=True)` tools are never executed
  by the loop: the call is recorded in `result.pending_actions` as a
  `PendingAction(id, tool, input, step, requested_at)` and the model is told it
  was queued. `on_approval="continue"` (default) carries on;
  `on_approval="stop"` ends the loop with `"approval_required"`. After a human
  approves, `loop.execute_approved(action)` runs it.
- **Number guard.** `guard=` (e.g. `fields_guard(facts, ["summary"])`) checks
  the `finish` result; a failure is sent back once (`guard_retries=1`) with the
  unsupported numbers named, then `fallback()` is used (`narrative_source ==
  "template"`), or the loop stops with `"guard_failed"` if there's no fallback.
  Failures are logged to `data/guard_failures.jsonl` like any other guard.
- **Replay.** `result.trajectory` records every model response. Save one from a
  real run and replay it in tests — no network, same tool calls, same result:

  ```python
  from agents_core.agent_loop import ReplayClient
  from agents_core.costs import CostTracker
  from agents_core.llm import LLM


  # once, from a live run:  result.trajectory.save("tests/fixtures/brief.json")
  def test_brief_replays():
      llm = LLM(
          CostTracker(agent="fed", run_id="test"), client=ReplayClient("tests/fixtures/brief.json")
      )
      result = build_loop(llm, DATA).run("Write today's brief.")
      assert result.ok and result.tools_called() == ["get_series", "get_series"]
  ```

  `ReplayClient(..., strict=True)` (default) also checks that each request
  offers the same tools and carries the same number of messages as when it
  was recorded, so a replay that diverges fails with `ReplayMismatch`.

The loop is built on `ctx.llm.converse(tier, messages, system=..., tools=...)`,
a single budget-checked, cost-logged, traced request in a conversation you
manage, if you need a different control flow.

### Tracing

Every `agents-run` is traced. The runner opens a `run` span with `phase` spans
for fetch/transform/analyze/publish, and `agents_core.llm`, `agents_core.http`
and the agent loop add their own spans automatically:

| span kind | recorded |
|---|---|
| `llm_call` | tier, model, purpose, input/output/cache tokens, usd, estimated_usd, stop_reason |
| `guard` | outcome (`pass`, `pass_after_retry`, `fallback`, `failed`), attempts, unsupported |
| `http` | method, url (secrets redacted), status, retries, from_cache |
| `agent_loop` | steps, tool_calls, pending_actions, stop_reason, usd |
| `tool_call` | tool, input, output preview, is_error, pending_approval |

Every span has `started_at`, `duration_ms`, a `status` and, on failure, the
exception. Add your own with `tracing.span`:

```python
from agents_core import tracing

with tracing.span("custom", "score listings", n=len(listings)) as span:
    kept = score(listings)
    span.set(kept=len(kept))
```

After every non-dry run (failed ones included) the runner writes
`trace.json` to the publish dir and a `trace_summary` (`steps`, `tool_calls`,
`llm_calls`, `total_latency_ms`, `cost_usd`, `guard_retries`) into
`manifest-entry.json`. Before writing, the trace is **redacted** — values under
secret-looking keys (`api_key`, `token`, `authorization`, `password`,
`cookie`, ...), well-known credential formats (Anthropic, GitHub, AWS, Slack
keys, bearer tokens, JWTs, secret query params) and the literal values of
secret-looking environment variables (`*_KEY`, `*_TOKEN`, `*_SECRET`, ...) all
become `***` — and **size-capped** at `AGENTS_CORE_TRACE_MAX_BYTES` (default
256 KB): long strings are shortened first, then the latest spans are dropped
(`truncated`, `dropped_spans`). The summary is computed before truncation, so
it's always exact. `trace.schema.json` (the JSON Schema of `trace.json`,
`agents_core.schema.Trace`) is written alongside it; `python -m
agents_core.export_schemas --trace` writes it by hand.

### Evals

`agents_core.evals` scores an agent's prompts and loops against fixed cases:

```python
# fed_agent/evals.py
from agents_core.evals import (
    EvalSuite,
    LLMJudge,
    forbidden_tools_not_called,
    load_cases,
    max_steps,
    numeric,
    required_tools_called,
    set_overlap,
    stop_reason,
)
from fed_agent.agent import PROMPT_VERSION, build_loop


def run_brief(case, ectx):
    # ectx.llm bills a tracker capped at the suite's spend cap
    return build_loop(ectx.llm, case.input).run("Write today's brief.")


SUITE = EvalSuite(
    name="fed-brief",
    prompt_version=PROMPT_VERSION,
    cases=load_cases("evals/cases/fed-brief.jsonl"),  # {"id", "input", "expected", ...}
    task=run_brief,
    scorers=[
        set_overlap(output="series_used", expected="series_used", threshold=1.0),
        required_tools_called(["get_series"]),
        forbidden_tools_not_called(["post_comment"]),
        max_steps(4),
        stop_reason("finished"),
        LLMJudge("Two sentences, neutral tone, no forecasts, no numbers not in the input."),
    ],
)
```

```bash
uv run agents-evals run fed_agent.evals:SUITE --max-usd 1.00
uv run agents-evals compare --threshold 0.05     # exit 1 on a regression
```

- **Scorers** return a `Score(name, value in [0, 1], passed, detail)`:
  `exact`, `numeric(tolerance=, rel_tolerance=)`, `set_overlap(threshold=)`
  (Jaccard) — each takes `output=`/`expected=` (a dotted path or a callable)
  to pick what to compare; trajectory scorers `required_tools_called`,
  `forbidden_tools_not_called` (a queued approval counts as called),
  `max_steps`, `stop_reason` — these need the task to return a `LoopResult`
  (or `EvalOutput(output, loop=...)`); and `LLMJudge(rubric, tier="fast",
  pass_threshold=0.75, temperature=None, max_tokens=None)` (`None` uses the
  tier's setting; a small `max_tokens` shrinks the judge's worst-case pre-call
  estimate against the spend cap). Any `(case, out, ectx) -> Score` callable with a
  `name` works too.
- **Judge calibration.** `judge.calibrate(llm, [LabeledExample(case, output,
  human_score), ...])` returns agreement, mean absolute error, bias and
  correlation against human labels; check `report.ok()` before trusting a
  judge, and pass `adjust=report.offset_adjust()` (or any `float -> float`)
  to correct a constant bias.
- **Spend cap.** `--max-usd`, `EvalSuite(max_usd=)` or
  `AGENTS_CORE_EVAL_MAX_USD` (default $1.00) caps the whole suite, judge calls
  included. When the next call could pass it, that case and the rest are
  skipped and the report says `budget_exhausted`.
- **Total spend cap.** `agents-evals run a:SUITE b:SUITE --total-max-usd 2.00`
  (or `AGENTS_CORE_EVAL_TOTAL_MAX_USD`, or `run_suites(suites,
  total_max_usd=)`) caps the whole run across suites: each suite gets the
  smaller of its own cap and what the earlier suites left. A suite reached with
  nothing left is still reported (every case skipped, `budget_exhausted`), and
  `compare` doesn't count its empty pass rate as a regression. Unset, only the
  per-suite cap applies.
- **Output.** `run_suite` writes `evals/results/<YYYY-MM-DD>.json` (that day's
  latest report per suite, every case's scores) and appends one line per suite
  to `evals/history.jsonl`: `ts`, `suite`, `prompt_version`, `git_sha`,
  `model`, `scores`, `pass_rate`, `usd`, `n_cases`, `n_scored`,
  `budget_exhausted`. Commit the history file; it's the baseline.
- **Compare.** `agents-evals compare` compares each suite's latest history
  entry with its previous one and prints (or `--markdown FILE` appends) a
  table of score deltas; any score (or `pass_rate`) that drops by more than
  `--threshold` is a regression and the exit code is 1. Without `--suite`,
  only suites whose latest entry has the same git SHA as the file's last line
  are compared — the ones the latest run just wrote.

In CI, `.github/workflows/run-evals.yml` runs this on pull requests (see
[the reusable workflows](#the-reusable-workflows)).

### Cost cap

Every LLM call goes through your run's `CostTracker`, which logs tokens/USD
to `data/costs.jsonl` and raises `BudgetExceeded` — failing the run — the
moment a call's estimated or actual cost would push the run's total spend
past `MAX_RUN_USD` (`AGENTS_CORE_MAX_RUN_USD`, default `$0.50`). Model tiers
and pricing come from `config/models.toml`, deep-merged over the package's
[shipped defaults](src/agents_core/config/models.toml) — your repo's file
only needs to override what it wants to change:

```toml
# config/models.toml — only what differs from the package default
[tiers.smart]
model = "claude-opus-5"
max_tokens = 8000
```

### The data-branch contract

This is the part a consuming website depends on exactly. After a run, the
agent's publish dir (`public-data/`, or `$AGENTS_CORE_PUBLISH_DIR`) holds:

```
public-data/
├── latest.json              # your AgentOutput, plus whatever files you passed as `files=`
│                             # (e.g. metros/<slug>.json, all.json)
├── history/YYYY-MM-DD.json  # one snapshot per run day, trimmed to history_keep
├── manifest-entry.json      # id, name, route, status, last_run_at, last_data_change_at,
│                             # expected_interval_hours, next_run_hint, headline, key_stats,
│                             # run_cost_usd, items_count
├── costs-summary.json       # month, total_usd, runs, daily, all_time_usd
├── schema.json              # latest.json's JSON Schema — written automatically, you
│                             # never have to call agents_core.export_schemas yourself
├── trace.json               # this run's spans (redacted, size-capped) + summary
└── trace.schema.json        # trace.json's JSON Schema
```

`manifest-entry.json` also carries `trace_summary` (`steps`, `tool_calls`,
`llm_calls`, `total_latency_ms`, `cost_usd`, `guard_retries`; `null` in entries
written before v0.3.0). `trace.json` and `trace.schema.json` are written after
every non-dry run, including failed ones, so a failure can be diagnosed from
the data branch. They're reserved names: `AgentResult.files` can't use them.

Every `latest.json` (and history snapshot) starts with the shared `meta` block:
`agent`, `schema_version`, `run_id`, `started_at`, `finished_at`, `status`,
`data_changed`, `cost_usd`, `model_usage`, `sources`, `warnings`,
`meta_schema_version`, plus any agent-specific meta fields declared in a
`RunMeta` subclass.

There's no cross-agent `manifest.json` here — each agent publishes its own
single `manifest-entry.json`, and a website assembling a dashboard across
agents does that merge itself (each agent's `data` branch is a separate,
self-describing unit).

### The reusable workflows

`.github/workflows/run-agent.yml` here is `workflow_call`-only. An agent
repo's own cron workflow calls it:

```yaml
# .github/workflows/agent.yml, in your agent repo
name: Run real_estate agent
on:
  schedule:
    - cron: "0 15 * * 5"   # Fridays 08:00 PT
  workflow_dispatch:
jobs:
  run:
    permissions:
      contents: write   # required
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.3.1
    with:
      agent: real_estate
      max_run_usd: "0.50"
      site_repo: Kghaffari26/agents-hub   # optional; omit to skip the dispatch
    secrets: inherit   # passes through ANTHROPIC_API_KEY, FRED_API_KEY, SAM_API_KEY,
                        # CENSUS_API_KEY, REPO_MAINT_TOKEN, SITE_DISPATCH_TOKEN — any may be unset
```

It:

1. restores your existing `data` branch (if there is one) into `public-data/`,
   so `history/` accumulates, `ctx.previous_latest()` returns the last
   published `latest.json`, and `manifest-entry.json` keeps
   `last_data_change_at` across no-change runs;
2. runs `agents-run <agent> <extra_args>` with `GITHUB_TOKEN` (this run's
   token), `APPLY_CHANGES` (`"true"`/`"false"`, from the `apply_changes`
   input, default false), `AGENTS_CORE_MAX_RUN_USD`, and the secrets above in
   its environment;
3. commits `data/` (local run state — cost log, guard-failure log, ops-alert
   state) back to your default branch (skipped if nothing changed, but still
   committed after a failed run so the failure is recorded);
4. force-pushes `public-data/` — and nothing else from the workspace — to your
   repo's `data` branch as a single orphan commit; that branch's root is
   exactly the data-branch contract above;
5. if `site_repo` and a `SITE_DISPATCH_TOKEN` secret are both set, sends a
   `repository_dispatch` event (`agent-data-updated`) to that repo; otherwise
   skips that step quietly.

**Permissions.** `run-agent.yml` declares no permissions of its own, so it runs
with exactly what the calling job grants: a called workflow can never exceed
its caller's grant, and GitHub refuses to start one that asks for more. The
most any agent needs is `contents: write, issues: write, pull-requests: read,
checks: read`; grant only what yours uses, per calling job (`contents: write`
is always required). That lets a least-privilege caller split read-only and
apply runs into two jobs:

```yaml
jobs:
  report:
    if: vars.APPLY_CHANGES != 'true'
    permissions: { contents: write, issues: read, pull-requests: read }
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.3.1
    with: { agent: repo_maint }
    secrets: inherit
  apply:
    if: vars.APPLY_CHANGES == 'true'
    permissions: { contents: write, issues: write, pull-requests: read }
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.3.1
    with: { agent: repo_maint, apply_changes: true, extra_args: --apply }
    secrets: inherit
```

`extra_args` is split on whitespace and passed as arguments — it's not
evaluated by a shell, so quotes and `$(...)` in it are literal.

`ANTHROPIC_API_KEY` is read by `agents_core.llm`; if it's unset (some cloud
dev environments reserve that name for their own use), it falls back to
`AGENTS_ANTHROPIC_API_KEY`.

`.github/workflows/run-evals.yml` is the evals gate, also `workflow_call`-only.
Call it from a PR workflow filtered to the paths that change behaviour:

```yaml
# .github/workflows/evals.yml, in your agent repo
name: Evals
on:
  pull_request:
    paths: ["prompts/**", "src/**", "evals/**", "config/models.toml"]
jobs:
  evals:
    permissions:
      contents: read
    uses: Kghaffari26/agents-core/.github/workflows/run-evals.yml@v0.3.1
    with:
      eval_command: uv run agents-evals run fed_agent.evals:SUITE
      max_usd: "1.00"               # AGENTS_CORE_EVAL_MAX_USD
      total_max_usd: "2.00"         # optional; AGENTS_CORE_EVAL_TOTAL_MAX_USD
      regression_threshold: "0.05"  # fail if any score drops by more than this
    secrets: inherit
```

It checks the PR's changed files against its `paths` input (space-separated
globs, default `prompts/* src/* evals/* config/*`; a second line of defence
behind the caller's filter) and skips when none match; otherwise it runs
`eval_command` (through `bash -c`, in your checkout — it must append to
`history_path`, default `evals/history.jsonl`, as `agents-evals run` does),
then `agents-evals compare`, which writes a markdown table of score deltas to
the job summary and fails the job on a regression beyond
`regression_threshold`. Like `run-agent.yml` it declares no permissions of its
own; it needs only `contents: read`. The only secret it reads is
`ANTHROPIC_API_KEY` (optional). `max_usd` caps each suite; `total_max_usd`
(default empty: none) caps everything `eval_command` runs through
`agents-evals run`.
## A complete example agent

Everything above in one agent: numbers computed in `transform`, a budgeted,
guarded tool-use loop in `analyze` whose tools only serve those numbers, a
deterministic fallback, warnings, and a published `narrative_source`. Tracing
and `trace.json` come for free. (This exact code runs in this repo's tests.)

```python
# fed_agent/agent.py
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from agents_core import settings
from agents_core.agent import Agent, AgentResult, RunContext
from agents_core.agent_loop import AgentLoop, LoopBudget, tool
from agents_core.guards import fields_guard
from agents_core.schema import AgentOutput, KeyStat, NarrativeSource, Source

PROMPT_VERSION = "2026-09-26"
FRED = "https://api.stlouisfed.org/fred/series/observations"
SYSTEM = (
    "You write a two-sentence brief on US inflation and unemployment for a"
    " general audience. Look up each series you mention with get_series and use"
    " only the numbers it returns. Neutral tone, no forecasts."
)


class SeriesQuery(BaseModel):
    series_id: Literal["CPI_YOY", "UNRATE"]


class Brief(BaseModel):
    summary: str = Field(description="Two sentences")
    series_used: list[str]


class FedOutput(AgentOutput):
    cpi_yoy: float
    unemployment: float
    brief: str
    narrative_source: NarrativeSource


def template_brief(data: dict[str, float]) -> Brief:
    return Brief(
        summary=(
            f"Consumer prices are up {data['CPI_YOY']}% from a year ago."
            f" The unemployment rate is {data['UNRATE']}%."
        ),
        series_used=["CPI_YOY", "UNRATE"],
    )


def build_loop(llm, data: dict[str, float]) -> AgentLoop[Brief]:
    """Shared by the agent and its evals (fed_agent/evals.py)."""

    @tool(timeout_seconds=10)
    def get_series(args: SeriesQuery) -> dict:
        """Latest computed value of one series, in percent."""
        return {"series_id": args.series_id, "value": data[args.series_id]}

    return AgentLoop(
        llm,
        tools=[get_series],
        result_model=Brief,
        system=SYSTEM,
        max_tokens=2000,
        budget=LoopBudget(max_steps=6, max_usd=0.10, max_seconds=120),
        guard=fields_guard(data, ["summary"]),
        fallback=lambda: template_brief(data),
        purpose="brief",
    )


class FedAgent(Agent):
    id = "fed"
    name = "Macro & Fed Agent"
    route = "/fed"
    schema_version = "1.0.0"
    expected_interval_hours = 24
    next_run_hint = "Weekdays 07:00 PT"
    history_keep = 90
    output_model = FedOutput

    def fetch(self, ctx: RunContext) -> dict:
        key = settings.require_env("FRED_API_KEY")
        return {
            sid: ctx.http.get_json(
                FRED,
                params={
                    "series_id": sid,
                    "api_key": key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 13,
                },
            )
            for sid in ("CPIAUCSL", "UNRATE")
        }

    def transform(self, ctx: RunContext, raw: dict) -> dict[str, float]:
        cpi = [float(o["value"]) for o in raw["CPIAUCSL"]["observations"]]
        return {
            "CPI_YOY": round((cpi[0] / cpi[12] - 1) * 100, 1),
            "UNRATE": float(raw["UNRATE"]["observations"][0]["value"]),
        }

    def analyze(self, ctx: RunContext, data: dict[str, float]) -> AgentResult:
        result = build_loop(ctx.llm, data).run("Write today's brief.")
        if result.ok:
            brief, source = result.result, result.narrative_source
        else:
            ctx.warn(f"brief loop stopped ({result.stop_reason}); published the template")
            brief, source = template_brief(data), "template"
        return AgentResult(
            body={
                "cpi_yoy": data["CPI_YOY"],
                "unemployment": data["UNRATE"],
                "brief": brief.summary,
                "narrative_source": source,
            },
            sources=[
                Source(
                    name="FRED", url="https://fred.stlouisfed.org/", retrieved_at=datetime.now(UTC)
                )
            ],
            headline=f"CPI inflation {data['CPI_YOY']}%, unemployment {data['UNRATE']}%.",
            key_stats=[
                KeyStat(label="CPI inflation (YoY)", value=data["CPI_YOY"], format="decimal1"),
                KeyStat(label="Unemployment rate", value=data["UNRATE"], format="decimal1"),
            ],
        )


AGENT = FedAgent()
```

Register it (`fed = "fed_agent.agent:AGENT"` under
`[project.entry-points."agents_core.agents"]`), add the `fed_agent/evals.py`
suite from [Evals](#evals) and a replay test from [the agent
loop](#the-agent-loop), and call `run-agent.yml` on a schedule and
`run-evals.yml` on pull requests.

## Modules

| Module | What it's for |
|---|---|
| `agents_core.agent` | `Agent` (the contract you subclass), `AgentResult`, `RunContext` (incl. `ctx.warn`, `ctx.alert`, `ctx.tracer`). |
| `agents_core.registry` | Entry-point discovery: `discover_agents`, `load`, `load_available`. |
| `agents_core.runner` | `agents-run` / `python -m agents_core.runner` — fetch → transform → analyze → validate → publish → export schema, trace. |
| `agents_core.llm` | The **only** place the Anthropic SDK is imported. Tiered models, per-tier/per-call temperature, prompt caching, structured outputs, the Batch API (with a synchronous timeout fallback), concurrent `run_many`, multi-turn `converse`, the number guard hook, cost logging. |
| `agents_core.agent_loop` | `AgentLoop`, `@tool`, `LoopBudget`, `LoopResult`, `PendingAction`, `Trajectory`, `ReplayClient`, `wrap_untrusted`. |
| `agents_core.tracing` | `Tracer`, `span`, `use`, `current_span`, `redact`, `write_trace`. |
| `agents_core.evals` | `EvalSuite`, `EvalCase`, `run_suite`, scorers, `LLMJudge` + `calibrate`, `compare`; the `agents-evals` command. |
| `agents_core.guards` | The number guard: `verify_numbers`, `collect_numbers`, `text_guard`, `fields_guard`. |
| `agents_core.costs` | `CostTracker`, `BudgetExceeded`, `SpendScope`/`ScopeBudgetExceeded` (sub-budgets), `summarize`, `publish_costs_summary`. |
| `agents_core.http` | Retries with backoff, per-host rate limiting and daily request budgets (sent requests only, UTC day), a 2xx-only on-disk cache that never stores or logs secrets, conditional-GET streaming `download`. |
| `agents_core.alerts` | `ops_alert` (also `ctx.alert`): one deduplicated GitHub issue per alert title, at most once per 7 days. |
| `agents_core.publish` | Atomic JSON writes, dated history with trimming, `write_manifest_entry`/`read_previous_manifest_entry`. |
| `agents_core.export_schemas` | `write_schema`, `write_trace_schema` — called automatically by the runner. |
| `agents_core.schema` | `RunMeta`, `Source`, `Citation`, `AgentOutput`, `KeyStat`, `ManifestEntry`, `TraceSummary`, `Trace`, `TraceSpan`, `CostsSummary`, `Timestamp`. |
| `agents_core.settings` | `data_dir()`/`publish_dir()`/`evals_dir()` and every other configurable path; `config/models.toml` loading. |

None of these assume a particular website's layout or a fixed set of agents.

## Migrating from v0.3.0

v0.3.1 is a bug-fix release; every v0.3.0 call still works, and `latest.json`,
`meta` and the data-branch files are unchanged.

1. **Bump the pin** to `v0.3.1` in `pyproject.toml` and `uv lock`, and in your
   workflows' `uses: ...run-agent.yml@v0.3.1` / `run-evals.yml@v0.3.1`.
2. **`temperature` works on synchronous calls now.** Drop any local shim that
   moved it into `extra_body` (sam-agent's `llm_compat.py`, a judge subclass),
   or keep it — it's a no-op once the parameter already arrives in
   `extra_body`. Tests whose fake client asserted `kwargs["temperature"]` should
   read `kwargs["extra_body"]["temperature"]`.
3. **Optional:** replace a per-repo "pass each suite what's left" wrapper with
   `agents-evals run ... --total-max-usd` or the workflow's `total_max_usd`.

## Migrating from v0.2.0

v0.3.0 is additive: every v0.2.0 call still works and means the same thing, and
`latest.json`/`meta` are unchanged (`meta_schema_version` stays `1.1.0`).

1. **Bump the pin** to `v0.3.0` in `pyproject.toml` and `uv lock`, and in your
   workflow's `uses: ...run-agent.yml@v0.3.0`.
2. **Two new files on the data branch**, `trace.json` and
   `trace.schema.json`, written after every run. A consuming site that lists
   or mirrors every file should expect them; nothing needs to read them.
   `AgentResult.files` can no longer use those two names.
3. **`manifest-entry.json` gains `trace_summary`** (an object; `null` in
   entries written by older versions). A consumer that rejects unknown keys
   needs to accept it.
4. **LLM and HTTP calls are traced automatically.** No code change; if you
   pass secrets somewhere unusual (e.g. inside a prompt), the trace redacts
   known formats and secret-looking env var values, but prompts themselves
   are never recorded.
5. **Optional, recommended:** move hand-rolled tool-use loops to
   `AgentLoop` (budgets, approval gating, replay), wrap your own expensive
   steps in `tracing.span(...)`, and add an eval suite plus the
   `run-evals.yml` PR gate. New console script: `agents-evals`.

## Migrating from v0.1.0

v0.2.0 is backwards compatible for agent code: every v0.1.0 call still works
and means the same thing, and a `latest.json` published by v0.1.0 still
validates. What changes, and what to do:

1. **Bump the pin** to `v0.2.0` in `pyproject.toml` (`tag = "v0.2.0"`, then
   `uv lock`) and in your workflow's `uses: ...run-agent.yml@v0.3.0`.
2. **Grant permissions in the calling job.** `run-agent.yml` no longer
   declares `permissions: contents: write` itself — it inherits the caller's
   grant. Add `permissions: contents: write` (plus `issues: write` etc. if the
   agent needs them) to the job that calls it, or the push steps fail on a
   repo whose default token is read-only.
3. **Drop the history workarounds.** The workflow now restores the `data`
   branch into `public-data/` before each run. You can remove `public-data`
   from `cache_path` (keep any other cache paths you want), and drop
   committed-state fallbacks that existed only because `previous_latest()`
   was always `None` in CI.
4. **Published `meta` gains two fields**: `warnings` (`[]` unless you add
   some) and `meta_schema_version` (`"1.1.0"`). Consumers that generate types
   from `schema.json` get them automatically; consumers that reject unknown
   keys need to accept them. Consider bumping your agent's `schema_version`
   minor version.
5. **Replace workarounds with the new hooks** (optional, recommended):
   - warnings logged into run state → `ctx.warn(...)` / `AgentResult(warnings=...)`;
   - a `RunMeta` subclass fed through a before-validator → keep the subclass,
     drop the validator, return `AgentResult(meta_fields={...})`;
   - a local conditional-GET downloader → `ctx.http.download(url, dest)`
     (same `<dest>.meta.json` sidecar keys: `etag`, `last_modified`, `url`);
   - `ttl_seconds=0` used to dodge cached error pages → no longer needed;
   - a sequential batch-timeout fallback → `batch(..., on_timeout="sync",
     max_concurrency=5)` or `run_many(...)`;
   - an unimplemented "open an issue" step → `ctx.alert(title, body)`.
6. **Budgeted hosts no longer retry** by default. If you relied on retries
   for a host with `daily_budget`, set `HostPolicy(daily_budget=..,
   max_attempts=N)`. The budget day is now UTC (it was local time), and
   requests that never connected no longer count.
7. **`extra_args` is no longer shell-evaluated**: it's split on whitespace.
   Plain flags (`--force-briefs`, `--lookback-days=3`) are unaffected.
8. **New env in the agent step**: `GITHUB_TOKEN`, `REPO_MAINT_TOKEN` (if the
   secret is set) and `APPLY_CHANGES` (`"false"` unless you pass
   `apply_changes: true`). An agent that read `GITHUB_TOKEN` from somewhere
   else should check it doesn't now pick up the workflow token unexpectedly.
9. **`BatchTimeout`** is raised on batch timeout instead of a plain
   `LLMError`; it subclasses `LLMError`, so existing `except LLMError` works.

## Developing this repo

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`tests/test_install_entry_point.py` is a real (not mocked) integration test:
it builds a temp project depending on agents-core via a local path, `uv
sync`s it, and runs a dummy agent through the actual `agents-run` console
script — proving the packaging works, not just the Python API.

## Specs

`docs/specs/` carries historical design specs from an earlier, single-repo
version of this project (a combined site + four agents in one monorepo).
They predate the entry-point-based package this repo is now and aren't the
contract above — kept for reference, not current instructions.
