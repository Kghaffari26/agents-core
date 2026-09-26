# agents-core

Shared framework for a family of scheduled data agents: HTTP caching and rate
limiting, a per-run LLM budget cap, a number guard that keeps LLM narrative
honest to the numbers you computed, tiered/cached Claude calls, a JSON
publisher with dated history, and agent registration via Python entry points.

It ships **no agents of its own** — it's a dependency, not a standalone tool.
Four repos depend on it: `real-estate-agent`, `fed-agent`, `sam-agent`, and
`repo-maintain-agent` (all under `Kghaffari26`).

> **Tag note:** the install line and the reusable workflow's `uses:` line
> below both pin `@v0.2.0`. That tag needs to exist on this repo (`git tag
> v0.2.0 && git push origin v0.2.0`, done by a human, not by an agent) before
> either will resolve — until then, pin to a commit SHA instead. Upgrading
> from v0.1.0? See [Migrating from v0.1.0](#migrating-from-v010) and
> [CHANGELOG.md](CHANGELOG.md).

## Using agents-core in an agent repo

### Install

```toml
# pyproject.toml
[project]
dependencies = ["agents-core"]

[tool.uv.sources]
agents-core = { git = "https://github.com/Kghaffari26/agents-core", tag = "v0.2.0" }
```

or from the command line:

```bash
uv add "agents-core @ git+https://github.com/Kghaffari26/agents-core@v0.2.0"
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
(`complete`, `structured`, `batch`, `run_many`, `guard_batch`) takes a
`temperature=` override. Neither is sent unless set — leave both unset for
models that reject sampling parameters.

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
└── schema.json               # latest.json's JSON Schema — written automatically, you
                              # never have to call agents_core.export_schemas yourself
```

Every `latest.json` (and history snapshot) starts with the shared `meta` block:
`agent`, `schema_version`, `run_id`, `started_at`, `finished_at`, `status`,
`data_changed`, `cost_usd`, `model_usage`, `sources`, `warnings`,
`meta_schema_version`, plus any agent-specific meta fields declared in a
`RunMeta` subclass.

There's no cross-agent `manifest.json` here — each agent publishes its own
single `manifest-entry.json`, and a website assembling a dashboard across
agents does that merge itself (each agent's `data` branch is a separate,
self-describing unit).

### The reusable workflow

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
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.2.0
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
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.2.0
    with: { agent: repo_maint }
    secrets: inherit
  apply:
    if: vars.APPLY_CHANGES == 'true'
    permissions: { contents: write, issues: write, pull-requests: read }
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.2.0
    with: { agent: repo_maint, apply_changes: true, extra_args: --apply }
    secrets: inherit
```

`extra_args` is split on whitespace and passed as arguments — it's not
evaluated by a shell, so quotes and `$(...)` in it are literal.

`ANTHROPIC_API_KEY` is read by `agents_core.llm`; if it's unset (some cloud
dev environments reserve that name for their own use), it falls back to
`AGENTS_ANTHROPIC_API_KEY`.

## Modules

| Module | What it's for |
|---|---|
| `agents_core.agent` | `Agent` (the contract you subclass), `AgentResult`, `RunContext` (incl. `ctx.warn`, `ctx.alert`). |
| `agents_core.registry` | Entry-point discovery: `discover_agents`, `load`, `load_available`. |
| `agents_core.runner` | `agents-run` / `python -m agents_core.runner` — fetch → transform → analyze → validate → publish → export schema. |
| `agents_core.llm` | The **only** place the Anthropic SDK is imported. Tiered models, per-tier/per-call temperature, prompt caching, structured outputs, the Batch API (with a synchronous timeout fallback), concurrent `run_many`, the number guard hook, cost logging. |
| `agents_core.guards` | The number guard: `verify_numbers`, `collect_numbers`, `text_guard`, `fields_guard`. |
| `agents_core.costs` | `CostTracker`, `BudgetExceeded`, `summarize`, `publish_costs_summary`. |
| `agents_core.http` | Retries with backoff, per-host rate limiting and daily request budgets (sent requests only, UTC day), a 2xx-only on-disk cache that never stores or logs secrets, conditional-GET streaming `download`. |
| `agents_core.alerts` | `ops_alert` (also `ctx.alert`): one deduplicated GitHub issue per alert title, at most once per 7 days. |
| `agents_core.publish` | Atomic JSON writes, dated history with trimming, `write_manifest_entry`/`read_previous_manifest_entry`. |
| `agents_core.export_schemas` | `write_schema` — called automatically by the runner. |
| `agents_core.schema` | `RunMeta`, `Source`, `Citation`, `AgentOutput`, `KeyStat`, `ManifestEntry`, `CostsSummary`, `Timestamp`. |
| `agents_core.settings` | `data_dir()`/`publish_dir()` and every other configurable path; `config/models.toml` loading. |

None of these assume a particular website's layout or a fixed set of agents.

## Migrating from v0.1.0

v0.2.0 is backwards compatible for agent code: every v0.1.0 call still works
and means the same thing, and a `latest.json` published by v0.1.0 still
validates. What changes, and what to do:

1. **Bump the pin** to `v0.2.0` in `pyproject.toml` (`tag = "v0.2.0"`, then
   `uv lock`) and in your workflow's `uses: ...run-agent.yml@v0.2.0`.
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
