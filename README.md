# agents-core

Shared framework for a family of scheduled data agents: HTTP caching and rate
limiting, a per-run LLM budget cap, a number guard that keeps LLM narrative
honest to the numbers you computed, tiered/cached Claude calls, a JSON
publisher with dated history, and agent registration via Python entry points.

It ships **no agents of its own** — it's a dependency, not a standalone tool.
Four repos depend on it: `real-estate-agent`, `fed-agent`, `sam-agent`, and
`repo-maintain-agent` (all under `Kghaffari26`).

> **Tag note:** the install line and the reusable workflow's `uses:` line
> below both pin `@v0.1.0`. That tag needs to exist on this repo (`git tag
> v0.1.0 && git push origin v0.1.0`, done by a human, not by an agent) before
> either will resolve — until then, pin to a commit SHA instead.

## Using agents-core in an agent repo

### Install

```toml
# pyproject.toml
[project]
dependencies = ["agents-core"]

[tool.uv.sources]
agents-core = { git = "https://github.com/Kghaffari26/agents-core", tag = "v0.1.0" }
```

or from the command line:

```bash
uv add "agents-core @ git+https://github.com/Kghaffari26/agents-core@v0.1.0"
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
        # Optional: per-host rate limits and daily request budgets.
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
- The fallback must be deterministic, built from the same facts, never raise,
  and never call the LLM.

### Bulk work: the Batch API

```python
from agents_core.llm import BatchItem

items = [BatchItem(custom_id=listing.id, prompt=score_prompt(listing)) for listing in listings]
results = ctx.llm.batch("fast", items, system="...", output_model=Score)
```

`ctx.llm.batch` runs many independent prompts through Anthropic's Batch API
(50% of standard price) and blocks until it ends, returning results keyed by
`custom_id`. To guard batch results, retry failures synchronously, and fall
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
    uses: Kghaffari26/agents-core/.github/workflows/run-agent.yml@v0.1.0
    with:
      agent: real_estate
      max_run_usd: "0.50"
      site_repo: Kghaffari26/agents-hub   # optional; omit to skip the dispatch
    secrets: inherit   # passes through ANTHROPIC_API_KEY, FRED_API_KEY, SAM_API_KEY,
                        # CENSUS_API_KEY, SITE_DISPATCH_TOKEN — any may be unset
```

It runs `agents-run <agent>`, commits `data/` (local run state — cost log,
guard-failure log) back to your default branch (skipped if nothing changed,
but still committed after a failed run so the failure is recorded), then
force-pushes `public-data/` to your repo's `data` branch as a single orphan
commit — that branch's root is exactly the data-branch contract above. If
`site_repo` and a `SITE_DISPATCH_TOKEN` secret are both set, it sends a
`repository_dispatch` event (`agent-data-updated`) to that repo; otherwise it
skips that step quietly.

`ANTHROPIC_API_KEY` is read by `agents_core.llm`; if it's unset (some cloud
dev environments reserve that name for their own use), it falls back to
`AGENTS_ANTHROPIC_API_KEY`.

## Modules

| Module | What it's for |
|---|---|
| `agents_core.agent` | `Agent` (the contract you subclass), `AgentResult`, `RunContext`. |
| `agents_core.registry` | Entry-point discovery: `discover_agents`, `load`, `load_available`. |
| `agents_core.runner` | `agents-run` / `python -m agents_core.runner` — fetch → transform → analyze → validate → publish → export schema. |
| `agents_core.llm` | The **only** place the Anthropic SDK is imported. Tiered models, prompt caching, structured outputs, the Batch API, the number guard hook, cost logging. |
| `agents_core.guards` | The number guard: `verify_numbers`, `collect_numbers`, `text_guard`, `fields_guard`. |
| `agents_core.costs` | `CostTracker`, `BudgetExceeded`, `summarize`, `publish_costs_summary`. |
| `agents_core.http` | Retries with backoff, per-host rate limiting and daily request budgets, an on-disk cache that never stores or logs secrets. |
| `agents_core.publish` | Atomic JSON writes, dated history with trimming, `write_manifest_entry`/`read_previous_manifest_entry`. |
| `agents_core.export_schemas` | `write_schema` — called automatically by the runner. |
| `agents_core.schema` | `RunMeta`, `Source`, `Citation`, `AgentOutput`, `KeyStat`, `ManifestEntry`, `CostsSummary`, `Timestamp`. |
| `agents_core.settings` | `data_dir()`/`publish_dir()` and every other configurable path; `config/models.toml` loading. |

None of these assume a particular website's layout or a fixed set of agents.

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
