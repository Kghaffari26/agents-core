# agents-core

A pip-installable shared package for a family of scheduled data agents. It ships
**no agents of its own** — four separate repos depend on it: `real-estate-agent`,
`fed-agent`, `sam-agent`, and `repo-maintain-agent` (all under `Kghaffari26`). See
`README.md` for the full public contract (install line, agent registration, the
number guard, the data-branch contract, the reusable workflow); this file is
working notes for whoever's changing code in *this* repo.

## Layout

```
agents-core/
├── pyproject.toml              # name "agents-core", src layout, hatchling
├── src/agents_core/
│   ├── agent.py                 # Agent base class, AgentResult, RunContext (the agent contract)
│   ├── registry.py              # entry-point discovery (group "agents_core.agents")
│   ├── runner.py                 # agents-run console script: fetch->transform->analyze->publish
│   ├── llm.py                   # the ONLY place the Anthropic SDK is imported
│   ├── agent_loop.py            # budgeted tool-use loop, @tool, approvals, replay (ReplayClient)
│   ├── tracing.py               # nested spans, redaction, size-capped trace.json
│   ├── evals.py                 # eval suites, scorers, LLM judge, history + `agents-evals` CLI
│   ├── guards.py                 # the number guard (verify_numbers, text_guard, fields_guard)
│   ├── http.py                  # retries, rate limiting, request budgets, on-disk cache, download()
│   ├── alerts.py                # ops_alert / ctx.alert: one deduped GitHub issue per title
│   ├── costs.py                  # CostTracker, BudgetExceeded, SpendScope, costs-summary.json
│   ├── publish.py                # atomic JSON writes, dated history, manifest-entry.json
│   ├── export_schemas.py        # schema.json + trace.schema.json, written at publish time
│   ├── schema.py                 # shared pydantic models (RunMeta, ManifestEntry, etc.)
│   ├── settings.py               # data_dir()/publish_dir(), config/models.toml loading
│   └── config/models.toml       # packaged default model tiers + pricing
├── tests/
│   ├── fake_agent.py             # FakeAgent + module-level AGENT, used across most tests
│   ├── loop_helpers.py           # fake tool-use responses + sample tools for loop/eval tests
│   ├── test_readme_example.py    # executes the README's complete example agent end to end
│   ├── test_install_entry_point.py  # real uv sync + agents-run subprocess integration test
│   ├── test_run_agent_workflow.py   # run-agent.yml contract + its git scripts run for real
│   └── test_run_evals_workflow.py   # run-evals.yml contract + its change-detection script
├── CHANGELOG.md                 # one section per tagged version
├── DECISIONS.md                 # one line per judgment call, per release
├── docs/specs/                  # historical, pre-package design specs — not current
└── .github/workflows/
    ├── ci.yml                    # ruff + pytest + actionlint
    ├── run-agent.yml             # workflow_call only; agent repos call this
    └── run-evals.yml             # workflow_call only; agent repos' PR eval gate
```

## Rules

- **`agents_core.llm` is the only place the `anthropic` package is imported.** Never
  import it elsewhere — agents get Claude access only through `ctx.llm`.
- **Numbers come from data, never from the model.** Compute every figure in
  `transform`; `analyze` passes those numbers into prompts and asks for narrative
  only. Guard every call whose output reaches published narrative with `text_guard`/
  `fields_guard` (`agents_core.guards`) — see the README's "number guard" section.
- **This package ships no agents.** Don't add an `agents/` directory or agent-specific
  code here — that lives in the four dependent repos, registered via the
  `agents_core.agents` entry-point group.
- **No cross-agent state.** `data_dir()`/`publish_dir()` are always for *one* agent's
  own run; there's no merged `manifest.json` or shared `schemas/` directory here — an
  agent publishes its own `manifest-entry.json`/`schema.json`, and a consuming site
  does any cross-agent merge itself.
- **Paths are CWD-relative, not repo-relative.** This package is installed into other
  repos' environments; nothing in it may assume it can find its own source tree at
  runtime (no `Path(__file__).parents[...]`-based repo-root resolution outside of
  `importlib.resources` for the packaged `config/models.toml` default).
- **`run-agent.yml` and `run-evals.yml` are `workflow_call`-only.** They must never
  gain a `push`/`pull_request`/`schedule` trigger of their own — they only run when
  another repo's workflow calls them.
- **Neither reusable workflow declares `permissions:`** (neither top-level nor on a
  job). Each inherits the calling job's grant; declaring any makes GitHub refuse to
  start it for callers that grant less (e.g. repo-maintain's `issues: read` report
  job). Document new permission needs in its header/README instead (current
  maximum: `run-agent.yml` `contents: write, issues: write, pull-requests: read,
  checks: read`; `run-evals.yml` `contents: read`).
- **Never interpolate `${{ inputs.* }}` into a `run:` script** in either reusable
  workflow — pass it through `env:` (tests enforce this).
- **The data branch stays a single orphan commit of `public-data/` only**, restored
  into `public-data/` before each run. Don't reintroduce `checkout --orphan` (it
  can sweep untracked workspace files in).
- **Ops alerts, request budgets and tracing must never break a run by themselves**:
  `alerts` never raises; budgets count only requests actually sent, and budgeted
  hosts don't retry unless their `HostPolicy.max_attempts` says so; a failure to
  write `trace.json` is logged, not raised.
- **Instrument new LLM/HTTP/tool code with `tracing.span`** (it's a no-op without
  an active tracer). Anything that starts threads must run the work in
  `contextvars.copy_context()` so spans nest. Never record prompts or raw
  credentials in span attrs; `trace.json` is redacted, but don't rely on it.
- **`AgentLoop` budget stops return, they don't raise.** `max_steps`/`max_usd`/
  `max_seconds`/`run_budget` produce a partial `LoopResult`; keep the worst-case
  pre-call check (`SpendScope.check`) ahead of every `converse` call.
- **`meta` is shared across four repos.** Additive fields only, with defaults, so
  older `latest.json` files still validate; bump `schema.META_SCHEMA_VERSION`
  (minor for additions) whenever `RunMeta` changes.
- Every change to a public module (`agent.py`, `schema.py`, `llm.py`, `guards.py`,
  `registry.py`, `runner.py`, `publish.py`, `costs.py`, `export_schemas.py`,
  `http.py`, `settings.py`, `alerts.py`, `agent_loop.py`, `tracing.py`, `evals.py`) or
  to the data-branch contract is a breaking
  change for four other repos — update `README.md`'s matching section and
  `CHANGELOG.md` in the same change, and keep old callers working where possible
  (add a "Migrating from" note to the README when they can't).
- Log each judgment call as one line in `DECISIONS.md`.

## Commands

```bash
uv sync                                    # install (editable)
uv run pytest                              # tests, incl. the real install/entry-point test,
                                           # run-agent.yml's git scripts against a local remote,
                                           # and the README's example agent
uv run ruff check .                        # lint
uv run ruff format --check .               # format check
/tmp/actionlint .github/workflows/*.yml    # or `actionlint` if on PATH
```

## Releasing

Agent repos and the reusable workflows' doc examples pin a tag (currently `@v0.3.1`).
Bump `pyproject.toml`'s `version` and `agents_core.__version__`, run `uv lock`, add a
`CHANGELOG.md` section, update the README's pins, then a human (not an agent session)
tags and pushes: `git tag v0.3.1 && git push origin v0.3.1` — agent sessions in this
repo should never push a tag themselves.
