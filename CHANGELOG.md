# Changelog

All notable changes to agents-core. Versions are git tags on this repo
(`@vX.Y.Z`); agent repos pin one. See the README's "Migrating from ..." sections
for upgrade steps.

## v0.3.0 — 2026-09-26

The agentic building blocks: a budgeted tool-use loop, run tracing, and an eval
harness with a reusable PR gate. Additive — every v0.2.0 call still works, and
`latest.json`/`meta` are unchanged. See the README's "Migrating from v0.2.0".

### Agent loop (`agents_core.agent_loop`, new)

- `AgentLoop`: a tool-use loop on the Messages API. Tools are typed functions with a
  pydantic input model (`@tool`), with a per-run allowlist (`allowed_tools`).
- Budgets (`LoopBudget`): `max_steps`, `max_usd` (enforced through the new
  `costs.SpendScope`, with a worst-case check before every call) and `max_seconds`.
  Budget exhaustion — including the run-wide MAX_RUN_USD — returns a graceful
  partial `LoopResult` with a `stop_reason`, never an exception.
- Stop conditions: a required `finish` tool whose input validates as the
  `result_model`; `end_turn` without `finish` is a failure; refusals and truncation
  stop the loop.
- Tool outputs and errors are wrapped in `<untrusted-tool-output>` delimiters
  (delimiters inside are defused) and the system prompt says they're data.
- Per-tool timeouts; exceptions, timeouts and invalid inputs are sent back to the
  model as `is_error` tool results (`ToolError` for clean messages).
- `requires_approval=True` tools are recorded as `PendingAction`s instead of
  executed; the loop continues or stops (`on_approval`); `execute_approved()` runs
  one after a human approves.
- The number guard can check the `finish` result (`guard=`, `fallback=`,
  `guard_retries=`), logged to `guard_failures.jsonl` like other guards.
- Deterministic replay: every response is recorded in `LoopResult.trajectory`;
  `ReplayClient` replays a saved `Trajectory` (strict mode detects divergence).
- `LLM.converse()`: one budget-checked, cost-logged, traced request in a
  caller-managed conversation (cache breakpoints on system, tools and the last
  message), returning a `Turn` with plain-dict content blocks. `LLM.estimate_usd()`.

### Tracing (`agents_core.tracing`, new)

- Nested spans (`run`, `phase`, `agent_loop`, `llm_call`, `tool_call`, `http`,
  `guard`, `custom`) recording tokens, USD, latency, retries, stop reasons and
  guard outcomes. `agents_core.llm`, `agents_core.http` and the agent loop emit
  them automatically; the runner traces every run, with a span per phase.
  `RunContext.tracer` added.
- `trace.json` is written to the publish dir after every non-dry run (failed runs
  included): secrets redacted (secret-looking keys, known credential formats,
  secret query params, secret env var values), size-capped at
  `AGENTS_CORE_TRACE_MAX_BYTES` (default 256 KB).

### Data-branch contract

- New files `trace.json` and `trace.schema.json` (`schema.Trace`; also
  `python -m agents_core.export_schemas --trace`). Both are reserved names for
  `AgentResult.files`.
- `manifest-entry.json` gains `trace_summary` (`steps`, `tool_calls`, `llm_calls`,
  `total_latency_ms`, `cost_usd`, `guard_retries`); `null` in older entries.

### Evals (`agents_core.evals`, new)

- `EvalSuite`/`EvalCase`/`run_suite` with scorers `exact`, `numeric`,
  `set_overlap`, trajectory scorers `required_tools_called`,
  `forbidden_tools_not_called`, `max_steps`, `stop_reason`, and `LLMJudge` (rubric,
  1-5 verdict normalized to 0..1) with `calibrate()` against human labels and an
  `adjust=` hook.
- Spend cap (`--max-usd`, `EvalSuite.max_usd`, `AGENTS_CORE_EVAL_MAX_USD`, default
  $1.00); `evals/results/<date>.json`; `evals/history.jsonl` lines with
  prompt_version, git SHA, model(s), scores, pass rate and cost.
- `agents-evals run module:SUITE` and `agents-evals compare` (markdown deltas,
  exit 1 on regressions beyond `--threshold`). New console script `agents-evals`.

### Reusable workflows

- New `run-evals.yml` (`workflow_call` only, no declared permissions): inputs
  `eval_command`, `max_usd`, `regression_threshold`, `history_path`, `paths`,
  `python_version`; skips PRs that touch none of `paths`, writes the comparison to
  the job summary, fails on regressions.
- `run-agent.yml`: doc pins bumped to `@v0.3.0`; no behaviour change.

### Other

- `costs.SpendScope` / `ScopeBudgetExceeded` (a `BudgetExceeded` subclass): a
  sub-budget on a run's tracker.
- `settings.evals_dir()` (`AGENTS_CORE_EVALS_DIR`, default `evals/`) and
  `settings.eval_max_usd()`.
- LLM calls made on worker threads (`run_many`, `guard_batch`) run in a copy of the
  caller's context, so their spans nest correctly.

## v0.2.0 — 2026-09-26

Fixes the gaps the four agent repos (real-estate-agent, fed-agent, sam-agent,
repo-maintain-agent) reported after wiring in v0.1.0. Backwards compatible for
agent code; the reusable workflow's permission handling changed (see below).

### Reusable workflow (`run-agent.yml`)

- The agent step now gets `GITHUB_TOKEN` (the run's token), `REPO_MAINT_TOKEN`
  (optional secret) and `APPLY_CHANGES` (new boolean input `apply_changes`,
  default false). All secrets are declared as optional `workflow_call` secrets.
- **Permissions are no longer declared by the workflow**: its job inherits
  exactly the caller's grant, so a caller can split read-only and apply runs
  into least-privilege jobs. The maximum an agent needs is documented:
  `contents: write, issues: write, pull-requests: read, checks: read`.
  Callers must now grant at least `contents: write` themselves.
- The existing `data` branch is restored into `public-data/` before the run, so
  `history/` accumulates, `ctx.previous_latest()` works in CI, and
  `last_data_change_at` survives no-change runs.
- The `data` branch commit is built from `public-data/` alone through a
  throwaway index, so untracked workspace files (`.cache/`, a gitignored
  `public-data/` itself) can no longer leak into it. Still a single orphan
  commit, force-pushed.
- `git pull --rebase --autostash` when committing `data/`, so a tracked
  `public-data/` modified by the run no longer blocks the push.
- Inputs are passed to scripts through env vars, never interpolated;
  `extra_args` is split on whitespace rather than shell-evaluated.

### Published `meta` (meta schema 1.1.0)

- `RunMeta.warnings: list[str]` (default empty), filled from
  `AgentResult.warnings` and the new `ctx.warn()`.
- `RunMeta.meta_schema_version` (`"1.1.0"`, `schema.META_SCHEMA_VERSION`).
- Agent-specific meta fields: subclass `RunMeta`, use it as the output model's
  `meta` type, return values in `AgentResult.meta_fields`. Published and in
  `schema.json`.
- `narrative_source` stays `"llm" | "template"`; documented that any
  deterministic, non-LLM text counts as `"template"`.

### LLM

- Optional per-tier `temperature` in `models.toml`, and a `temperature=`
  override on every call. Not sent unless set.
- `LLM.run_many()`: synchronous calls with `max_concurrency` (per call,
  `LLM(max_concurrency=)`, or `[llm] max_concurrency` in `models.toml`;
  default 1). Same result shape as `batch()`, marked `via="sync"`.
- `batch(on_timeout="sync")` cancels a timed-out batch and reruns it through
  `run_many`; the default still raises, now as `BatchTimeout(LLMError)`.
- `guard_batch` retries run `max_concurrency` at a time.
- `CostTracker` is thread-safe and reserves in-flight worst-case cost
  (`CostTracker.reserve`), so concurrent calls can't jointly pass `MAX_RUN_USD`.

### HTTP

- Only 2xx responses are cached; `get_json` never caches a non-JSON body, and
  ignores such a cached entry. Cache keys record which secrets were present
  (never their values), so an error fetched without a key isn't served after
  the key is added.
- Daily request budgets count only requests actually sent (not cache hits, not
  connections that never opened), per UTC day (was local time).
- Budgeted hosts get one attempt by default; `HostPolicy.max_attempts`
  configures retries per host. Retries stop at the budget.
- `Http.download(url, dest)`: streaming conditional-GET download (ETag /
  Last-Modified sidecar), returns `DownloadResult`, reports not-modified.

### Ops alerts

- New `agents_core.alerts.ops_alert` / `ctx.alert(title, body)`: opens or
  comments on one GitHub issue per title (label `ops-alert`), at most once per
  7 days per title; a logged no-op without `GITHUB_TOKEN` and
  `GITHUB_REPOSITORY`; never raises.

### Other

- New settings path: `AGENTS_CORE_OPS_ALERTS_PATH` (default
  `data/ops_alerts.json`).
- Dev dependency: `pyyaml` (workflow contract tests).

## v0.1.0

First release as a pip-installable package with entry-point agent
registration, `agents-run`, the number guard, cost cap, publisher and the
reusable `run-agent.yml`.
