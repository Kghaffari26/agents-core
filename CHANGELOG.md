# Changelog

All notable changes to agents-core. Versions are git tags on this repo
(`@vX.Y.Z`); agent repos pin one. See the README's "Migrating from v0.1.0"
section for upgrade steps.

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
