# Decisions

One line per judgment call, newest release first.

## v0.2.0

- Gaps were taken from the four agent repos' STATUS.md "agents-core gaps / Needed from agents-core" sections (cloned read-only); every item there is addressed, including sam-agent's minor "budget day should be UTC".
- run-agent.yml declares **no** `permissions:` instead of declaring the maximum on its job: GitHub refuses to start a called workflow whose job requests more than the caller grants ("nested job is requesting X but is only allowed Y"), so a declared maximum would break repo-maintain's `issues: read` report job and any `contents: write`-only caller; inheriting achieves the intended least-privilege split, and the maximum set is documented in the workflow header and README instead.
- `apply_changes` is exported as the literal strings "true"/"false" (how Actions renders a boolean input); agents compare against "true".
- All secrets the workflow reads are declared as optional `workflow_call` secrets, so explicit `secrets:` mappings work as well as `secrets: inherit`.
- The data-branch restore extracts the branch over `public-data/` with `git archive | tar`, so the data branch wins over cached or committed copies, while files that exist only locally are kept.
- The data-branch publish now builds its tree from `public-data/` via a throwaway `GIT_INDEX_FILE` + `commit-tree` (still one parentless commit, force-pushed) instead of `checkout --orphan`, which could sweep untracked workspace files into the branch once `.gitignore` was removed.
- Added `--autostash` to the data/ commit's `pull --rebase`, since a repo that tracks `public-data/` would otherwise fail to push after the restore/run modified it.
- Workflow inputs reach scripts through env vars; `extra_args` is word-split, not shell-evaluated (closes a script-injection path; plain flags behave the same).
- Agent-specific meta uses subclassing (`meta: MyMeta` on the output model) plus `AgentResult.meta_fields`, not a free-form `extra` dict: it's typed, validated (`extra="forbid"` still applies) and shows up field-by-field in schema.json; sam-agent's existing subclass keeps working.
- `meta_fields` may not set shared RunMeta fields (the run fails) so an agent can't spoof `agent`, `cost_usd`, etc.
- "Bump meta schema_version to 1.1.0" is implemented as a new `RunMeta.meta_schema_version` field (default "1.1.0", `schema.META_SCHEMA_VERSION`), because `meta.schema_version` is each agent's own version and must stay agent-owned.
- `ctx.warn()` added alongside `AgentResult.warnings` so fetch/transform can record warnings; result warnings come first, duplicates dropped.
- `temperature` is only sent when set (tier or call), since some models (e.g. claude-sonnet-5) reject sampling parameters; shipped defaults set none.
- `max_concurrency` lives under a new `[llm]` table in models.toml (default 1 = the old sequential behaviour), overridable by `LLM(max_concurrency=)` and per call.
- Batch-timeout fallback is opt-in (`on_timeout="sync"`); the default still raises (now `BatchTimeout`, a subclass of `LLMError`, so existing handlers still match). Results carry `via="sync"` so an agent can warn about the fallback.
- `guard_batch` also runs its per-item work `max_concurrency` at a time (default 1 keeps it sequential).
- CostTracker got a lock and in-flight reservations (`reserve()`), so concurrent calls can't jointly exceed MAX_RUN_USD between check and record; `check()` is unchanged for single-threaded callers.
- Non-2xx responses were already uncached in v0.1.0; the real-estate bug was a cached *200* error body plus a cache key blind to the missing key, so the fix is: `get_json` only caches parseable JSON, and cache keys include which secret params/headers were present (names only). Keys without secrets are unchanged, so existing caches stay valid.
- A request counts toward a daily budget once it was sent: any HTTP response (including 4xx) or a transport error after connecting; connect errors/timeouts and pool timeouts don't count.
- Budgeted hosts default to one attempt (`HostPolicy.max_attempts=None` → 1 if `daily_budget` else `Http.max_attempts`); an explicit `max_attempts` re-enables retries, which still stop at the budget.
- The budget day is now the UTC date (sam-agent gap 6); identical in CI.
- `Http.download` keeps real-estate-agent's sidecar format (`<dest>.meta.json` with `etag`, `last_modified`, `url`) so it can drop its local helper without re-downloading; the stored URL is redacted; partial downloads go to `<dest>.part` and never replace the file.
- Ops alerts dedupe on exact title among open `ops-alert`-labelled issues, plus a local `data/ops_alerts.json` timestamp (committed back as run state) so repeat alerts within 7 days make no API calls; an open issue updated in the last 7 days also suppresses a comment.
- Ops alerts read `GITHUB_TOKEN`/`GITHUB_REPOSITORY` (not REPO_MAINT_TOKEN) so alerts land in the agent's own repo with the workflow token; missing either → logged no-op; any API error → logged, returns "failed", state not recorded so the next run retries.
- Added `pyyaml` as a dev dependency so the workflow test can parse run-agent.yml and execute its restore/publish scripts against a local bare git remote.
- Left `http.USER_AGENT` unchanged to avoid perturbing any upstream allow-listing.
- Pushed to `main` as instructed (also mirrored to the session branch); no tag created — a human tags v0.2.0.
