# Decisions

One line per judgment call, newest release first.

## v0.3.0

- Built on v0.2.0 as released (main == tag v0.2.0 == 50cf3f9); everything is additive, so `meta_schema_version` stays 1.1.0 (RunMeta is unchanged) and old manifest entries still validate (`trace_summary` defaults to null).
- The loop is a manual loop over a new `LLM.converse()` rather than the SDK's beta tool runner, so the Anthropic import stays in llm.py and the loop can enforce budgets, approvals and replay between turns.
- `max_usd` is enforced by a new `costs.SpendScope` (a sub-budget measured as the run tracker's spend since the loop started) whose `check()` runs with the same worst-case estimate `LLM` already uses, before `tracker.reserve()`; a scope stop raises `ScopeBudgetExceeded(BudgetExceeded)` internally so the loop can tell it apart from MAX_RUN_USD.
- Hitting MAX_RUN_USD inside a loop is also a graceful stop (`stop_reason="run_budget"`) rather than a run failure, since the spec asks for partial results on budget exhaustion; nothing over the cap is ever sent because the check is pre-call.
- `end_turn` without `finish` is a hard failure with no nudge/retry, as specified; `pause_turn` is resent as-is; `refusal` and `max_tokens` stop the loop (a truncated turn may hold a truncated tool call).
- When `finish` appears in a turn, tool calls before it in the same turn run and calls after it are dropped; `finish` with invalid input or a failing guard is answered with an `is_error` tool_result so the model can retry within the step budget.
- Tool timeouts run each call on a daemon thread and abandon it on timeout (Python can't kill threads); documented that tools should be idempotent reads. The per-call timeout is also clamped to the loop's remaining wall-clock budget.
- Untrusted delimiters are a fixed tag (`<untrusted-tool-output tool="...">`) with any occurrence of the tag inside the output defused, plus a system-prompt notice; tool *errors* are wrapped too (exception text can carry fetched content), the loop's own messages (validation, allowlist, approval) are not.
- Approval-gated calls get a non-error tool_result telling the model the action was queued; `PendingAction.id` is the tool_use id; `execute_approved()` re-validates input and is traced with `approved=True`.
- A tool not on the per-run allowlist is neither offered nor executable (an error result if the model calls it anyway); `finish` is reserved and always offered.
- Tool definitions are sent sorted by name and cache breakpoints go on system/context, the last tool and the last message (3 of the 4 allowed), so each turn reads the previous ones from cache.
- `tool_choice` is left at the API default (auto): forced tool choice returns 400 on newer models, and the finish requirement is carried by the system prompt instead.
- Replay: the loop records each response (content blocks as plain dicts, stop reason, usage, and what the request offered); `ReplayClient(strict=True)` checks tool names and message count per request instead of whole-request equality, so replays survive prompt wording changes but catch control-flow divergence.
- Trace nesting is by `parent_id` in a flat span list (not nested JSON), which keeps the JSON Schema non-recursive and lets the size cap drop whole spans.
- Tracing uses `contextvars` (no global tracer); `LLM._map` and tool threads run in `copy_context()` so spans made on worker threads nest under the caller's span. `tracing.span()` is a no-op without an active tracer, so library use outside the runner is unaffected.
- The first guarded call's `llm_call` span precedes its `guard` span; only the retry nests inside the guard (the guard can't start before there's output to check).
- Batch calls get one `llm_call` span with summed tokens/USD (not one per item); SDK-internal retries of Anthropic calls aren't visible to us, so `retries` is recorded on `http` spans only.
- Prompts and completions are never recorded in spans (size and leakage); tool inputs and a 500-char output preview are.
- Redaction is key-based (a key *ending* in api_key/token/secret/password/authorization/cookie/..., so `input_tokens` survives), pattern-based (sk-ant-, sk-, gh*_, github_pat_, AKIA, xox*, JWTs, bearer/basic tokens, secret query params) and value-based (literal values ≥8 chars of env vars named *KEY/*TOKEN/*SECRET/*PASSWORD/*CREDENTIAL).
- Size cap default 256 KB (`AGENTS_CORE_TRACE_MAX_BYTES`): shorten strings to 2000/500/120 chars first, then keep the largest prefix of spans that fits; the summary is computed before any truncation so `trace_summary` is exact.
- `trace.json` is written after failed runs too (not after dry runs, which publish nothing); `trace.schema.json` is published beside it rather than folded into `schema.json`, which stays the agent's own `latest.json` schema.
- `trace_summary` has exactly the six requested fields; `steps` counts agent-loop model turns, `total_latency_ms` sums root spans, `guard_retries` sums `attempts - 1` over guard spans.
- Runner phases (fetch/transform/analyze/publish) got their own `phase` spans, an extra span kind beyond the five requested, so latency is attributable.
- Evals results are one file per UTC date holding that day's latest report per suite (`{"date", "suites": {name: report}}`), so several suites (or reruns) on one day don't overwrite each other.
- `compare` without `--suite` only compares suites whose latest entry shares the git SHA of the history's last line, so an old regression in a suite that wasn't re-run isn't re-reported; `pass_rate` is compared as a score too; a missing baseline is not a regression.
- A spend-cap-truncated eval run is flagged in the report, history and markdown but not failed by `compare` on its own; its partial scores still count toward regressions.
- The eval spend cap uses its own `CostTracker` (`data/eval_costs.jsonl`) with `max_usd` = the cap, so judge calls count too and the existing pre-call reserve prevents overshoot.
- LLM judge: 1-5 verdict via structured output on the `fast` tier, normalized to (score-1)/4; calibration reports agreement at the pass threshold, MAE, bias and Pearson correlation, and `offset_adjust()` is the provided correction hook.
- Scorers are small classes named in lower case (`exact(...)`, `numeric(...)`) so suites read like function calls; any `(case, out, ctx) -> Score` callable works.
- Git SHA for history: `AGENTS_CORE_GIT_SHA`, then `git rev-parse HEAD`, then `GITHUB_SHA`.
- run-evals.yml runs `eval_command` via `bash -c "$EVAL_COMMAND"` from env (never interpolated): it's the caller's own command from its own workflow file, like a `run:` line.
- run-evals.yml re-checks changed paths itself (`paths` input, fnmatch globs against `git diff base...HEAD`) because a called workflow can't declare its own `pull_request` trigger; the caller's `paths:` filter remains the primary filter.
- run-evals.yml declares no permissions (like run-agent.yml) and only reads `ANTHROPIC_API_KEY`; the summary goes to `$GITHUB_STEP_SUMMARY`, which needs no token permission (no PR comment, which would need `pull-requests: write`).
- New console script `agents-evals` (subcommands `run`, `compare`) rather than more `agents-run` flags, keeping the runner's CLI unchanged.
- The README's complete example agent is executed by `tests/test_readme_example.py`; doing so caught that a $0.05 loop budget can never fit one worst-case smart-tier call at 8000 max_tokens, so the example sets `max_tokens=2000` and the README documents the pitfall.
- Tests stay SDK-free like the existing suite; `converse()`'s handling of real SDK content blocks (thinking signatures preserved) was checked once by hand against `anthropic.types.Message`.
- Pushed to `main` as instructed (and mirrored to the session branch); no tag created — a human tags v0.3.0.

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

## v0.3.1

- `temperature` always goes to `messages.create`/`parse` via `extra_body` (not detected per SDK version): anthropic 1.8 (uv.lock) has no `temperature=` on either, `extra_body` works on any version, and the API sees the same JSON field; batches keep it in `params` because they're plain JSON (verified against the real SDK).
- Only `temperature` is moved (`_BODY_ONLY_PARAMS`); the internal `params` dict keeps it top-level so cost estimates, guard-retry params, prompt hashes and batch requests are unchanged.
- The SDK test uses a real local `http.server` stub, not respx/httpx.MockTransport: the SDK now uses `httpx2` and rejects `httpx` clients, and respx isn't a dependency; the SDK client is built via `agents_core.llm.anthropic` so no other module imports `anthropic`.
- Total eval cap is additive: `run_suites` + `--total-max-usd` + `AGENTS_CORE_EVAL_TOTAL_MAX_USD` + a `total_max_usd` workflow input defaulting to empty (no total cap), so existing callers behave exactly as before; a total never raises a suite's own cap.
- A suite reached with no budget left still runs `run_suite` with a $0 cap (all cases skipped) so history records that it didn't run, rather than silently omitting it.
- `compare` treats an entry with `n_scored == 0` as having no pass rate; otherwise a budget-starved suite (now more likely with a total cap) would fail the PR gate with a fake pass-rate regression. Budget-partial suites still compare as before.
- `agents-evals run` now loads every suite before running any, so a typo in the last target fails before money is spent.
- Added `LLMJudge(temperature=, max_tokens=)` in this patch (small, additive, directly tied to the temperature bug and to repo-maintain's inflated judge estimate); deferred the judge `input=` selector, `DownloadResult` headers, a multiples/ratios guard and dirty-tree eval attribution as features/design work beyond a patch — listed in the CHANGELOG.
- Dirty-tree attribution deferred specifically because the eval run itself writes tracked files (`evals/results/*.json`, `history.jsonl`), so a naive dirty check would split one run's suites across two "SHAs" and break `compare`'s same-SHA grouping.
- Doc pins moved to `@v0.3.1` (README install lines and workflow examples, run-agent.yml/run-evals.yml headers, CLAUDE.md); historical "Migrating from" sections keep their versions.
- Pushed to `main` as instructed (and to the session branch); no tag created — a human tags v0.3.1.

## v0.3.2

- Scope and shape of each item came from the agent repos' own STATUS.md (cloned read-only): real-estate-agent's "4.3 times" case study and its `_MULTIPLE` finish validator, repo-maintain-agent's rate-limit/`Link` pagination need, fed-agent's "judge grades against the fixture, not what the loop saw", sam-agent's dirty-tree attribution.
- `no_multiples` is opt-in (default False) so existing guards behave exactly as before; it flags phrases like "12 times a year" too, which is acceptable only when an agent chooses it, and `allow=` exempts a multiple the data really holds.
- Derived phrases go into `GuardResult.unsupported` (so every existing consumer — fallbacks, logs, `Guarded.unsupported`, trace spans — treats them as failures) and also into a new `derived` list so the retry message can say why; with no derived phrases the retry text is byte-for-byte the v0.3.1 `RETRY_INSTRUCTION`.
- The multiples pattern covers real-estate-agent's validator (N times/x, twice/double/triple, number-word times) plus -fold, ×, doubled/halved forms, "half/a third as|of", "N:1"/"N-to-1" and "ratio of N"; "N to 1" with spaces is excluded because "rose from 3 to 1..." is ordinary prose, and clock times stay masked.
- `RETRY_INSTRUCTION` moved to `guards` (llm imports guards, not the reverse) and is re-exported from `llm` so v0.3.1 imports keep working.
- `DownloadResult.headers` defaults to `{}` (frozen dataclass, keyword default) so code constructing it positionally still works; `set-cookie` is dropped like elsewhere in http.py, and the sidecar file is unchanged (headers aren't persisted, only returned).
- `links` is a property parsing RFC 8288 `Link` (multiple rels per link, quoted or bare); first occurrence of a rel wins.
- The judge's input comes from `EvalOutput.input` when the task sets it, else `case.input`, then `input=` narrows it — so fed-agent can either have the task report what the loop saw or select part of the fixture; `input=` mirrors `output=`/`expected=` (dotted path or 1-arg callable) rather than taking the whole case, for consistency.
- `dirty` is a separate field rather than a suffixed SHA (e.g. `abc123-dirty`), because `compare` groups the latest run's suites by exact `git_sha`; a suffix would have changed that grouping and broken baselines.
- Dirty means staged or unstaged changes to tracked files; untracked files don't count (they're mostly caches and scratch; an untracked new prompt is a gap, documented). The evals dir and `data_dir()` are excluded because a run writes them, and the check runs once per `run_suites` before any suite writes.
- `AGENTS_CORE_GIT_DIRTY` overrides the check (for CI or wrappers that know better); `null` when git is missing or the CWD isn't a work tree. `compare` only warns on dirty entries — it doesn't fail or skip them.
- Pushed to `main` (and to the session branch); no tag created — a human tags v0.3.2.
