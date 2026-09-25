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
│   ├── guards.py                 # the number guard (verify_numbers, text_guard, fields_guard)
│   ├── http.py                  # retries, rate limiting, request budgets, on-disk cache
│   ├── costs.py                  # CostTracker, BudgetExceeded, costs-summary.json
│   ├── publish.py                # atomic JSON writes, dated history, manifest-entry.json
│   ├── export_schemas.py        # schema.json, written automatically at publish time
│   ├── schema.py                 # shared pydantic models (RunMeta, ManifestEntry, etc.)
│   ├── settings.py               # data_dir()/publish_dir(), config/models.toml loading
│   └── config/models.toml       # packaged default model tiers + pricing
├── tests/
│   ├── fake_agent.py             # FakeAgent + module-level AGENT, used across most tests
│   └── test_install_entry_point.py  # real uv sync + agents-run subprocess integration test
├── docs/specs/                  # historical, pre-package design specs — not current
└── .github/workflows/
    ├── ci.yml                    # ruff + pytest + actionlint
    └── run-agent.yml             # workflow_call only; agent repos call this
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
- **`run-agent.yml` is `workflow_call`-only.** It must never gain a `push`/
  `pull_request`/`schedule` trigger of its own — it only runs when another repo's
  workflow calls it.
- Every change to a public module (`agent.py`, `schema.py`, `llm.py`, `guards.py`,
  `registry.py`, `runner.py`, `publish.py`, `costs.py`, `export_schemas.py`,
  `http.py`, `settings.py`) or to the data-branch contract is a breaking change for
  four other repos — update `README.md`'s matching section in the same change.

## Commands

```bash
uv sync                                    # install (editable)
uv run pytest                              # tests, including the real install/entry-point test
uv run ruff check .                        # lint
uv run ruff format --check .               # format check
/tmp/actionlint .github/workflows/*.yml    # or `actionlint` if on PATH
```

## Releasing

Agent repos and `run-agent.yml`'s own doc example pin a tag (`@v0.1.0`). Bump
`pyproject.toml`'s `version`, then a human (not an agent session) tags and pushes:
`git tag v0.1.0 && git push origin v0.1.0` — agent sessions in this repo should never
push a tag themselves.
