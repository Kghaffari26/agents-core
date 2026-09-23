# Agents Hub

A monorepo of four scheduled data agents that feed one static website, with each agent getting its own section.

| Agent | What it does | Schedule |
|---|---|---|
| `real_estate` | Tracks metro housing markets (prices, inventory, days on market, price cuts, permits) against mortgage rates; writes a weekly brief per metro | Weekly (Fri) |
| `macro` | Tracks key macro indicators and Fed communications; explains what changed since the last release | Weekdays |
| `grants` | Finds federal contracts and grants matching a business profile; scores fit and drafts a summary | Daily |
| `repo_maint` | Triages issues, flags stale PRs, drafts changelogs for configured GitHub repos | Daily |

## Architecture

```
agents-hub/
├── CLAUDE.md
├── pyproject.toml              # uv-managed, Python 3.12
├── config/
│   ├── models.toml             # model IDs per task tier
│   ├── metros.toml             # tracked metros (real estate)
│   ├── business_profile.toml   # NAICS codes, keywords, set-asides (grants)
│   └── repos.toml              # repos the maintenance agent watches
├── core/
│   ├── llm.py                  # the ONLY place the Anthropic SDK is imported
│   ├── http.py                 # retries, rate limiting, on-disk cache
│   ├── schema.py               # shared pydantic models (RunMeta, Citation, etc.)
│   ├── publish.py              # writes site data files + trims history
│   ├── costs.py                # token/$ logging and per-run budget cap
│   └── runner.py               # `python -m core.runner <agent>` entry point
├── agents/
│   ├── real_estate/            # fetch.py, transform.py, analyze.py, schema.py
│   ├── macro/
│   ├── grants/
│   └── repo_maint/
├── evals/                      # small fixture-based checks per agent
├── tests/
├── site/                       # Next.js (App Router, static export), Tailwind, Recharts
│   └── public/data/<agent>/    # latest.json + history/YYYY-MM-DD.json
└── .github/workflows/          # one cron workflow per agent + site deploy
```

Data flow: **fetch → transform (pure Python, no LLM) → analyze (LLM narrative) → validate (pydantic) → publish JSON → site rebuilds.**

## Specs (read the relevant one before working on a component)

| Component | Spec |
|---|---|
| Website (`site/`) | `docs/specs/SPEC_WEBSITE.md` |
| Macro & Fed agent (build first; defines `core/guards.py`) | `docs/specs/SPEC_MACRO.md` |
| Real estate agent | `docs/specs/SPEC_REAL_ESTATE.md` |
| Grants & contracts agent | `docs/specs/SPEC_GRANTS.md` |
| Repo maintenance agent | `docs/specs/SPEC_REPO_MAINT.md` |

Section §6 of each agent spec is the JSON contract the website depends on. Change it only together with the site types. **Where this file or BUILD_PLAN.md disagrees with a spec, the spec wins.**

## Rules

- **Numbers come from data, never from the model.** Compute every figure in Python. The LLM only writes narrative from numbers passed into the prompt and must not introduce new figures. Evals check this.
- Every narrative claim carries a citation (source name + URL) in the output schema.
- All LLM calls go through `core/llm.py`, which handles model tiers, prompt caching, retries, and cost logging. Never import `anthropic` elsewhere.
- Model tiers are set in `config/models.toml`: a `fast` tier for extraction and classification, a `smart` tier for final synthesis only. Default fast = `claude-haiku-4-5-20251001`, smart = `claude-sonnet-5` (verify current IDs at docs.claude.com).
- Use the Batch API for non-urgent bulk work (e.g., scoring many grant listings).
- Each run logs tokens and USD to `data/costs.jsonl` and aborts if it exceeds `MAX_RUN_USD` (default `0.50`).
- All HTTP goes through `core/http.py` with a cache, so re-runs during development don't re-download or re-bill.
- Every agent output validates against its pydantic schema before publishing. A failed validation fails the run, leaving the previous `latest.json` untouched.
- Keep published JSON small. Pre-aggregate, keep at most 52 weekly or 90 daily history files, and never ship raw source dumps to the site.
- Secrets come from environment variables only (`ANTHROPIC_API_KEY`, `FRED_API_KEY`, `BLS_API_KEY`, `SAM_API_KEY`, `GITHUB_TOKEN`). Never commit them. Include a `.env.example`.
- `repo_maint` is **read-only by default**. Labeling or commenting requires the `--apply` flag and a repo that is on the allowlist in `config/repos.toml`.
- Respect source terms: attribute Redfin, Zillow, FRED, etc. on the site, and don't redistribute raw datasets.

## Data sources

| Agent | Source | Access |
|---|---|---|
| real_estate | Redfin Data Center (metro-level market tracker) | Public download, attribution required |
| real_estate | Zillow Research (ZHVI home values, ZORI rents) | Public CSV, attribution required |
| real_estate | FRED: MORTGAGE30US, HOUST, CSUSHPINSA, MSPUS | Free API key |
| real_estate | Census Building Permits Survey (CBSA files), Gazetteer (centroids), ACS (median income) | Public; optional `CENSUS_API_KEY` |
| macro | FRED (inflation, labor, growth, rates, sentiment; see spec §2) and FRED release calendar | Free API key |
| macro | BLS API (optional fallback, off by default) | Free API key |
| macro | Federal Reserve RSS feed, FOMC statements and minutes, meeting calendar | Public pages |
| grants | SAM.gov Get Opportunities API v2 (**~10 requests/day on a basic key**, so budget every call) | Free API key (sam.gov account) |
| grants | Grants.gov `search2` + `fetchOpportunity` | Public, no key |
| repo_maint | GitHub REST API | `GITHUB_TOKEN`; `REPO_MAINT_TOKEN` (fine-grained PAT) for writes to your other repos |

Verify every endpoint and file URL before relying on it, since public dataset URLs change.

## Commands

```bash
uv sync                                   # install
uv run python -m core.runner <agent>      # run one agent (writes to site/public/data)
uv run python -m core.runner <agent> --dry-run   # fetch + transform, skip LLM + publish
uv run pytest                             # tests
uv run python -m evals.run <agent>        # evals
cd site && npm run dev                    # local site
cd site && npm run build                  # static export to site/out
```

## Website

A single Next.js site with static export, deployed to GitHub Pages. The full spec is in `docs/specs/SPEC_WEBSITE.md`; the summary follows.

- `/` shows an overview with a card per agent: last run time, status, headline finding, and link.
- `/real-estate` is **interactive**. It has a metro search and select, a compare mode for up to 3 metros, metric toggles (median sale price, inventory, days on market, % with price drops, sale-to-list), a time-range selector, a 30-yr mortgage rate overlay, a map colored by YoY price change, an affordability calculator (monthly payment on the metro median at the current rate), and the weekly AI brief per metro.
- `/macro` shows an indicator grid with sparklines and the latest-vs-prior change, a "what changed" summary, and an FOMC statement diff view.
- `/grants` has a filterable, sortable table (deadline, agency, amount, NAICS, fit score) with an expandable fit summary.
- `/repos` shows per-repo health (untriaged issues, stale PRs, latest changelog draft).
- Every section shows data source attribution and "last updated."
- The site reads only from `public/data/**`. No runtime API calls and no secrets in the frontend.
- It works at phone width and supports dark mode.
