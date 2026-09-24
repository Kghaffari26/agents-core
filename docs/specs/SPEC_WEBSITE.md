# Spec: Agents Hub Website (Host Site)

> **Status:** Build spec for Claude Code · **Version:** 1.0
> **Location in repo:** `site/`
> **Depends on:** the four agent specs, which produce the JSON this site reads:
> [`SPEC_REAL_ESTATE.md`](./SPEC_REAL_ESTATE.md) · [`SPEC_MACRO.md`](./SPEC_MACRO.md) · [`SPEC_GRANTS.md`](./SPEC_GRANTS.md) · [`SPEC_REPO_MAINT.md`](./SPEC_REPO_MAINT.md)
> Section **§6 of each agent spec** defines that agent's output JSON. This site treats those shapes as a contract.

---

## 1. Purpose

One website that hosts all four agents, each in its own section. It has three jobs:

1. **Portfolio piece.** A recruiter or client lands on it and within 10 seconds understands that autonomous agents are producing fresh, real data on a schedule.
2. **Product demo.** Each section looks like something you could sell: polished, fast, trustworthy, with sources shown.
3. **Zero-cost deployment.** The site is static and hosted free on GitHub Pages, with no server, no database, and no secrets in the browser.

### Goals

- One site with a shared shell (header, nav, footer, theme) and one section per agent.
- A fully interactive **Real Estate** section (charts, compare, map, calculator).
- Every page shows **when the data was last updated** and **where it came from**.
- Fast loads on a phone (Lighthouse mobile Performance ≥ 90).
- Adding a fifth agent later takes one new route plus one data folder, with no shell changes.

### Non-goals (v1)

- User accounts, logins, or server-side saved preferences.
- Runtime calls to Anthropic or any data API from the browser.
- A CMS. All copy lives in the repo.
- Paid hosting or a database.

---

## 2. Architecture

```
GitHub Actions (cron)                    GitHub Pages
┌──────────────────────┐   commit JSON   ┌─────────────────────────┐
│ agent runs (Python)  │ ──────────────▶ │ site/public/data/**     │
└──────────────────────┘                 │        │                │
          │ workflow_call                │        ▼ (build time)   │
          └────────────────────────────▶ │ Next.js static export   │
                                         │        │                │
                                         │        ▼                │
                                         │ site/out → Pages        │
                                         └─────────────────────────┘
```

- Agents write validated JSON to `site/public/data/<agent>/`.
- After committing, each agent workflow calls the deploy workflow (see §13). The deploy workflow runs `next build` with `output: 'export'` and publishes `site/out` to GitHub Pages.
- **Data loading strategy:**
  - **Small, page-critical data** (headline numbers, briefs) is read at **build time** in Server Components via `fs`. It lands in the HTML, which gives a fast first paint and good SEO.
  - **Large or interactive data** (per-metro time series, the full grants table) is fetched **client-side** from `/data/...json` on demand, which keeps the HTML small.

### Tech stack

| Concern | Choice | Why |
|---|---|---|
| Framework | Next.js (App Router), `output: 'export'` | Exports statically to Pages and is recognizable on a résumé |
| Language | TypeScript, `strict: true` | Type-safe data contracts |
| Styling | Tailwind CSS with CSS variables for tokens | Fast, themeable, supports dark mode |
| Charts | Recharts | Solid React line and bar charts with tooltips, responsive |
| Map | Leaflet + react-leaflet, OpenStreetMap tiles | Free, no API key |
| Tables | TanStack Table (headless) | Sorting, filtering and pagination for grants |
| Diff view | `diff` (jsdiff) | Word-level FOMC statement diff |
| Markdown | `react-markdown` + `rehype-sanitize` | Renders changelog drafts safely |
| Icons | lucide-react | Consistent and tree-shakable |
| Validation | zod (generated from JSON Schema) | Catches bad data at build time |
| Unit tests | Vitest + Testing Library | Component logic |
| E2E tests | Playwright | Smoke-tests every route on the static build |
| Lint/format | ESLint (next config) + Prettier | Standard tooling |

> **Leaflet caveat:** Leaflet touches `window`. Load map components with `next/dynamic` and `{ ssr: false }`, or the static export fails.
> **Tile usage:** OSM's public tile server has a fair-use policy. Low portfolio traffic is fine. If traffic grows, switch to a free-tier provider such as Stadia or MapTiler, configured through one env var.

---

## 3. Data contracts

The site talks to the agents only through JSON in `site/public/data/`. **The Python pydantic models are the source of truth.**

### Contract pipeline

1. Each agent's pydantic output models export JSON Schema to `schemas/<agent>.schema.json` (repo root) through `uv run python -m core.export_schemas`.
2. A site script, `npm run gen:types`, uses `json-schema-to-typescript` to generate `site/src/types/generated/<agent>.ts`, and `json-schema-to-zod` to generate `site/src/lib/validators/<agent>.ts`.
3. At build time, every JSON file the site loads is parsed with its zod schema. **A validation failure fails the build**, because a failed build is better than a broken page going live.
4. CI checks that the generated types are current: it runs `gen:types`, then `git diff --exit-code`.
5. Every agent `latest.json` has `meta.schema_version` (semver). The site declares which major version it supports per agent. If the major version doesn't match, the build fails with a clear message.

### Folder layout (produced by agents)

```
site/public/data/
├── manifest.json                     # written by core.publish after any agent run
├── costs/summary.json                # aggregated from data/costs.jsonl at publish time
├── real_estate/
│   ├── latest.json                   # national + metro index (small)
│   ├── metros/<slug>.json            # per-metro detail + series (lazy-loaded)
│   └── history/YYYY-MM-DD.json
├── macro/
│   ├── latest.json
│   └── history/YYYY-MM-DD.json
├── grants/
│   ├── latest.json                   # stats + top matches with summaries
│   ├── all.json                      # full scored list (lazy-loaded)
│   └── history/YYYY-MM-DD.json
└── repo_maint/
    ├── latest.json
    └── history/YYYY-MM-DD.json
```

### Shared `meta` block (every `latest.json`)

```json
{
  "meta": {
    "agent": "macro",
    "schema_version": "1.0.0",
    "run_id": "2026-09-23T14-00-05Z-a1b2c3",
    "started_at": "2026-09-23T14:00:05Z",
    "finished_at": "2026-09-23T14:01:12Z",
    "status": "ok",
    "data_changed": true,
    "cost_usd": 0.041,
    "model_usage": { "fast": { "input_tokens": 0, "output_tokens": 0 }, "smart": { "input_tokens": 6120, "output_tokens": 540 } },
    "sources": [{ "name": "FRED", "url": "https://fred.stlouisfed.org/", "retrieved_at": "2026-09-23T14:00:09Z" }]
  }
}
```

`data_changed: false` means the agent ran, found no new source data, and reused the previous narrative. The site shows "Checked 2h ago · no new data since Sep 19".

### `manifest.json` (shared; drives the overview page)

```json
{
  "generated_at": "2026-09-23T14:05:11Z",
  "agents": [
    {
      "id": "real_estate",
      "name": "Real Estate Market Agent",
      "route": "/real-estate",
      "status": "ok",
      "last_run_at": "2026-09-19T15:02:44Z",
      "last_data_change_at": "2026-09-19T15:02:44Z",
      "expected_interval_hours": 168,
      "next_run_hint": "Fridays 08:00 PT",
      "headline": "Inventory up 14% YoY nationally; 31 of 50 metros saw more price cuts than a year ago.",
      "key_stats": [
        { "label": "US median sale price", "value": 431200, "format": "currency_compact", "delta": 0.021, "delta_format": "percent_signed", "good_direction": "neutral" }
      ],
      "run_cost_usd": 0.07,
      "items_count": 50
    }
  ]
}
```

- `status` is one of `ok | stale | failed`.
- The site also computes **stale on the client**. If `last_run_at` is older than `2 × expected_interval_hours`, the card shows "Stale" even when the manifest says `ok`.
- `headline` and `key_stats` are generated **deterministically** by each agent (see each spec's §6), not by the LLM.

### `costs/summary.json`

```json
{
  "month": "2026-09",
  "total_usd": 3.87,
  "by_agent": [{ "agent": "grants", "usd": 2.10, "runs": 23 }],
  "daily": [{ "date": "2026-09-01", "usd": 0.14 }],
  "all_time_usd": 11.52,
  "avg_cost_per_run": [{ "agent": "macro", "usd": 0.012 }]
}
```

---

## 4. Information architecture

| Route | Page | Data |
|---|---|---|
| `/` | Overview / landing | `manifest.json`, `costs/summary.json` |
| `/real-estate` | Real estate dashboard (interactive) | `real_estate/latest.json` + lazy `metros/<slug>.json` |
| `/real-estate/[slug]` | Single-metro deep link (static params from the metro index) | `metros/<slug>.json` |
| `/macro` | Macro and Fed dashboard | `macro/latest.json` |
| `/grants` | Grants and contracts finder | `grants/latest.json` + lazy `all.json` |
| `/repos` | Repo maintenance dashboard | `repo_maint/latest.json` |
| `/about` | How it works: architecture, costs, methods, limitations | static + `costs/summary.json` |
| `/404` | Not found | none |

`/real-estate/[slug]` uses `generateStaticParams()`, which reads the metro index at build time. Every metro gets a shareable URL, such as `/real-estate/austin-tx`.

---

## 5. Global shell

### Header

- Left: the site name as a text wordmark ("Agents Hub"), linking to `/`.
- Center/right: nav links **Real Estate · Macro · Grants · Repos · About**, each with a tiny status dot from the manifest (green ok, amber stale, red failed).
- Far right: a theme toggle (system / light / dark) and a GitHub icon linking to the repo.
- Mobile (< 768px): the nav collapses into a menu button that opens a full-width sheet. The active route is highlighted.

### Footer

- "Data updates automatically via GitHub Actions" with a link to the repo's Actions tab.
- Attribution links: Redfin, Zillow, FRED, BLS, U.S. Census Bureau, Federal Reserve Board, SAM.gov, Grants.gov, GitHub.
- "Not financial, legal, or investment advice. AI-generated summaries may contain errors; verify with the linked sources."
- Build timestamp and short commit SHA, injected at build through `NEXT_PUBLIC_BUILD_SHA`.

### Shared components

| Component | Purpose |
|---|---|
| `<PageHeader title subtitle meta sources />` | Consistent top of every section, containing `<LastUpdated>` and `<SourceList>` |
| `<LastUpdated at dataChangedAt intervalHours />` | Relative time ("3 hours ago"), absolute time on hover, a stale badge, and a "no new data since…" line |
| `<SourceList sources />` | Collapsible "Sources" list with links and retrieval times |
| `<StatCard label value delta deltaFormat goodDirection format sparkline? />` | A big number with a semantic up/down delta |
| `<Sparkline data />` | A tiny inline Recharts line with no axes |
| `<AiBrief text bullets? citations model generatedAt />` | A styled AI narrative block with an "AI-generated" label, footnoted citation links, and the model name in small print |
| `<EmptyState />`, `<ErrorState />`, `<Skeleton />` | Loading, empty and failure states for every data view |
| `<AgentStatusBadge status />` | An ok / stale / failed pill |
| `<FailureBanner meta />` | Shown when the last run failed: "Last update failed at …; showing data from …" |

**Delta coloring is semantic, not just sign-based.** For example, rising unemployment is "bad" (red), while rising inventory is "neutral" (amber). Every metric definition in `src/lib/metrics.ts` carries `goodDirection: "up" | "down" | "neutral"`. Agents also send `good_direction` in their JSON. The agent's value wins, and the site's value is the fallback.

---

## 6. Design system

### Tokens (CSS variables on `:root`, overridden under `.dark`)

```css
:root {
  --bg: #fbfbfa;          --surface: #ffffff;     --surface-2: #f3f3f1;
  --text: #16181d;        --text-muted: #5b6270;  --border: #e4e4e0;
  --accent: #2f5bea;      --accent-contrast: #ffffff;
  --good: #1a7f4b;        --bad: #c23a2b;         --neutral: #8a6d1f;
  --chart-1: #2f5bea; --chart-2: #e0781f; --chart-3: #1a9e8a; --chart-4: #9b4dca;
  --diverge-neg: #c23a2b; --diverge-mid: #e9e6df; --diverge-pos: #1a7f4b;
  --radius: 10px;
}
.dark {
  --bg: #0e1015;          --surface: #161920;     --surface-2: #1e222b;
  --text: #eceef2;        --text-muted: #9aa2b1;  --border: #2a2f3a;
  --accent: #6d8cff;      --good: #3fbf7f;        --bad: #ff6b5b;  --neutral: #d9b44a;
  --chart-1: #6d8cff; --chart-2: #ffa14d; --chart-3: #3fd1bb; --chart-4: #c18bff;
  --diverge-neg: #ff6b5b; --diverge-mid: #3a3f4a; --diverge-pos: #3fbf7f;
}
```

- Charts read colors from these variables through `getComputedStyle`, so they theme automatically. They re-read the variables when the theme changes.
- **Never rely on color alone.** Every delta also shows ▲/▼ and a sign.
- Theme handling: a `class` strategy on `<html>`, with an inline pre-hydration script that reads the stored choice (wrapped in try/catch) and falls back to `prefers-color-scheme`. This prevents a flash.

### Typography

- UI font: Inter, loaded with `next/font/google`. It is self-hosted at build time, so there is no runtime Google request.
- Numbers: `font-variant-numeric: tabular-nums` wherever figures appear.
- Scale: 12 / 14 / 16 (base) / 20 / 24 / 32 / 40.

### Layout

- Max content width is 1200px, with 16px gutters on mobile and 24px on desktop.
- Stat card grid: 1 column under 640px, 2 columns from 640px, and 3–4 columns from 1024px.
- Nothing scrolls horizontally at 360px width. The one exception is tables, which scroll inside their own container.

### Number formatting (`src/lib/format.ts`)

| Kind | Example |
|---|---|
| `currency_compact` | `$412K`, `$1.2M` |
| `currency` | `$412,500` |
| `percent` (a level, e.g. a rate) | `6.84%` |
| `percent_signed` (a change) | `+4.2%` |
| `pp_signed` (change in a rate) | `+0.25 pp`. Never express a rate change as a %. |
| `count` | `12,431` |
| `count_signed` | `+1,420` |
| `count_signed_thousands` (value already in thousands, e.g. payrolls) | `+142K` |
| `decimal1` | `5.8` (e.g. months of supply) |
| `days` | `34 days` |
| `ratio` | `0.987` or `98.7%` (sale-to-list shows as a %) |
| dates | `Sep 19, 2026`; relative time through `Intl.RelativeTimeFormat` |

Every formatter has unit tests, including null/undefined handling, which renders "—".

---

## 7. Page specs

### 7.1 Overview (`/`)

**Goal:** show in one screen that four agents are alive and producing value.

- **Hero:** one line ("Four AI agents that track housing, the economy, federal contracts, and code, updated automatically."), a live "Last activity: 2 hours ago" computed from the manifest, and a link to `/about`.
- **Agent cards:** a 2×2 grid, stacked on mobile. Each card shows:
  - the name, a lucide icon, `<AgentStatusBadge>` and `<LastUpdated>`
  - the `headline` sentence from the manifest
  - the `key_stats` (up to 2) as mini `<StatCard>`s
  - an "Open →" link to the section
- **Cost transparency strip:** "This month: $X.XX across N runs" plus a small horizontal bar chart by agent, from `costs/summary.json`. This is a strong portfolio signal because it shows you manage LLM cost.
- **How it works:** a four-step visual (Fetch → Compute → AI summary → Publish) that links to `/about`.

### 7.2 Real Estate (`/real-estate`), interactive

This is the flagship page. Data comes from `real_estate/latest.json` (the index plus national data) and the lazy-loaded `metros/<slug>.json`. Exact shapes are in `SPEC_REAL_ESTATE.md` §6.

**Layout on desktop, top to bottom:**

1. **`<PageHeader>`** with last updated and sources. It notes the data period ("Market data through Aug 2026 · rates as of Sep 18").
2. **National strip:** stat cards for median sale price (YoY), active inventory (YoY), median days on market (YoY), % of homes sold with price drops (YoY), 30-yr mortgage rate (weekly change in pp), and housing starts (MoM).
3. **National brief:** `<AiBrief>` with the agent's national summary.
4. **Alerts strip:** chips for the most notable deterministic flags across metros (e.g., "Inventory +25% YoY: Austin, Tampa, Denver"), each linking to its metro.
5. **Metro explorer (the interactive core):**
   - **Metro picker:** a combobox with type-ahead over the metro index (name + state), keyboard accessible. You can select up to **3 metros** to compare, shown as removable chips. The default selection is the #1 metro by homes sold plus the top price mover. A "US" pseudo-metro (the national series) can also be selected.
   - **Metric toggle (segmented control, scrollable on mobile):** built from the agent's `metric_registry`, so new metrics appear automatically. The v1 metrics are Median sale price · Inventory · Days on market · % price drops · Sale-to-list · Months of supply · Homes sold · New listings · Zillow home value · Zillow rent · Building permits.
   - **Time range:** 1Y · 2Y · 3Y (the maximum shipped).
   - **Overlay toggle:** "Show 30-yr mortgage rate" adds a secondary right-hand y-axis in the muted color.
   - **Chart:** a Recharts `LineChart` with one line per selected metro. The tooltip shows every selected metro for the hovered month. Null points break the line (`connectNulls={false}`) instead of drawing 0. Chart height is reserved to avoid layout shift.
   - **Stat table under the chart:** one column per selected metro and one row per metric (latest value and YoY). The best and worst in each row get a subtle highlight, with direction from `goodDirection`.
   - **Market temperature:** each selected metro shows its temperature label (Hot / Warm / Balanced / Cool / Cold) and score, plus a tooltip that explains the components.
   - **Per-metro brief:** one tab per selected metro, each showing that metro's `<AiBrief>` and its flags.
6. **Map:** a US map (Leaflet, OSM tiles) with a circle marker per metro at its CBSA centroid.
   - Color encodes the YoY change of the selected metric on a diverging scale (`--diverge-*`), clamped at the 5th and 95th percentiles. Size encodes market size (homes sold, sqrt scale).
   - Clicking a marker adds that metro to the comparison, or replaces the oldest one if 3 are already selected.
   - The legend shows the scale, and the map follows the metric toggle. For metrics without a YoY value, the map uses the level on a sequential scale.
   - An accessible fallback, "View as table", shows the same data as a sortable table.
7. **Movers:** three lists ("Biggest price gains YoY", "Biggest price declines YoY" and "Fastest-growing inventory"), each with the top 5. Every item links to its metro.
8. **Affordability calculator:**
   - Inputs: metro (defaults to the first selected), home price (defaults to that metro's median sale price), down payment % (slider 0–50, default 20), interest rate (defaults to the latest 30-yr rate and is editable), term (15/30), property tax % (default 1.1), insurance $/yr (default 1,800), HOA $/mo (default 0).
   - Outputs: monthly principal and interest, a breakdown of the total monthly payment (stacked bar plus list), and the **income needed** at a 28% front-end ratio. When the agent provides `median_household_income`, it also shows the payment as % of the metro's median household income.
   - **Then vs now:** the same metro's payment one year ago, using last year's median and last year's rate, both from the metro JSON.
   - Formula: standard amortization, `M = P·r(1+r)^n / ((1+r)^n − 1)`, with `r` the monthly rate. When `r = 0`, `M = P/n`. It is unit-tested against known values.
   - Labeled as an estimate, not financial advice.
9. **Methodology and sources** (collapsible): the definition of each metric, the data lag (Redfin monthly, Zillow monthly, Freddie Mac weekly), how the temperature score works, and attribution.

**URL state:** selected metros, metric, range and overlay are mirrored in the query string (`?m=austin-tx,denver-co&metric=inventory&range=2y&rate=1`), so any view is shareable. Use `useSearchParams` with `router.replace(..., { scroll: false })`. Parsing is defensive: unknown slugs are dropped, and invalid values fall back to defaults. With static export, `useSearchParams` components must sit inside `<Suspense>`.

**Performance:** only the index loads at first. Each metro file loads when selected and is cached in memory. `<Skeleton>` shows while it loads. First-load JS for this route stays at or under 250KB gzipped, with the map split into its own chunk.

**`/real-estate/[slug]`** renders the same explorer preselected to that metro, with its brief expanded. It gets `metadata` like "Austin, TX housing market — Aug 2026 | Agents Hub".

### 7.3 Macro (`/macro`)

Data comes from `macro/latest.json`. Shapes are in `SPEC_MACRO.md` §6.

1. `<PageHeader>`.
2. **Regime strip:** five compact chips from the agent's deterministic `regimes`: Inflation (e.g., "Cooling · Core PCE 2.8%"), Labor ("Softening · 4.4%"), Growth, Policy ("Holding · 4.00–4.25%") and Curve.
3. **"What changed" brief:** an `<AiBrief>` with 3–6 bullets. Each bullet cites its indicator release and links to the FRED series.
4. **Indicator grid:** one card per indicator, grouped as Inflation / Labor / Growth / Rates / Sentiment. Each card shows:
   - the latest value, the period it covers ("Aug 2026") and the release date
   - the change vs the prior reading with semantic coloring, plus YoY where meaningful
   - a 24-month sparkline
   - a **revision badge** when the prior period was revised ("Jul revised: +73K → +41K")
   - a "Next release: Oct 3" line
   - Clicking a card expands it into a larger chart with a 2Y / 5Y / 10Y range toggle.
5. **Yield curve panel:** 2y and 10y lines with the 10y–2y spread shaded underneath, and inversion periods (spread < 0) highlighted. A small "current curve" chart plots the latest 3M / 2Y / 5Y / 10Y / 30Y points as a snapshot.
6. **Fed panel:**
   - The latest FOMC decision (target range, change in bp and the vote with dissents), the meeting date, and a countdown to the next meeting.
   - The AI plain-English read of the statement, plus a **tone shift** badge (More hawkish / Unchanged / More dovish) with its rationale.
   - A **statement diff viewer** comparing the previous statement to the latest. It has inline and side-by-side modes, shows additions in green and deletions in red with strikethrough, and hides unchanged paragraphs behind "Show N unchanged paragraphs". On mobile it is inline only. Word-level diffs are computed client-side with jsdiff from `previous_text` and `latest_text`. The agent's sentence-level `changes` list drives a "Key edits" summary above the diff.
   - A minutes link and summary, when minutes for the latest meeting have been published.
7. **Upcoming releases calendar:** the next 14 days of scheduled releases for tracked indicators, plus FOMC dates.
8. Methodology and sources.

### 7.4 Grants and Contracts (`/grants`)

Data comes from `grants/latest.json` and the lazy-loaded `grants/all.json`. Shapes are in `SPEC_GRANTS.md` §6.

1. `<PageHeader>` with a **profile summary chip** ("Matching for: Small software consultancy · NAICS 541511, 541512 · Small business"). The chip links to an explanation of how to change the profile in the repo.
2. **Summary strip:** new since the last run, closing within 14 days, total active matches (score ≥ threshold), and the largest disclosed value among matches.
3. **Top matches:** cards for the top 10 by fit score. Each card shows:
   - the title, agency, a source badge (SAM.gov / Grants.gov), notice type (Solicitation, Sources Sought, Grant, Forecast…) and set-aside
   - the value, award ceiling or estimate when known, or "Not disclosed"
   - the deadline with a countdown ("closes in 6 days", red under 7 days)
   - the fit score as a 0–100 meter with sub-score bars (capability, eligibility, size, timeline, strategic)
   - a **recommendation** badge: Pursue / Consider / Pass
   - the AI fit summary, collapsed to 2 lines and expandable, followed by "Risks" and "Next steps" lists
   - "Why it matches" reason chips, and a "NEW" badge if it was first seen this run
   - a link to the official listing
4. **Full table** (TanStack Table), which loads `all.json` on demand:
   - Columns: Fit, Title, Agency, Source, Type, NAICS, Set-aside, Posted, Deadline, Value.
   - Filters: text search, source, type, agency (multi-select), set-aside, NAICS, closing within N days, minimum fit score, and "new only".
   - Every column sorts. Pagination is 25 per page. CSV export covers the filtered rows (client-side, RFC 4180 quoting).
   - Expanding a row shows the short reasons and, if the item is in the top 20, the full summary.
5. **Deadline timeline:** a horizontal timeline of the next 30 days, with top matches plotted by deadline and colored by recommendation.
6. **Methodology:** how scoring works, the rubric, the limitations, and "Always verify on the official listing before acting."

### 7.5 Repos (`/repos`)

Data comes from `repo_maint/latest.json`. Shapes are in `SPEC_REPO_MAINT.md` §6.

1. `<PageHeader>` with a **mode badge**: "Report mode (read-only)" or "Apply mode".
2. **Repo cards,** one per watched repo, showing:
   - a health score (0–100) with a letter grade. Hovering shows the penalty breakdown.
   - open issues (untriaged count highlighted), open PRs, stale PRs, median time to first response, and default-branch CI status
   - a 12-week activity sparkline of issues opened vs closed
3. **Triage queue:** a table of untriaged issues showing the classification, suggested labels (chips), priority, confidence, missing-info checklist and duplicate candidates (linked with similarity %). In apply mode, a column shows whether labels were applied.
4. **Stale PRs:** a list with author, age, last activity, review state, CI state and the suggested nudge text (copy button).
5. **Changelog drafts:** the latest draft for each repo, rendered as sanitized Markdown with no raw HTML, plus a "Copy Markdown" button and a "since v1.2.0" note.
6. **Actions log:** what the agent did this run, or would have done in report mode, with statuses planned / applied / skipped and a reason.

### 7.6 About (`/about`)

- An architecture diagram: an inline SVG committed to the repo (hand-authored, or exported from Mermaid with `@mermaid-js/mermaid-cli` via an npm script) so no Mermaid runtime is shipped.
- Three sentences on how each agent works, each linking to its spec in the repo.
- A cost section with a monthly cost chart by agent and cost per run, from `costs/summary.json`.
- An "Engineering choices" list: numbers computed in code rather than by the model, evals, cost caps, caching, and schema contracts.
- Data sources and licenses, and a limitations and disclaimer section.
- "Built with" and a link to the repo.

---

## 8. State, loading and errors

- **Build-time loaders** live in `src/lib/data/server.ts`: `getManifest()`, `getRealEstateIndex()`, `getMacro()`, `getGrantsLatest()`, `getRepoMaint()` and `getCostSummary()`. Each reads from `public/data` (or `DATA_DIR`), validates with zod, and throws an error that names the file and the zod path when validation fails.
- **Client loaders** live in `src/lib/data/client.ts` as `useJson<T>(path, schema)`: a small hook with an in-memory cache, abort on unmount and zod validation. When validation fails, it renders `<ErrorState>`: "This data didn't load correctly. It'll refresh after the next update."
- All data URLs go through `dataUrl(path)`, which prefixes `NEXT_PUBLIC_BASE_PATH`. URLs are never built by hand.
- **Missing optional data** (e.g., a metro without Zillow rent data) renders "—", never 0.
- **An agent whose last run failed** leaves the previous `latest.json` in place, since agents never overwrite on failure. The manifest marks it `failed`, and the page shows `<FailureBanner>`.
- **No global state library.** URL params and local component state are enough.

---

## 9. Accessibility

- WCAG 2.1 AA contrast for all text in both themes.
- Every interactive control can be reached by keyboard and has a visible focus ring. The combobox follows the WAI-ARIA combobox pattern.
- Charts carry `aria-label` summaries generated from the data ("Median sale price, Austin vs Denver, last 2 years; Austin down 3.1%, Denver up 1.4%") and a "View data table" toggle beside them.
- The map has a table fallback (§7.2).
- `prefers-reduced-motion` disables chart animations.
- Axe checks run in Playwright on every route in both themes (`@axe-core/playwright`), and serious or critical violations fail CI.

---

## 10. SEO and sharing

- Each route's `metadata` (title, description) is generated from the data, e.g., "Macro dashboard — CPI 2.9% YoY (Aug 2026)".
- Per-section OG images come from `opengraph-image.tsx` for static routes. For `[slug]` routes, a build script (`scripts/gen-og.mjs`, using `satori` + `@resvg/resvg-js`) pre-renders PNGs into `public/og/`.
- `sitemap.xml` and `robots.txt` are generated at build and include every metro slug.
- Canonical URLs use `NEXT_PUBLIC_SITE_URL`.

---

## 11. Performance budgets

| Metric | Budget |
|---|---|
| Lighthouse mobile Performance, `/` | ≥ 95 |
| Lighthouse mobile Performance, `/real-estate` | ≥ 90 |
| First-load JS, `/` | ≤ 120KB gzipped |
| First-load JS, `/real-estate` (excluding the lazy map) | ≤ 250KB gzipped |
| Largest data file fetched before first paint | ≤ 300KB |
| CLS | < 0.05 (chart heights reserved) |

These are enforced with `@next/bundle-analyzer` (manual) and a Lighthouse CI step (`treosh/lighthouse-ci-action`) in the deploy workflow, which warns but doesn't block in v1.

---

## 12. Project structure

```
site/
├── next.config.mjs          # output: 'export', basePath, images.unoptimized, trailingSlash
├── package.json
├── tailwind.config.ts
├── tsconfig.json
├── playwright.config.ts
├── vitest.config.ts
├── scripts/
│   ├── gen-types.mjs        # JSON Schema → TS + zod
│   ├── gen-sitemap.mjs
│   └── gen-og.mjs
├── public/
│   ├── data/                # written by agents; never hand-edit
│   └── og/                  # generated OG images
├── test/fixtures/           # small realistic data for tests (mirrors public/data)
└── src/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                       # overview
    │   ├── real-estate/page.tsx
    │   ├── real-estate/[slug]/page.tsx
    │   ├── macro/page.tsx
    │   ├── grants/page.tsx
    │   ├── repos/page.tsx
    │   ├── about/page.tsx
    │   └── not-found.tsx
    ├── components/
    │   ├── shell/        (Header, Footer, ThemeToggle, MobileNav)
    │   ├── common/       (PageHeader, StatCard, Sparkline, AiBrief, SourceList, LastUpdated, FailureBanner, states)
    │   ├── real-estate/  (MetroPicker, MetricToggle, CompareChart, MetroTable, Temperature, MetroMap, Movers, AlertsStrip, AffordabilityCalc)
    │   ├── macro/        (RegimeStrip, IndicatorGrid, IndicatorCard, YieldCurve, FomcPanel, StatementDiff, ReleaseCalendar)
    │   ├── grants/       (ProfileChip, TopMatches, MatchCard, FitMeter, GrantsTable, DeadlineTimeline)
    │   └── repos/        (RepoCard, TriageTable, StalePrs, ChangelogDraft, ActionsLog)
    ├── lib/
    │   ├── data/server.ts, data/client.ts, data/url.ts
    │   ├── format.ts, mortgage.ts, colors.ts, metrics.ts, urlState.ts, csv.ts
    │   └── validators/   # zod (generated)
    └── types/generated/  # TS types (generated)
```

### `next.config.mjs` essentials

```js
const repo = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? '';
const isPages = process.env.GITHUB_ACTIONS === 'true' && !process.env.CUSTOM_DOMAIN;
const basePath = isPages ? `/${repo}` : '';
export default {
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
    NEXT_PUBLIC_BUILD_SHA: process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev',
    NEXT_PUBLIC_SITE_URL: process.env.SITE_URL ?? '',
  },
};
```

If you use a custom domain, set the `CUSTOM_DOMAIN` repo variable and add `public/CNAME`.

---

## 13. Deployment workflow (`.github/workflows/deploy-site.yml`)

> **Important:** agent workflows commit data with the default `GITHUB_TOKEN`, and **pushes made with `GITHUB_TOKEN` do not trigger other workflows.** To handle that, `deploy-site.yml` also accepts `workflow_call`, and each agent workflow calls it as a final job after committing. The caller must grant `pages: write` and `id-token: write`.

```yaml
name: Deploy site
on:
  push:
    branches: [main]
    paths: ['site/**', 'schemas/**']
  workflow_dispatch:
  workflow_call:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: site } }
    steps:
      - uses: actions/checkout@v4
        with: { ref: main }   # pick up the data commit the caller just pushed
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: npm, cache-dependency-path: site/package-lock.json }
      - run: npm ci
      - run: npm run gen:types && git diff --exit-code src/types src/lib/validators
      - run: npm run lint && npm run test -- --run
      - run: npm run build
      - run: npx playwright install --with-deps chromium && npm run e2e
      - uses: actions/upload-pages-artifact@v3
        with: { path: site/out }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: { name: github-pages, url: '${{ steps.d.outputs.page_url }}' }
    steps:
      - id: d
        uses: actions/deploy-pages@v4
```

In the repo settings, set Pages → Source to **GitHub Actions**.

---

## 14. Testing

| Layer | What | Tool |
|---|---|---|
| Unit | Formatters, mortgage math, metric direction and coloring, URL-state parsing, CSV export, stale detection | Vitest |
| Component | MetroPicker (select, remove, max 3, keyboard), AffordabilityCalc (inputs → outputs), GrantsTable (filter, sort, CSV), StatementDiff, LastUpdated (stale) | Vitest + Testing Library |
| Contract | Every fixture in `test/fixtures/**` validates against its zod schema, plus one deliberately broken fixture that must fail | Vitest |
| E2E | Each route renders with fixture data; the real-estate compare flow and URL round-trip; the grants filter and export flow; theme toggle; no console errors | Playwright on the static `out/`, served with `npx serve` |
| A11y | Axe on every route in both themes | @axe-core/playwright |

Fixture data lives in `site/test/fixtures/`: small, realistic and committed. E2E builds with `DATA_DIR=test/fixtures` so tests don't depend on live agent output.

---

## 15. Acceptance criteria

- [ ] `npm run build` produces `site/out` with every route, including each `/real-estate/[slug]`.
- [ ] The site deploys to GitHub Pages and works under the repo basePath, including client-side data fetches.
- [ ] The overview shows 4 agent cards with correct statuses and relative times. Manually aging a timestamp flips the card to "Stale".
- [ ] Real estate: select 3 metros, switch metric and range, toggle the rate overlay, and click a map marker to add a metro. Copying the URL and opening it in a new tab reproduces the view.
- [ ] The affordability calculator matches a known amortization example to the cent (e.g., $400,000 at 6.5% for 30 years is $2,528.27/mo P&I).
- [ ] Macro: indicator cards, revision badges, the yield curve and the FOMC diff all render, and the diff shows additions and deletions correctly.
- [ ] Grants: filtering by "closing within 14 days" plus min fit 70, then sorting by deadline and exporting CSV, produces a file matching the visible rows.
- [ ] Repos: the triage table, stale PRs, changelog and actions log render, with the correct mode badge.
- [ ] Both themes pass axe with no serious or critical issues, and every page is usable at 360px width.
- [ ] Lighthouse mobile scores meet §11 on `/` and `/real-estate`.
- [ ] A corrupted data file fails the build with an error naming the file and field.

---

## 16. Build order (prompts for Claude Code)

1. **Scaffold:** "Read `docs/specs/SPEC_WEBSITE.md`. Scaffold `site/` per §12: Next.js App Router, strict TypeScript, Tailwind with the §6 tokens, class-based dark mode with the no-flash script, `next.config.mjs` per §12, the shell from §5, and placeholder pages for every route in §4. Add Vitest, Playwright, ESLint and Prettier. Make `npm run build` pass."
2. **Contracts and fixtures:** "Implement §3 and §8: the `gen:types` script, zod validators, server and client loaders, `dataUrl`, and realistic fixtures in `test/fixtures` for all four agents that match §6 of each agent spec. Add contract tests, including a broken fixture that must fail."
3. **Common components:** "Build every shared component in §5 and the formatters in §6, with unit tests."
4. **Overview and About:** "Build §7.1 and §7.6."
5. **Real estate:** "Build all of §7.2: URL state, the map (dynamic import, no SSR) with table fallback, movers, alerts, temperature, the calculator with tests, and the `[slug]` route with `generateStaticParams`."
6. **Macro:** "Build §7.3, including the statement diff viewer and revision badges."
7. **Grants:** "Build §7.4 with TanStack Table, filters and CSV export."
8. **Repos:** "Build §7.5."
9. **Quality and deploy:** "Add axe checks, Lighthouse CI, sitemap and robots, OG images and metadata (§9–§11), and the deploy workflow in §13 with `workflow_call`. Fix anything that fails."

---

## 17. Future (post-v1)

- An email digest signup using a static form service, which is the monetization path for grants.
- Per-visitor saved metros in `localStorage`, as a convenience only and wrapped in try/catch.
- A `/status` page with run history per agent (success rate, duration, cost), built from `history/`.
- White-label mode: env vars switch the branding and limit the sections for client deployments.
- An embeddable widget (`/embed/real-estate/[slug]`) for brokers to iframe a metro chart.
