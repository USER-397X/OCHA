# Geo-Insight: Which Crises Are Most Overlooked?

ETH Datathon 2026 — OCHA challenge submission by Team Epoch.

A Streamlit dashboard that ranks humanitarian crises by the mismatch between documented need and actual funding, and overlays global media attention to expose systematic neglect.

## What it does

**Crisis Dashboard** — ranks ~100 countries by a *gap score* combining funding coverage, INFORM severity, and crisis scale. Interactive world map, bar chart, scatter plot, and multi-year neglect trends.

**Media Attention** — animated choropleth (2021–2026) showing each country's share of English-language global news (GDELT). Click any country to see its full media timeline. Toggle to Funding Gap view to compare against FTS total funding shortfalls.

**Bias Analysis** — systematic analysis of whether UN pooled-fund allocation correlates with crisis severity, geography, or crisis type.

## Gap score formula

```
gap_score = (1 − coverage) × (severity / 5) × log(requirements) / log(max_req) × 100
```

Optional structural neglect bonus upweights crises that have been underfunded for multiple consecutive years.

## Data sources

| File | Source | Used for |
|------|--------|----------|
| `country_year_severity_funding.csv` | CERF + CBPF allocations vs HRP requirements | Gap score, coverage |
| `inform_severity_cleaned.csv` | INFORM Severity Index | Crisis severity, type, trend |
| `fts_requirements_funding_global.csv` | OCHA FTS (all donors) | Funding gap map |
| `hpc_hno_{year}.csv` | HNO people-in-need | Bubble sizes |
| `data/media/{ISO3}.csv` | GDELT via BigQuery | Media attention |

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Authenticate with Google Cloud

Required for BigQuery (GDELT media data).

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project ocha-493723
```

### 3. Pre-fetch media attention data

Downloads GDELT data for all 103 crisis countries (2021–2026) into `data/media/`. Takes ~10–15 minutes with 4 parallel BigQuery workers.

```bash
uv run scripts/prefetch_media.py
```

Options:
```
--workers 8          # more parallelism
--iso3 SDN,UKR,SYR  # specific countries only
--force              # re-fetch even if cached
--start 2021-01-01   # custom date range
--end   2026-03-31
```

### 4. Run the app

```bash
uv run streamlit run app.py
```

## Project structure

```
app.py          — Streamlit entry point, layout, sidebar
data.py         — data loading and caching (@st.cache_data)
scoring.py      — gap score computation
charts.py       — Plotly figure builders (pure functions)
media.py        — GDELT CSV cache + GDELT API fetching
bias.py         — bias analysis tab
chat.py         — in-app chatbot (Gemini)
scripts/
  prefetch_media.py  — one-time BigQuery data fetch
data/
  media/             — per-country GDELT CSVs (ISO3.csv)
scratch/             — exploratory scripts (BigQuery demos)
```

## Caveats

- **Coverage is pooled-fund only** (CERF + CBPF) in the gap score. The Funding Gap map uses FTS total funding.
- Countries without a formal HRP are understated.
- Media attention uses English-language GDELT only — local-language coverage is not captured.
- Gap scores support prioritisation conversations, not automated decisions.
