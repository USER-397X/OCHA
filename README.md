# Geo-Insight: Which Crises Are Most Overlooked?

ETH Datathon 2026 — OCHA challenge submission by Team Epoch.

A Streamlit dashboard that ranks humanitarian crises by the mismatch between documented need and actual funding, and overlays global media attention to expose systematic neglect.

## What did we do?
We have a website called teamstroopwafel.streamlit.app where you can see an interactive view of some maps we created and you're able to view our entire notebook! We didn't have time to create an elaborate README anymore, so please check it out! 

- Pieter, 20min before deadline

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
