"""Media Attention — GDELT-powered news coverage vs. funding analysis."""
import streamlit as st
import pandas as pd

from data import load_gap_df, load_severity_df, build_name_map, load_fts_funding
from scoring import compute_gap_scores
from charts import media_overview_map, media_timeseries
from media import get_media_attention, get_annual_media_map, is_stale, _gap_fill_and_save

# ── Data ──────────────────────────────────────────────────────────────────────
gap_df      = load_gap_df()
sev_df      = load_severity_df()
name_map    = build_name_map(sev_df)
fts_df      = load_fts_funding()
full_scored = compute_gap_scores(gap_df, use_neglect=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    with st.expander("🎬 Demo controls"):
        force_stale = st.checkbox(
            "Force stale data (show refresh button)",
            help="Makes all media CSVs appear outdated so the live refresh can be demonstrated.",
        )

# ── Top navigation ───────────────────────────────────────────────────────────
_nc = st.columns([1, 1, 1, 1, 2])
with _nc[0]:
    st.page_link("pages/dashboard.py", label="🌍 Overview", use_container_width=True)
with _nc[1]:
    st.page_link("pages/media_attention.py", label="📰 Media Attention", use_container_width=True)
with _nc[2]:
    st.page_link("pages/bias_analysis.py", label="🔍 Bias Analysis", use_container_width=True)
with _nc[3]:
    st.page_link("pages/stroopwafel.py", label="🧇 Stroopwafel EDA", use_container_width=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #0a2318 0%, #14472e 55%, #166534 100%);
  padding: 2rem 2.5rem 1.8rem;
  border-radius: 16px;
  margin-bottom: 1.6rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.22);
">
  <p style="
    color: #86efac; font-size: .8rem; letter-spacing: .14em;
    text-transform: uppercase; margin: 0 0 .5rem; font-weight: 600;
  ">Deep Dive · GDELT English news corpus</p>
  <h1 style="color: #fff; margin: 0; font-size: 2.5rem; font-weight: 800; line-height: 1.1;">
    📰 Media Attention
  </h1>
  <p style="color: #86efac; margin: .65rem 0 0; font-size: 1.08rem; font-weight: 400;">
    Does news coverage drive humanitarian funding — or just follow it?
  </p>
  <p style="color: #3a6e4a; margin: .35rem 0 0; font-size: .88rem;">
    Animated choropleth 2021–2026 · Click any country for its full media timeline
  </p>
</div>
""", unsafe_allow_html=True)

# ── Metric selector ───────────────────────────────────────────────────────────
metric = st.selectbox(
    "Metric",
    ["Media Attention", "Funding Gap"],
    key="media_metric",
)

# ── Build base country list ───────────────────────────────────────────────────
media_base = (
    pd.concat([
        gap_df[["Country_ISO3"]],
        sev_df[["ISO3"]].rename(columns={"ISO3": "Country_ISO3"}),
        fts_df[["Country_ISO3"]],
    ])
    .drop_duplicates("Country_ISO3")
    .copy()
)
media_base["country_name"] = media_base["Country_ISO3"].map(name_map).fillna(
    media_base["Country_ISO3"]
)

_ANIM_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# ── Build multi-year animation dataframe ──────────────────────────────────────
if metric == "Media Attention":
    metric_col   = "media_frac_pct"
    metric_label = "Avg daily coverage (% of English news)"
    color_range  = None
    clickable    = True

    frames = []
    for yr in _ANIM_YEARS:
        sev_yr = (
            sev_df[sev_df["Year"] == yr][
                ["ISO3", "INFORM Severity Index", "INFORM Severity category", "CRISIS"]
            ].drop_duplicates("ISO3")
        )
        base_yr = (
            media_base
            .merge(sev_yr, left_on="Country_ISO3", right_on="ISO3", how="left")
            .drop(columns=["ISO3"])
        )
        media_yr = get_annual_media_map(yr)
        yr_df = base_yr.merge(media_yr, on="Country_ISO3", how="left")
        yr_df["_Year"] = str(yr)
        frames.append(yr_df)
    plot_df = pd.concat(frames, ignore_index=True)

    has_any_data = plot_df[metric_col].notna().any()
    if not has_any_data:
        st.warning("No cached media data for any year yet. Click a country to fetch its timeline.")

else:  # Funding Gap
    metric_col   = "fts_gap_pct"
    metric_label = "Funding gap — FTS total (% of requirements unfunded)"
    color_range  = [0, 100]
    clickable    = False
    has_any_data = True

    frames = []
    for yr in _ANIM_YEARS:
        sev_yr = (
            sev_df[sev_df["Year"] == yr][
                ["ISO3", "INFORM Severity Index", "INFORM Severity category", "CRISIS"]
            ].drop_duplicates("ISO3")
        )
        base_yr = (
            media_base
            .merge(sev_yr, left_on="Country_ISO3", right_on="ISO3", how="left")
            .drop(columns=["ISO3"])
        )
        fts_yr = fts_df[fts_df["Year"] == yr][["Country_ISO3", "fts_gap_pct", "fts_pct_funded"]]
        yr_df = base_yr.merge(fts_yr, on="Country_ISO3", how="left")
        yr_df["_Year"] = str(yr)
        frames.append(yr_df)
    plot_df = pd.concat(frames, ignore_index=True)

# ── Animated choropleth ───────────────────────────────────────────────────────
st.caption("Grey = no cached data · Use ▶ Play to animate year-by-year")

if has_any_data:
    event = st.plotly_chart(
        media_overview_map(
            plot_df,
            media_base,
            metric_col,
            metric_label,
            color_range=color_range,
            clickable=clickable,
            animation_col="_Year",
        ),
        use_container_width=True,
        on_select="rerun",
        selection_mode=["points"],
        key="media_map",
    )
else:
    event = None

# ── Click-to-timeseries (Media Attention only) ────────────────────────────────
if metric == "Media Attention":
    points = (event.selection or {}).get("points", []) if event else []
    if points:
        iso3         = points[0]["customdata"][0]
        country_name = points[0]["customdata"][1]
        st.subheader(f"{country_name} — Full Media Attention History")

        if force_stale or is_stale(iso3):
            if st.button("🔄 New data available — click to refresh", type="primary"):
                with st.spinner("Fetching latest data from GDELT…"):
                    _gap_fill_and_save(iso3, country_name)
                get_media_attention.clear()
                get_annual_media_map.clear()
                st.rerun()

        try:
            df_media = get_media_attention(iso3, country_name, year=2026)
            st.plotly_chart(media_timeseries(df_media, country_name), use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to load media data for {country_name}: {exc}")
    else:
        st.info("Click a country on the map to view its full media attention history.")

else:
    # Funding Gap: show summary table for the most recent year
    st.divider()
    latest_yr = _ANIM_YEARS[-1]
    fts_latest = fts_df[fts_df["Year"] == latest_yr][["Country_ISO3", "fts_gap_pct", "fts_pct_funded"]]
    tbl_df = media_base.merge(fts_latest, on="Country_ISO3", how="inner")
    top_gap = (
        tbl_df[["country_name", "fts_pct_funded", "fts_gap_pct"]]
        .dropna(subset=["fts_gap_pct"])
        .sort_values("fts_gap_pct", ascending=False)
        .head(15)
        .rename(columns={
            "country_name": "Country",
            "fts_pct_funded": "% Funded (FTS)",
            "fts_gap_pct": "Gap %",
        })
    )
    st.caption(f"Top 15 most underfunded crises — {latest_yr}")
    st.dataframe(top_gap, use_container_width=True, hide_index=True)
