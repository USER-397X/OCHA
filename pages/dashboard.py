"""Crisis Dashboard — the unified front page."""
import streamlit as st
import pandas as pd

from data import load_gap_df, load_severity_df, build_name_map, enrich_year, load_hno_core
from scoring import compute_gap_scores, format_rankings_table, fmt_usd
from charts import world_map, rankings_bar, severity_scatter, neglect_trends
from claude_chat import render_claude_chat

METHODOLOGY = """
### Gap Score

```
gap_score = (1 − coverage) × (severity / 5) × log(requirements) / log(max_req) × 100
```

| Component | Source | Rationale |
|-----------|--------|-----------|
| `1 − coverage` | CERF + CBPF allocations ÷ revised HRP requirements | Direct measure of unmet need |
| `severity / 5` | INFORM Severity Index (1–5) | Weights by humanitarian impact |
| `log(scale)` | Log-normalised requirements | Large crises rank higher, but not disproportionately |

### Structural Neglect Bonus *(optional)*

```
multiplier = 1 + 0.3 × log(consecutive_years_below_20% + 1) / log(7)
adjusted_score = gap_score × multiplier
```

A crisis underfunded for 4 consecutive years gets ~+20 % relative to one newly underfunded.
This distinguishes **structural neglect** from **acute emergencies**.

### Data Sources

- `country_year_severity_funding.csv` — CERF / CBPF allocations vs. HRP requirements per country-year
- `inform_severity_cleaned.csv` — INFORM Severity Index, crisis type, trend signal
- `hpc_hno_{year}.csv` — People-in-need figures (HNO, country level)

### Limitations & Caveats

- **Coverage is pooled-fund only** — CERF + CBPF, not total FTS funding. Ratios appear lower than
  bilateral/earmarked aid would suggest; crises with strong bilateral donors are not disadvantaged.
- Countries with documented needs but **no formal HRP** are understated.
- INFORM Severity is updated periodically and may lag sudden-onset crises.
- Sector-level gaps (health vs. food vs. shelter) within a country are not captured.
- **Gap scores support prioritisation conversations — they are not automated decisions.**
  A humanitarian coordinator should always verify context before acting on rankings.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
gap_df = load_gap_df()
sev_df = load_severity_df()
name_map = build_name_map(sev_df)

# ── Chat toggle state ─────────────────────────────────────────────────────────
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# ── Toggle button CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stButton"][id="chat-toggle-wrap"] button {
    border-radius: 20px;
    border: 1px solid #cbd5e1;
    background: transparent;
    color: #475569;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    transition: background 0.15s, color 0.15s;
}
div[data-testid="stButton"][id="chat-toggle-wrap"] button:hover {
    background: #f1f5f9;
    color: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ── Top navigation ────────────────────────────────────────────────────────────
_nc = st.columns([1, 1, 1, 1, 1.5, 0.55])
with _nc[0]:
    st.page_link("pages/dashboard.py", label="🌍 Overview", use_container_width=True)
with _nc[1]:
    st.page_link("pages/media_attention.py", label="📰 Media Attention", use_container_width=True)
with _nc[2]:
    st.page_link("pages/bias_analysis.py", label="🔍 Bias Analysis", use_container_width=True)
with _nc[3]:
    st.page_link("pages/stroopwafel.py", label="🧇 Stroopwafel EDA", use_container_width=True)
with _nc[5]:
    toggle_label = "✕ Chat" if st.session_state.chat_open else "💬 Chat"
    if st.button(toggle_label, key="chat_toggle", use_container_width=True,
                 help="Toggle AI assistant"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

# ── Layout: conditionally split main + chat ───────────────────────────────────
if st.session_state.chat_open:
    col_main, col_chat = st.columns([3, 1], gap="large")
    with col_chat:
        render_claude_chat()
else:
    col_main = st.container()

with col_main:
    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="
  background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5f 55%, #1e5799 100%);
  padding: 2rem 2.5rem 1.8rem;
  border-radius: 16px;
  margin-bottom: 1.6rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.18);
">
  <p style="
    color: #7eb8f7; font-size: .8rem; letter-spacing: .14em;
    text-transform: uppercase; margin: 0 0 .5rem; font-weight: 600;
  ">Humanitarian Intelligence Platform</p>
  <h1 style="color: #fff; margin: 0; font-size: 2.5rem; font-weight: 800; line-height: 1.1;">
    🌍 Geo-Insight
  </h1>
  <p style="color: #a8c8ee; margin: .65rem 0 0; font-size: 1.08rem; font-weight: 400;">
    Which humanitarian crises are being systematically overlooked?
  </p>
  <p style="color: #4e7faa; margin: .35rem 0 0; font-size: .88rem;">
    Ranks crises by the gap between documented need and CERF / CBPF pooled-fund coverage.
    Higher score = more overlooked.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Metric selector (drives map + rankings) ───────────────────────────────
    _METRIC_OPTIONS = {
        "A — Need Rate": ("need_rate", True),
        "B — Coverage Rate": ("coverage_rate", False),
        "C — USD / Person in Need": ("usd_per_in_need", False),
        "D — Mismatch Score": ("mismatch", True),
    }
    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = list(_METRIC_OPTIONS)[0]

    years = [y for y in sorted(gap_df["Year"].dropna().unique().astype(int), reverse=True)
             if 2021 <= y <= 2026]

    # ── Left controls + map side by side ─────────────────────────────────────
    _col_ctrl, _col_map = st.columns([1, 3])

    with _col_ctrl:
        st.caption("Metric")
        st.radio("", list(_METRIC_OPTIONS), key="selected_metric",
                 label_visibility="collapsed")
        st.divider()
        year = st.selectbox("Year", years)
        min_sev = st.slider(
            "Min. INFORM Severity", 0.0, 5.0, 2.0, 0.5,
            help="1 = low  ·  3 = high  ·  5 = very high",
        )
        top_n = st.slider("Top N crises", 5, 30, 15)
        use_neglect = st.toggle(
            "Structural neglect bonus", value=True,
            help="Upweights crises chronically below 20 % coverage.",
        )
        st.caption("OCHA FTS · CERF · CBPF · INFORM")

    # ── Compute base data + merge metric ─────────────────────────────────────
    full_scored = compute_gap_scores(gap_df, use_neglect)
    year_df = enrich_year(full_scored, sev_df, name_map, year, min_sev)

    _metric_col, _high_is_bad = _METRIC_OPTIONS[st.session_state["selected_metric"]]
    _core = st.session_state.get("stroopwafel_core") or load_hno_core()
    _core_year = _core[_core["year"] == year][["iso3", _metric_col]].dropna()
    year_df = year_df.merge(_core_year, left_on="Country_ISO3", right_on="iso3", how="left")

    _raw = year_df[_metric_col]
    _pct = _raw.rank(pct=True, method="average") * 100
    year_df["_neglect_score"] = _pct if _high_is_bad else (100 - _pct)
    sort_col, x_label = "_neglect_score", st.session_state["selected_metric"]

    # ── World map ─────────────────────────────────────────────────────────────
    with _col_map:
        st.subheader(f"World Map — {x_label}")
        map_df = year_df.nlargest(top_n, "_neglect_score")
        st.plotly_chart(world_map(map_df, color_col="_neglect_score", color_label=x_label),
                        use_container_width=True)

    # ── KPI metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Countries / crises", len(year_df))
    c2.metric("Total funding gap", fmt_usd(year_df["Funding_Gap"].sum(), 1))
    c3.metric("Avg. coverage", f"{year_df['Pct_Funded'].mean():.0f}%")
    c4.metric("< 20 % funded", int((year_df["Pct_Funded"] < 20).sum()))
    c5.metric("Avg. severity", f"{year_df['INFORM Severity Index'].mean():.1f} / 5")

    st.divider()

    # ── Rankings ──────────────────────────────────────────────────────────────

    st.subheader(f"Top {top_n} Most Overlooked Crises — {year}  ·  {x_label}")

    top = year_df.nlargest(top_n, "_neglect_score").copy()
    top["label"] = top.apply(
        lambda r: r["country_name"]
        + (f" · {str(r['CRISIS'])[:22]}" if pd.notna(r.get("CRISIS")) else ""),
        axis=1,
    )

    # Build annotation text showing the raw metric value
    def _fmt_metric(row):
        v = row.get(_metric_col)
        if pd.isna(v):
            return "n/a"
        if _metric_col in ("need_rate", "coverage_rate"):
            return f"{v*100:.1f}%"
        if _metric_col == "usd_per_in_need":
            return f"${v:,.0f}/person"
        return f"{v:.2f}"

    top["_bar_text"] = top.apply(_fmt_metric, axis=1)

    col_bar, col_tbl = st.columns([5, 6])
    with col_bar:
        st.plotly_chart(rankings_bar(top, top_n, "_neglect_score", x_label,
                                     text_series=top["_bar_text"]),
                        use_container_width=True)
    with col_tbl:
        st.dataframe(
            format_rankings_table(top),
            use_container_width=True,
            height=max(380, top_n * 32),
        )

    st.divider()

    # ── Severity scatter ──────────────────────────────────────────────────────
    st.subheader("Severity vs. Coverage  (bubble size = requirements)")
    st.plotly_chart(severity_scatter(year_df), use_container_width=True)

    # ── Structural neglect trends ─────────────────────────────────────────────
    if use_neglect:
        st.divider()
        st.subheader("Structural Neglect: Multi-Year Coverage Trends")
        st.caption(
            "Countries persistently below 20 % signal **chronic** under-resourcing, "
            "not just a point-in-time gap."
        )
        top_iso = year_df.nlargest(10, "gap_score")["Country_ISO3"].tolist()
        trend_df = full_scored[full_scored["Country_ISO3"].isin(top_iso)].copy()
        trend_df["label"] = trend_df["Country_ISO3"].map(name_map).fillna(trend_df["Country_ISO3"])
        st.plotly_chart(neglect_trends(trend_df), use_container_width=True)

    # ── Download + methodology ────────────────────────────────────────────────
    st.divider()
    dl_cols = [
        "Country_ISO3", "country_name", "Year", "CRISIS", "TYPE OF CRISIS",
        "INFORM Severity Index", "INFORM Severity category",
        "revisedRequirements", "Total_Actual_Funding", "Funding_Gap",
        "Pct_Funded", "gap_score",
    ]
    dl_df = year_df[[c for c in dl_cols if c in year_df.columns]]
    st.download_button(
        "⬇️  Download ranked crisis data (CSV)",
        data=dl_df.to_csv(index=False),
        file_name=f"crisis_gap_{year}.csv",
        mime="text/csv",
    )
    with st.expander("📐  Methodology & Limitations"):
        st.markdown(METHODOLOGY)
