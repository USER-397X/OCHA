"""
Geo-Insight: Which Crises Are Most Overlooked?
Entry point — layout and Streamlit UI only.
"""
import streamlit as st
import pandas as pd

from data import load_gap_df, load_severity_df, build_name_map, enrich_year
from scoring import compute_gap_scores, format_rankings_table, fmt_usd
from charts import world_map, rankings_bar, severity_scatter, neglect_trends

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
)

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


def main():
    st.title("🌍 Geo-Insight: Which Crises Are Most Overlooked?")
    st.caption(
        "Ranks humanitarian crises by the mismatch between documented need and "
        "CERF / CBPF pooled-fund coverage.  Higher score = more overlooked."
    )

    gap_df = load_gap_df()
    sev_df = load_severity_df()
    name_map = build_name_map(sev_df)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        years = sorted(gap_df["Year"].dropna().unique().astype(int), reverse=True)
        year = st.selectbox("Year", years)
        min_sev = st.slider("Min. INFORM Severity", 0.0, 5.0, 2.0, 0.5, help="1=low · 3=high · 5=very high")
        use_neglect = st.toggle(
            "Structural neglect bonus", value=True,
            help="Upweights crises below 20 % coverage for multiple consecutive years.",
        )
        top_n = st.slider("Top N crises to highlight", 5, 30, 15)
        st.markdown("---")
        st.markdown("**Gap Score**  \n`= (1−coverage) × severity × log(scale)`  \n\nScaled 0–100.")
        st.markdown("---")
        st.markdown("Data: OCHA FTS · CERF · CBPF · INFORM")

    # ── Compute ───────────────────────────────────────────────────────────────
    full_scored = compute_gap_scores(gap_df, use_neglect)
    year_df = enrich_year(full_scored, sev_df, name_map, year, min_sev)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Countries / crises", len(year_df))
    c2.metric("Total funding gap", fmt_usd(year_df["Funding_Gap"].sum(), 1))
    c3.metric("Avg. coverage", f"{year_df['Pct_Funded'].mean():.0f}%")
    c4.metric("< 20 % funded", int((year_df["Pct_Funded"] < 20).sum()))
    c5.metric("Avg. severity", f"{year_df['INFORM Severity Index'].mean():.1f} / 5")

    st.divider()

    # ── World map ─────────────────────────────────────────────────────────────
    st.subheader("World Map — Crisis Gap Score")
    st.plotly_chart(world_map(year_df), use_container_width=True)

    st.divider()

    # ── Rankings ──────────────────────────────────────────────────────────────
    st.subheader(f"Top {top_n} Most Overlooked Crises — {year}")

    top = year_df.nlargest(top_n, "gap_score").copy()
    top["label"] = top.apply(
        lambda r: r["country_name"]
        + (f" · {str(r['CRISIS'])[:22]}" if pd.notna(r.get("CRISIS")) else ""),
        axis=1,
    )

    col_bar, col_tbl = st.columns([5, 6])
    with col_bar:
        st.plotly_chart(rankings_bar(top, top_n), use_container_width=True)
    with col_tbl:
        st.dataframe(
            format_rankings_table(top),
            use_container_width=True,
            height=max(380, top_n * 32),
        )

    st.divider()

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.subheader("Severity vs. Coverage  (bubble size = requirements)")
    st.plotly_chart(severity_scatter(year_df), use_container_width=True)

    # ── Neglect trends ────────────────────────────────────────────────────────
    if use_neglect:
        st.divider()
        st.subheader("Structural Neglect: Multi-Year Coverage Trends")
        st.caption(
            "Countries persistently below 20 % coverage signal **chronic** under-resourcing, "
            "not just a point-in-time gap."
        )
        top_iso = year_df.nlargest(10, "gap_score")["Country_ISO3"].tolist()
        trend_df = full_scored[full_scored["Country_ISO3"].isin(top_iso)].copy()
        trend_df["label"] = trend_df["Country_ISO3"].map(name_map).fillna(trend_df["Country_ISO3"])
        st.plotly_chart(neglect_trends(trend_df), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
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

    # ── Methodology ──────────────────────────────────────────────────────────
    with st.expander("📐  Methodology & Limitations"):
        st.markdown(METHODOLOGY)


if __name__ == "__main__":
    main()
