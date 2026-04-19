"""Bias analysis charts and render function for the Geo-Insight dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scoring import fmt_usd

PROJECT_ROOT = Path(__file__).parent

DATA = PROJECT_ROOT / "data"

REGION_COLORS = {
    "Africa": "#e74c3c",
    "Middle east": "#e67e22",
    "Asia": "#3498db",
    "Americas": "#2ecc71",
    "Europe": "#9b59b6",
    "Pacific": "#1abc9c",
}

CRISIS_COLORS = {
    "Conflict / Complex": "#c0392b",
    "Displacement": "#e67e22",
    "Natural Disaster": "#3498db",
    "Food Insecurity": "#f39c12",
    "Political / Economic": "#9b59b6",
    "Other": "#95a5a6",
    "Unknown": "#bdc3c7",
}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_bias_data():
    cluster = pd.read_csv(DATA / "fts_requirements_funding_cluster_global.csv")
    cluster["requirements"] = pd.to_numeric(cluster["requirements"], errors="coerce")
    cluster["funding"] = pd.to_numeric(cluster["funding"], errors="coerce")
    cluster["year"] = pd.to_numeric(cluster["year"], errors="coerce")

    fts_in = pd.read_csv(DATA / "fts_incoming_funding_global.csv")
    fts_in["amountUSD"] = pd.to_numeric(fts_in["amountUSD"], errors="coerce")

    return cluster, fts_in


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_crisis(t: str) -> str:
    if pd.isna(t):
        return "Unknown"
    t = str(t).lower()
    if any(k in t for k in ("conflict", "violence", "complex")):
        return "Conflict / Complex"
    if "displace" in t:
        return "Displacement"
    if any(k in t for k in ("drought", "flood", "cyclone", "earthquake")):
        return "Natural Disaster"
    if "food" in t:
        return "Food Insecurity"
    if any(k in t for k in ("political", "economic")):
        return "Political / Economic"
    return "Other"


def _norm_cluster(c: str) -> str | None:
    if pd.isna(c):
        return None
    c = str(c).lower()
    if "food" in c:
        return "Food Security"
    if "nutrition" in c:
        return "Nutrition"
    if "health" in c:
        return "Health"
    if "wash" in c or ("water" in c and "sanit" in c):
        return "WASH"
    if "shelter" in c or "nfi" in c:
        return "Shelter / NFI"
    if "education" in c:
        return "Education"
    if "protection" in c:
        return "Protection"
    if "livelihoods" in c or "livelihood" in c:
        return "Livelihoods"
    if "cash" in c or "multi" in c:
        return "Multi-purpose Cash"
    if "refugee" in c:
        return "Refugee Response"
    if "logist" in c or "coord" in c or "telecom" in c:
        return "Coordination"
    return None


def _enrich_for_bias(
    full_scored: pd.DataFrame,
    sev_df: pd.DataFrame,
    name_map: dict,
) -> pd.DataFrame:
    sev = sev_df.copy()
    sev["region"] = sev["Regions"].str.split(",").str[0].str.strip()
    join = (
        sev[["ISO3", "Year", "TYPE OF CRISIS", "region"]]
        .rename(columns={"ISO3": "Country_ISO3"})
        .drop_duplicates(subset=["Country_ISO3", "Year"])
    )
    df = full_scored.merge(join, on=["Country_ISO3", "Year"], how="left")
    df["crisis_group"] = df["TYPE OF CRISIS"].apply(_classify_crisis)
    df["country_name"] = df["Country_ISO3"].map(name_map).fillna(df["Country_ISO3"])
    return df


def _compute_scorecard(df: pd.DataFrame, cluster_df: pd.DataFrame) -> dict:
    latest = df.sort_values("Year").groupby("Country_ISO3").last().reset_index()
    high_sev = latest.dropna(subset=["INFORM Severity Index"])
    high_sev = high_sev[high_sev["INFORM Severity Index"] >= 3.5]
    forgotten_n = int((high_sev["coverage"] < 0.20).sum())
    forgotten_pct = forgotten_n / max(len(high_sev), 1) * 100

    reg_medians = df.dropna(subset=["region"]).groupby("region")["coverage"].median()
    worst_region = reg_medians.idxmin() if len(reg_medians) else "N/A"
    best_region = reg_medians.idxmax() if len(reg_medians) else "N/A"
    worst_region_cov = float(reg_medians.get(worst_region, 0)) * 100
    regional_gap = float(reg_medians.get(best_region, 0) - reg_medians.get(worst_region, 0)) * 100

    valid = df.dropna(subset=["INFORM Severity Index", "coverage"])
    sev_corr = float(np.corrcoef(valid["INFORM Severity Index"], valid["coverage"])[0, 1]) if len(valid) > 2 else 0.0

    below20 = df[df["coverage"] < 0.20].groupby("Country_ISO3").size()
    if len(below20):
        worst_iso = below20.idxmax()
        worst_rows = df[df["Country_ISO3"] == worst_iso]["country_name"]
        worst_country = worst_rows.iloc[0] if len(worst_rows) else worst_iso
        worst_country_years = int(below20.max())
    else:
        worst_country, worst_country_years = "N/A", 0

    c2 = cluster_df[cluster_df["year"].fillna(0) >= 2020].copy()
    c2["cn"] = c2["cluster"].apply(_norm_cluster)
    c2 = c2.dropna(subset=["cn"])
    sec = c2.groupby("cn").agg(r=("requirements", "sum"), f=("funding", "sum"))
    sec["cov"] = sec["f"] / sec["r"].clip(lower=1)
    worst_sector = sec["cov"].idxmin() if len(sec) else "N/A"
    worst_sector_cov = float(sec.loc[worst_sector, "cov"]) * 100 if worst_sector != "N/A" else 0.0

    return dict(
        forgotten_n=forgotten_n,
        forgotten_pct=forgotten_pct,
        worst_region=worst_region,
        worst_region_cov=worst_region_cov,
        regional_gap=regional_gap,
        sev_corr=sev_corr,
        worst_country=worst_country,
        worst_country_years=worst_country_years,
        worst_sector=worst_sector,
        worst_sector_cov=worst_sector_cov,
    )


# ── Chart functions ───────────────────────────────────────────────────────────

def forgotten_quadrant(df: pd.DataFrame) -> go.Figure:
    latest = df.sort_values("Year").groupby("Country_ISO3").last().reset_index()
    plot = latest.dropna(subset=["INFORM Severity Index", "coverage", "region"]).copy()
    plot["req_M"] = plot["revisedRequirements"].clip(lower=1e5) / 1e6
    plot["marker_size"] = np.sqrt(plot["req_M"].clip(1)) * 2.5

    fig = go.Figure()

    for x0, x1, y0, y1, color in [
        (-2, 20, 3.5, 5.7, "rgba(192,57,43,0.10)"),
        (20, 112, 3.5, 5.7, "rgba(243,156,18,0.05)"),
        (-2, 20, 0.8, 3.5, "rgba(52,152,219,0.04)"),
        (20, 112, 0.8, 3.5, "rgba(46,204,113,0.04)"),
    ]:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line_width=0, layer="below",
        )

    for region, grp in plot.groupby("region"):
        fig.add_trace(go.Scatter(
            x=grp["coverage"] * 100,
            y=grp["INFORM Severity Index"],
            mode="markers",
            name=region,
            marker=dict(
                size=grp["marker_size"],
                color=REGION_COLORS.get(region, "#95a5a6"),
                opacity=0.75,
                line=dict(width=0.6, color="#555"),
            ),
            customdata=np.stack([
                grp["country_name"].fillna("").values,
                (grp["coverage"] * 100).round(1).astype(str).values,
                grp["INFORM Severity Index"].fillna(0).round(2).astype(str).values,
                grp["revisedRequirements"].apply(fmt_usd).values,
                grp["crisis_group"].fillna("Unknown").values,
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Coverage: %{customdata[1]}%<br>"
                "Severity: %{customdata[2]}<br>"
                "Requirements: %{customdata[3]}<br>"
                "Type: %{customdata[4]}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=3.5, line_dash="dash", line_color="#c0392b", line_width=1.5,
                  annotation_text="High severity (3.5)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#c0392b"))
    fig.add_vline(x=20, line_dash="dash", line_color="#c0392b", line_width=1.5,
                  annotation_text="20% funded",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#c0392b"))

    forgotten = plot[(plot["coverage"] < 0.20) & (plot["INFORM Severity Index"] >= 3.5)]
    for _, row in forgotten.nlargest(14, "INFORM Severity Index").iterrows():
        fig.add_annotation(
            x=row["coverage"] * 100,
            y=row["INFORM Severity Index"],
            text=row["country_name"],
            showarrow=True, arrowhead=2, arrowsize=0.7,
            arrowwidth=1, arrowcolor="#c0392b",
            ax=28, ay=-22,
            font=dict(size=9, color="#c0392b"),
        )

    n_forgotten = len(forgotten)
    fig.add_annotation(x=3, y=5.55, text=f"⚠️  FORGOTTEN  ({n_forgotten} countries)",
                       showarrow=False,
                       font=dict(size=13, color="#c0392b", family="Arial Black"))
    fig.add_annotation(x=65, y=5.55, text="✓  Severe but funded",
                       showarrow=False, font=dict(size=12, color="#e67e22"))

    fig.update_layout(
        height=560,
        xaxis=dict(title="Pooled-fund coverage (%, most recent year)", range=[-2, 112]),
        yaxis=dict(title="INFORM Severity Index (1–5)", range=[0.8, 5.7]),
        legend=dict(title="Region"),
        margin=dict(l=0, r=10, t=10, b=10),
    )
    return fig


def neglect_heatmap(df: pd.DataFrame, name_map: dict) -> go.Figure:
    df = df.copy()
    df["cov_pct"] = df["coverage"] * 100
    pivot = df.pivot_table(index="Country_ISO3", columns="Year", values="cov_pct", aggfunc="mean")

    region_first = df.groupby("Country_ISO3")["region"].first()
    avg_cov = df.groupby("Country_ISO3")["cov_pct"].mean()
    order = (
        pd.DataFrame({"region": region_first, "avg": avg_cov})
          .sort_values(["region", "avg"])
          .index
    )
    pivot = pivot.reindex(order)
    pivot.index = [name_map.get(iso, iso) for iso in pivot.index]

    z = pivot.values
    text = [
        [f"{int(round(v))}%" if not np.isnan(v) else "" for v in row]
        for row in z
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#c62828"], [0.18, "#ef5350"], [0.33, "#ff8a65"],
                    [0.5, "#fff176"], [0.7, "#aed581"], [1.0, "#2e7d32"]],
        zmin=0, zmax=60,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate="%{y} (%{x}): %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title="Coverage",
            tickvals=[0, 20, 40, 60],
            ticktext=["0%", "20%", "40%", "≥60%"],
            thickness=14,
        ),
    ))
    fig.update_layout(
        height=max(560, len(pivot) * 18),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def geographic_bias(df: pd.DataFrame) -> go.Figure:
    plot = df.dropna(subset=["region", "coverage"]).copy()
    plot["cov_pct"] = plot["coverage"] * 100
    order = (
        plot.groupby("region")["cov_pct"]
            .median()
            .sort_values()
            .index.tolist()
    )
    fig = px.box(
        plot, x="region", y="cov_pct",
        color="region",
        category_orders={"region": order},
        color_discrete_map=REGION_COLORS,
        points="outliers",
        hover_name="country_name",
        labels={"cov_pct": "Pooled-fund coverage (%)", "region": ""},
    )
    fig.add_hline(y=20, line_dash="dash", line_color="#c0392b",
                  annotation_text="20% threshold",
                  annotation_position="top right")
    fig.update_layout(
        showlegend=False, height=420,
        margin=dict(l=0, r=0, t=10, b=10),
    )
    return fig


def crisis_type_bias(df: pd.DataFrame) -> go.Figure:
    stats = (
        df.groupby("crisis_group")["coverage"]
          .agg(median="median", n="count")
          .reset_index()
          .sort_values("median")
    )
    stats["med_pct"] = stats["median"] * 100
    stats["label"] = stats.apply(lambda r: f"{r['med_pct']:.1f}%  (n={int(r['n'])})", axis=1)
    fig = px.bar(
        stats, x="med_pct", y="crisis_group", orientation="h",
        color="crisis_group",
        color_discrete_map=CRISIS_COLORS,
        text="label",
        labels={"med_pct": "Median coverage (%)", "crisis_group": ""},
    )
    fig.add_vline(x=20, line_dash="dash", line_color="#c0392b",
                  annotation_text="20% threshold",
                  annotation_position="top right")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, height=420,
        xaxis=dict(range=[0, stats["med_pct"].max() + 15]),
        margin=dict(l=0, r=130, t=10, b=10),
    )
    return fig


def severity_alignment(df: pd.DataFrame) -> go.Figure:
    plot = df.dropna(subset=["INFORM Severity Index", "coverage", "region"]).copy()
    plot["cov_pct"] = plot["coverage"] * 100

    x = plot["INFORM Severity Index"].values
    y = plot["cov_pct"].values
    m, b = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    fig = go.Figure()
    for region, grp in plot.groupby("region"):
        fig.add_trace(go.Scatter(
            x=grp["INFORM Severity Index"],
            y=grp["cov_pct"],
            mode="markers",
            name=region,
            marker=dict(size=8, color=REGION_COLORS.get(region, "#95a5a6"),
                        opacity=0.5, line=dict(width=0.4, color="#333")),
            text=grp["country_name"],
            hovertemplate="<b>%{text}</b><br>Severity: %{x:.2f}<br>Coverage: %{y:.1f}%<extra></extra>",
        ))

    x_line = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines",
        line=dict(color="black", width=2.5, dash="dot"),
        name=f"Trend (r={r:.2f})",
        hoverinfo="skip",
    ))

    direction = "↗ positive (funding tracks need)" if m > 0 else "↘ NEGATIVE — funding ignores need"
    box_color = "#27ae60" if m > 0 else "#c0392b"
    fig.add_annotation(
        x=0.97, y=0.95, xref="paper", yref="paper",
        text=f"<b>r = {r:.2f}</b><br>{direction}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        font=dict(size=11, color=box_color),
        bordercolor=box_color, borderwidth=1.5, borderpad=8,
        align="center",
    )
    fig.add_hline(y=20, line_dash="dash", line_color="#c0392b", line_width=1, opacity=0.4)
    fig.update_layout(
        height=420,
        xaxis=dict(title="INFORM Severity Index (1–5)"),
        yaxis=dict(title="Pooled-fund coverage (%)"),
        legend=dict(title="Region"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    return fig


def sector_gaps(cluster_df: pd.DataFrame) -> go.Figure:
    c2 = cluster_df[cluster_df["year"].fillna(0) >= 2020].copy()
    c2["cluster_norm"] = c2["cluster"].apply(_norm_cluster)
    c2 = c2.dropna(subset=["cluster_norm"])

    agg = (
        c2.groupby("cluster_norm")
          .agg(total_req=("requirements", "sum"), total_fund=("funding", "sum"))
          .reset_index()
    )
    agg["coverage"] = (agg["total_fund"] / agg["total_req"].clip(lower=1) * 100).clip(0, 100)
    agg["funded_B"] = agg["total_fund"] / 1e9
    agg["gap_B"] = (agg["total_req"] - agg["total_fund"]).clip(lower=0) / 1e9
    agg = agg.sort_values("coverage")

    bar_colors = [
        "#c0392b" if v < 15 else "#e67e22" if v < 30 else "#f39c12" if v < 50 else "#27ae60"
        for v in agg["coverage"]
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Coverage rate (%)", "Funded vs. Funding Gap ($B, 2020–2025)"],
        column_widths=[0.42, 0.58],
    )
    fig.add_trace(go.Bar(
        x=agg["coverage"], y=agg["cluster_norm"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0f}%" for v in agg["coverage"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=agg["funded_B"], y=agg["cluster_norm"],
        orientation="h",
        name="Funded", marker_color="#27ae60", opacity=0.85,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=agg["gap_B"], y=agg["cluster_norm"],
        orientation="h",
        name="Gap", marker_color="#c0392b", opacity=0.7,
    ), row=1, col=2)

    fig.add_vline(x=20, line_dash="dash", line_color="#c0392b", row=1, col=1)
    fig.update_xaxes(title_text="Coverage (%)", row=1, col=1)
    fig.update_xaxes(title_text="$ Billion", row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        height=460, barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=0, r=80, t=55, b=10),
    )
    return fig


def donor_region_chart(fts_in: pd.DataFrame, iso_region: dict) -> go.Figure:
    fts = fts_in.dropna(subset=["amountUSD"]).copy()
    fts["dest_iso3"] = fts["destLocations"].astype(str).str.extract(r"([A-Z]{3})")
    fts["dest_region"] = fts["dest_iso3"].map(iso_region)
    fts = fts.dropna(subset=["dest_region", "srcOrganization"])

    top_donors = fts.groupby("srcOrganization")["amountUSD"].sum().nlargest(10).index
    fts = fts[fts["srcOrganization"].isin(top_donors)]

    pivot = (
        fts.groupby(["srcOrganization", "dest_region"])["amountUSD"]
           .sum()
           .reset_index()
           .pivot(index="srcOrganization", columns="dest_region", values="amountUSD")
           .fillna(0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    short_names = {
        "Japan, Government of": "Japan",
        "European Commission's Humanitarian Aid and Civil Protection Department": "EU ECHO",
        "United States of America, Government of": "USA",
        "Germany, Government of": "Germany",
        "Norway, Government of": "Norway",
        "Switzerland, Government of": "Switzerland",
        "United Kingdom, Government of": "UK",
        "Canada, Government of": "Canada",
        "United States Department of State": "US Dept of State",
        "Qatar Charity": "Qatar Charity",
    }
    pct.index = [short_names.get(d, d[:22]) for d in pct.index]
    # sort by Africa share descending
    region_col_order = [r for r in ["Africa", "Asia", "Americas", "Middle east", "Europe", "Pacific"]
                        if r in pct.columns]
    pct = pct[region_col_order]
    pct = pct.sort_values(region_col_order[0] if region_col_order else pct.columns[0], ascending=False)

    text_vals = [[f"{int(round(v))}%" for v in row] for row in pct.values]

    fig = go.Figure(go.Heatmap(
        z=pct.values,
        x=pct.columns.tolist(),
        y=pct.index.tolist(),
        colorscale="Blues",
        zmin=0, zmax=80,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="%{y} → %{x}: %{z:.0f}%<extra></extra>",
        colorbar=dict(title="% of<br>donor total", thickness=14),
    ))
    fig.update_layout(
        height=360,
        xaxis=dict(side="top", title="Destination region"),
        yaxis=dict(title="Donor", tickfont=dict(size=11)),
        margin=dict(l=0, r=0, t=50, b=10),
    )
    return fig


# ── Main render function ──────────────────────────────────────────────────────

def render_bias_analysis(
    full_scored: pd.DataFrame,
    sev_df: pd.DataFrame,
    name_map: dict,
) -> None:
    cluster_df, fts_in = load_bias_data()
    df = _enrich_for_bias(full_scored, sev_df, name_map)
    iso_region = (
        df[["Country_ISO3", "region"]].dropna()
          .drop_duplicates()
          .set_index("Country_ISO3")["region"]
          .to_dict()
    )

    sc = _compute_scorecard(df, cluster_df)

    # ── Bias Scorecard ────────────────────────────────────────────────────
    st.markdown("#### 5 Bias Signals — 2020–2025")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Forgotten crises",
        f"{sc['forgotten_n']} countries",
        f"{sc['forgotten_pct']:.0f}% of high-severity",
        delta_color="inverse",
    )
    c2.metric(
        "Worst-funded region",
        sc["worst_region"],
        f"{sc['worst_region_cov']:.1f}% median  (−{sc['regional_gap']:.0f}pp gap)",
        delta_color="inverse",
    )
    corr_label = (
        "positive (aligned)" if sc["sev_corr"] > 0.3
        else "NEGATIVE — alarming" if sc["sev_corr"] < -0.1
        else "weak — misaligned"
    )
    c3.metric(
        "Severity–funding r",
        f"{sc['sev_corr']:.2f}",
        corr_label,
        delta_color="normal" if sc["sev_corr"] > 0.3 else "inverse",
    )
    c4.metric(
        "Most neglected country",
        sc["worst_country"],
        f"Below 20% for {sc['worst_country_years']} of 6 years",
        delta_color="inverse",
    )
    c5.metric(
        "Most underfunded sector",
        sc["worst_sector"],
        f"{sc['worst_sector_cov']:.1f}% coverage",
        delta_color="inverse",
    )

    st.divider()

    # ── Hero: Forgotten Quadrant ──────────────────────────────────────────
    st.subheader("⚠️  The Forgotten Crises Quadrant")
    st.caption(
        "Each bubble is one country (most recent year available). **Top-left zone = severe but unfunded** — "
        "the UN's structural blind spot. Bubble size = humanitarian requirements."
    )
    st.plotly_chart(forgotten_quadrant(df), use_container_width=True)

    st.divider()

    # ── Structural neglect heatmap ────────────────────────────────────────
    st.subheader("🗓️  Structural Neglect — Coverage by Country × Year")
    st.caption(
        "Red = critically underfunded.  Sorted by region then average coverage — "
        "**persistent red rows reveal chronic neglect**, not just point-in-time gaps."
    )
    st.plotly_chart(neglect_heatmap(df, name_map), use_container_width=True)

    st.divider()

    # ── Geographic + Crisis Type side by side ─────────────────────────────
    col_geo, col_ct = st.columns(2, gap="large")
    with col_geo:
        st.subheader("🌍  Geographic Bias")
        st.caption("Coverage distribution by region (2020–2025). Lower box = more underfunded.")
        st.plotly_chart(geographic_bias(df), use_container_width=True)
    with col_ct:
        st.subheader("⚡  Crisis-Type Bias")
        st.caption("Median pooled-fund coverage by crisis classification.")
        st.plotly_chart(crisis_type_bias(df), use_container_width=True)

    st.divider()

    # ── Severity alignment ────────────────────────────────────────────────
    st.subheader("📐  Is Funding Proportional to Need?")
    st.caption(
        "A positive slope means severity predicts more funding — the system works as intended. "
        "A flat or negative slope is evidence of **structural misalignment**: the most severe crises are not the best-funded."
    )
    st.plotly_chart(severity_alignment(df), use_container_width=True)

    st.divider()

    # ── Sector gaps ───────────────────────────────────────────────────────
    st.subheader("📦  Sector Funding Gaps")
    st.caption(
        "Aggregate 2020–2025 cluster-level coverage. "
        "Red = critically underfunded (<15%). Which sectors are chronically under-resourced?"
    )
    st.plotly_chart(sector_gaps(cluster_df), use_container_width=True)

    st.divider()

    # ── Donor geography ───────────────────────────────────────────────────
    st.subheader("💰  Donor Geographic Concentration")
    st.caption(
        "What % of each top donor's total FTS contribution goes to each region? "
        "**Strong column concentration = political alignment rather than need-based allocation.**"
    )
    st.plotly_chart(donor_region_chart(fts_in, iso_region), use_container_width=True)
