"""Plotly figure builders — pure functions, no Streamlit calls."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scoring import SCORE_LABEL, fmt_usd


def world_map(year_df: pd.DataFrame) -> go.Figure:
    year_df = year_df.copy()
    year_df["_hover"] = year_df.apply(
        lambda r: (
            f"<b>{r['country_name']}</b><br>"
            f"{r['CRISIS'] if pd.notna(r.get('CRISIS')) else ''}<br>"
            f"Type: {r['TYPE OF CRISIS'] if pd.notna(r.get('TYPE OF CRISIS')) else '—'}<br>"
            f"Severity: {r['INFORM Severity Index']:.1f}"
            f" ({r['INFORM Severity category'] if pd.notna(r.get('INFORM Severity category')) else ''})<br>"
            + (f"People in need: {r['In Need']/1e6:.1f}M<br>" if pd.notna(r.get("In Need")) else "")
            + f"Requirements: {fmt_usd(r['revisedRequirements'])}<br>"
            f"Coverage: {r['Pct_Funded']:.1f}%<br>"
            f"<b>Gap Score: {r['gap_score']:.1f}</b>"
        ),
        axis=1,
    )
    fig = px.choropleth(
        year_df,
        locations="Country_ISO3",
        color="gap_score",
        custom_data=["_hover"],
        color_continuous_scale="YlOrRd",
        range_color=[0, 100],
        labels={"gap_score": SCORE_LABEL},
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#aaa",
            showland=True, landcolor="#f2f2f2",
            showocean=True, oceancolor="#dceef7",
            showlakes=False,
            projection_type="natural earth",
        ),
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="Gap Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50 (Med)", "75", "100 (High)"],
            thickness=14,
        ),
    )
    return fig


def rankings_bar(top_df: pd.DataFrame, top_n: int) -> go.Figure:
    sorted_df = top_df.sort_values("gap_score")
    fig = px.bar(
        sorted_df,
        x="gap_score",
        y="label",
        orientation="h",
        color="gap_score",
        color_continuous_scale="Reds",
        range_color=[0, 100],
        text=sorted_df["Pct_Funded"].apply(lambda x: f"{x:.0f}% funded"),
        labels={"gap_score": "Gap Score", "label": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(380, top_n * 32),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=90, t=10, b=0),
        xaxis=dict(range=[0, 115]),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def severity_scatter(year_df: pd.DataFrame) -> go.Figure:
    df = year_df.copy()
    df["req_M"] = df["revisedRequirements"] / 1e6
    fig = px.scatter(
        df,
        x="Pct_Funded",
        y="INFORM Severity Index",
        size="req_M",
        color="gap_score",
        color_continuous_scale="YlOrRd",
        range_color=[0, 100],
        hover_name="country_name",
        hover_data={
            "Pct_Funded": ":.1f",
            "INFORM Severity Index": ":.2f",
            "req_M": ":.0f",
            "gap_score": ":.1f",
            "CRISIS": True,
        },
        labels={
            "Pct_Funded": "Coverage (%)",
            "req_M": "Requirements ($M)",
            "gap_score": SCORE_LABEL,
        },
        size_max=60,
    )
    fig.add_hline(y=3.5, line_dash="dash", line_color="#888",
                  annotation_text="High severity (3.5)", annotation_position="top right")
    fig.add_vline(x=50, line_dash="dash", line_color="#888",
                  annotation_text="50 % covered", annotation_position="top left")
    fig.add_annotation(x=8, y=4.85, text="⚠️  Most Overlooked",
                       showarrow=False, font=dict(color="#c0392b", size=13, family="Arial Black"))
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def media_attention_map(year_df: pd.DataFrame) -> go.Figure:
    df = year_df.copy()
    df["_hover"] = df.apply(
        lambda r: (
            f"<b>{r['country_name']}</b><br>"
            f"{r['CRISIS'] if pd.notna(r.get('CRISIS')) else ''}<br>"
            f"Severity: {r['INFORM Severity Index']:.1f}"
            f" ({r['INFORM Severity category'] if pd.notna(r.get('INFORM Severity category')) else ''})<br>"
            f"<i>Click to explore media attention</i>"
        ),
        axis=1,
    )
    fig = px.choropleth(
        df,
        locations="Country_ISO3",
        color="gap_score",
        custom_data=["Country_ISO3", "country_name", "_hover"],
        color_continuous_scale="YlOrRd",
        range_color=[0, 100],
        labels={"gap_score": SCORE_LABEL},
    )
    fig.update_traces(hovertemplate="%{customdata[2]}<extra></extra>")
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#aaa",
            showland=True, landcolor="#f2f2f2",
            showocean=True, oceancolor="#dceef7",
            showlakes=False,
            projection_type="natural earth",
        ),
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="Gap Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0", "25", "50 (Med)", "75", "100 (High)"],
            thickness=14,
        ),
    )
    return fig


def media_timeseries(df: pd.DataFrame, country_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["frac_pct"],
        mode="lines",
        line=dict(color="#4a90d9", width=1),
        opacity=0.35,
        name="Daily",
        hovertemplate="%{x|%d %b %Y}: %{y:.4f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_7d"],
        mode="lines",
        line=dict(color="#1a5fa8", width=2),
        name="7-day avg",
        hovertemplate="%{x|%d %b %Y}: %{y:.4f}%<extra></extra>",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date",
        yaxis_title="% of English news coverage",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def neglect_trends(trend_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        trend_df.sort_values("Year"),
        x="Year",
        y="Pct_Funded",
        color="label",
        markers=True,
        labels={"Pct_Funded": "Coverage (%)", "label": "Country"},
    )
    fig.add_hline(y=20, line_dash="dot", line_color="#c0392b",
                  annotation_text="20 % minimum threshold")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    return fig
