"""Plotly figure builders — pure functions, no Streamlit calls."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scoring import SCORE_LABEL, fmt_usd


def world_map(year_df: pd.DataFrame,
              color_col: str = "gap_score",
              color_label: str = "Gap Score",
              color_scale: str = "YlOrRd",
              range_color: list[float] | None = None) -> go.Figure:
    year_df = year_df.copy()
    if range_color is None:
        range_color = [0, 100]
    if "_hover" not in year_df.columns:
        year_df["_hover"] = year_df.apply(
            lambda r: f"<b>{r.get('country_name', r['Country_ISO3'])}</b>", axis=1)
    fig = px.choropleth(
        year_df,
        locations="Country_ISO3",
        color=color_col,
        custom_data=["_hover"],
        color_continuous_scale=color_scale,
        range_color=range_color,
        labels={color_col: color_label},
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
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=color_label,
            thickness=14,
        ),
    )
    return fig


def rankings_bar(top_df: pd.DataFrame, top_n: int,
                 sort_col: str = "gap_score", x_label: str = "Gap Score",
                 text_series=None) -> go.Figure:
    sorted_df = top_df.dropna(subset=[sort_col]).sort_values(sort_col)
    x_max = max(float(sorted_df[sort_col].max()), 1e-9) if not sorted_df.empty else 100
    if text_series is not None:
        text_col = text_series.reindex(sorted_df.index)
    elif "Pct_Funded" in sorted_df.columns:
        text_col = sorted_df["Pct_Funded"].apply(lambda x: f"{x:.0f}% funded")
    else:
        text_col = None
    fig = px.bar(
        sorted_df,
        x=sort_col,
        y="label",
        orientation="h",
        color=sort_col,
        color_continuous_scale="Reds",
        range_color=[0, x_max],
        text=text_col,
        labels={sort_col: x_label, "label": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(380, top_n * 32),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=90, t=10, b=0),
        xaxis=dict(range=[0, min(x_max * 1.2, 115)]),
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


_RB_SCALE = [
    [0.0,  "#2166ac"],
    [0.25, "#92c5de"],
    [0.5,  "#f7f7f7"],
    [0.75, "#f4a582"],
    [1.0,  "#d6604d"],
]

_GREY_NO_DATA = "#cccccc"   # in dataset but no cached media data
_GEO_STYLE = dict(
    showframe=False,
    showcoastlines=True, coastlinecolor="#aaa",
    showland=True,       landcolor="#f2f2f2",   # white = not in dataset at all
    showocean=True,      oceancolor="#dceef7",
    showlakes=False,
    projection_type="natural earth",
)


def media_overview_map(
    plot_df: pd.DataFrame,
    all_iso3_df: pd.DataFrame,          # every known ISO3 → grey "no data" background
    metric_col: str,
    metric_label: str,
    color_range: list[float] | None = None,
    clickable: bool = False,
    animation_col: str | None = None,
) -> go.Figure:
    """Two-trace choropleth: grey background for all known countries, coloured overlay for those with data.

    *all_iso3_df* must have columns Country_ISO3 and country_name.
    *plot_df* rows with NaN *metric_col* are shown grey (background trace); rows with
    actual values are coloured blue→red.  Countries absent from both DataFrames show
    as land colour (white).
    """
    df = plot_df.copy()

    # --- colour range from actual data (no sentinel) ---
    actual = df[metric_col].dropna()
    if color_range is None:
        if actual.empty:
            raise RuntimeError(
                f"No data for '{metric_col}' — cannot render map. "
                "Run scripts/prefetch_media.py first."
            )
        vmax = max(float(actual.quantile(0.99)), 1e-9)
        color_range = [0.0, vmax]
    else:
        vmax = color_range[1]

    # --- hover text (NaN → "no data", never shows raw numbers) ---
    def _hover(r):
        sev = r.get("INFORM Severity Index", float("nan"))
        sev_str = f"{sev:.1f}" if pd.notna(sev) else "—"
        cat = r.get("INFORM Severity category", "")
        cat_str = f" ({cat})" if pd.notna(cat) and cat else ""
        crisis = r.get("CRISIS", "")
        crisis_str = f"{crisis}<br>" if pd.notna(crisis) and crisis else ""
        val = r[metric_col]
        if pd.isna(val):
            val_str = "no data"
        elif "media" in metric_col:
            val_str = f"{val:.3f}%"
        else:
            val_str = f"{val:.1f}%"
        click_hint = "<br><i>Click for country profile</i>" if clickable else ""
        return (
            f"<b>{r['country_name']}</b><br>"
            f"{crisis_str}"
            f"Severity: {sev_str}{cat_str}<br>"
            f"{metric_label}: {val_str}"
            f"{click_hint}"
        )

    df["_hover"] = df.apply(_hover, axis=1)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trace 0 — grey background: all known ISO3s (including those with no data)
    grey_hover = all_iso3_df.apply(
        lambda r: f"<b>{r['country_name']}</b><br>No cached data — click to fetch",
        axis=1,
    )
    if animation_col:
        # Repeat grey trace once per frame so animation has something to show
        frame_labels = sorted(df[animation_col].dropna().unique())
        grey_rows = []
        for lbl in frame_labels:
            tmp = all_iso3_df.copy()
            tmp[animation_col] = lbl
            tmp["_ghover"] = grey_hover.values
            grey_rows.append(tmp)
        grey_df = pd.concat(grey_rows, ignore_index=True)
        fig.add_trace(go.Choropleth(
            locations=grey_df[grey_df[animation_col] == frame_labels[-1]]["Country_ISO3"],
            z=[0] * len(all_iso3_df),
            colorscale=[[0, _GREY_NO_DATA], [1, _GREY_NO_DATA]],
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=all_iso3_df["country_name"].apply(
                lambda n: f"<b>{n}</b><br>No cached data — click to fetch"
            ),
            marker_line_color="#aaa", marker_line_width=0.4,
            name="no data",
        ))
    else:
        fig.add_trace(go.Choropleth(
            locations=all_iso3_df["Country_ISO3"],
            z=[0] * len(all_iso3_df),
            colorscale=[[0, _GREY_NO_DATA], [1, _GREY_NO_DATA]],
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=grey_hover,
            marker_line_color="#aaa", marker_line_width=0.4,
            name="no data",
        ))

    # Trace 1 — coloured overlay: rows with actual metric values
    if animation_col:
        data_df = df  # full multi-year df; animation splits by animation_col
        kwargs_anim: dict = dict(animation_frame=animation_col,
                                 category_orders={animation_col: frame_labels})
    else:
        data_df = df
        kwargs_anim = {}

    data_sub = data_df.dropna(subset=[metric_col])
    fig2 = px.choropleth(
        data_sub,
        locations="Country_ISO3",
        color=metric_col,
        custom_data=["Country_ISO3", "country_name", "_hover"],
        color_continuous_scale=_RB_SCALE,
        range_color=color_range,
        labels={metric_col: metric_label},
        **kwargs_anim,
    )
    for tr in fig2.data:
        fig.add_trace(tr)

    # Carry animation frames from fig2 into fig
    if animation_col and fig2.frames:
        # Prepend an empty grey-trace delta to each frame so trace indices stay consistent
        new_frames = []
        for frame in fig2.frames:
            new_frames.append(go.Frame(
                data=[go.Choropleth()] + list(frame.data),  # trace 0 unchanged
                name=frame.name,
                traces=[0, 1],
            ))
        fig.frames = new_frames

    fig.update_traces(hovertemplate="%{customdata[2]}<extra></extra>",
                      selector={"type": "choropleth", "showscale": True})
    fig.update_layout(
        geo=_GEO_STYLE,
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=metric_label,
            thickness=14,
            tickmode="array",
            tickvals=[0, vmax * 0.25, vmax * 0.5, vmax * 0.75, vmax],
            ticktext=["0", f"{vmax*0.25:.3g}", f"{vmax*0.5:.3g}",
                      f"{vmax*0.75:.3g}", f"{vmax:.3g}"],
        ),
        showlegend=False,
    )

    if animation_col and fig2.frames:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200
        # Default to last frame
        last_idx = len(fig2.frames) - 1
        if fig.layout.sliders:
            fig.layout.sliders[0].active = last_idx
        # Sync base data to last frame
        if fig.frames and fig.frames[-1].data:
            for i, fdata in enumerate(fig.frames[-1].data):
                if i < len(fig.data) and fdata:
                    fig.data[i].update(fdata)

    return fig


def media_attention_map(year_df: pd.DataFrame) -> go.Figure:
    """Legacy wrapper — kept for backwards compat; use media_overview_map directly."""
    return media_overview_map(year_df, "gap_score", "Gap Score",
                               color_range=[0, 100], clickable=True)


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
