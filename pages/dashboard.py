"""Crisis Dashboard — the unified front page."""
import streamlit as st
import pandas as pd

from data import (load_severity_df, build_name_map, load_hno_core,
                  load_fts_funding, load_overlooked, load_alignment_map)
from charts import world_map
from claude_chat import render_claude_chat

# ── Data ──────────────────────────────────────────────────────────────────────
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
_nc = st.columns([2, 1, 1, 2, 0.55])
with _nc[1]:
    st.page_link("pages/dashboard.py", label="Overview", use_container_width=True)
with _nc[2]:
    st.page_link("pages/stroopwafel.py", label="Stroopwafel Notebook", use_container_width=True)
with _nc[4]:
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
    Geo-Insight
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

    # ── Metric selector (fixed bottom-right) ─────────────────────────────────
    _CATEGORIES = [
        "Invisible Crises",
        "Documented but Unplanned",
        "Planned but Unfunded",
        "Allocated Funding by Category",
    ]
    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = _CATEGORIES[0]

    _picker = st.container(key="cat_picker")
    st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > [data-testid="stVerticalBlock"] [data-testid="stSelectbox"]) {
    position: fixed !important;
    bottom: 24px;
    right: 24px;
    z-index: 999999;
    width: 260px;
    background: white;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    border: none !important;
    padding: 4px 8px;
}
</style>
""", unsafe_allow_html=True)
    with _picker:
        st.selectbox("", _CATEGORIES, key="selected_metric", label_visibility="collapsed")

    # ── Build map data per category ───────────────────────────────────────────
    _YEAR = 2025
    selected = st.session_state["selected_metric"]

    if selected == "Invisible Crises":
        df = load_overlooked(_YEAR)
        df["_hover"] = df.apply(
            lambda r: (
                f"<b>{r['country_name']}</b><br>"
                f"Severity: {r['INFORM Severity Index']:.1f}"
                f" ({r['INFORM Severity category'] if pd.notna(r.get('INFORM Severity category')) else ''})<br>"
                f"Crisis: {r['CRISIS'] if pd.notna(r.get('CRISIS')) else '—'}<br>"
                f"Type: {r['TYPE OF CRISIS'] if pd.notna(r.get('TYPE OF CRISIS')) else '—'}<br>"
                f"Stage: {r['pipeline_stage']}"
            ), axis=1)
        fig = world_map(df, color_col="INFORM Severity Index",
                        color_label="Severity Index",
                        color_scale="YlOrRd", range_color=[3.0, 5.0])

    elif selected == "Documented but Unplanned":
        core = load_hno_core()
        df = core[core["year"] == _YEAR].dropna(subset=["mismatch"]).copy()
        df = df.rename(columns={"iso3": "Country_ISO3"})
        df["country_name"] = df["Country_ISO3"].map(name_map).fillna(df["Country_ISO3"])
        df["_hover"] = df.apply(
            lambda r: (
                f"<b>{r['country_name']}</b><br>"
                f"Mismatch: {r['mismatch']:.2f}<br>"
                f"Need rate: {r['need_rate'] * 100:.1f}%<br>"
                f"USD / person in need: "
                + (f"${r['usd_per_in_need']:,.0f}" if pd.notna(r['usd_per_in_need']) else "N/A")
            ), axis=1)
        fig = world_map(df, color_col="mismatch",
                        color_label="Mismatch Score",
                        color_scale="RdBu_r", range_color=[-0.5, 0.5])

    elif selected == "Planned but Unfunded":
        fts = load_fts_funding()
        df = fts[fts["Year"] == _YEAR].copy()
        df["country_name"] = df["Country_ISO3"].map(name_map).fillna(df["Country_ISO3"])
        df["_hover"] = df.apply(
            lambda r: (
                f"<b>{r['country_name']}</b><br>"
                f"Coverage: {r['fts_pct_funded']:.1f}%<br>"
                f"Funding gap: {r['fts_gap_pct']:.1f}%<br>"
                f"Requirements: ${r['requirements'] / 1e6:,.0f}M<br>"
                f"Funding: ${r['funding'] / 1e6:,.0f}M"
            ), axis=1)
        fig = world_map(df, color_col="fts_gap_pct",
                        color_label="Funding Gap %",
                        color_scale="YlOrRd", range_color=[0, 100])

    elif selected == "Allocated Funding by Category":
        df = load_alignment_map()
        df["country_name"] = df["Country_ISO3"].map(name_map).fillna(df["Country_ISO3"])
        df["_hover"] = df.apply(
            lambda r: (
                f"<b>{r['country_name']}</b><br>"
                f"Alignment score: {r['alignment_score']:.2f}<br>"
                f"<br><b>Per sector:</b><br>"
                f"{r['_cluster_detail']}"
            ), axis=1)
        fig = world_map(df, color_col="alignment_score",
                        color_label="Alignment Score",
                        color_scale="RdYlGn", range_color=[0, 1])

    # ── World map ─────────────────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True)
