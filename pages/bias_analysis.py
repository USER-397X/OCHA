"""Bias Analysis — deep dive into systematic funding disparities."""
import streamlit as st

from data import load_gap_df, load_severity_df, build_name_map
from scoring import compute_gap_scores
from bias import render_bias_analysis

# ── Data ──────────────────────────────────────────────────────────────────────
gap_df     = load_gap_df()
sev_df     = load_severity_df()
name_map   = build_name_map(sev_df)
full_scored = compute_gap_scores(gap_df, use_neglect=True)

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
  background: linear-gradient(135deg, #1a0533 0%, #2d1b69 55%, #5b21b6 100%);
  padding: 2rem 2.5rem 1.8rem;
  border-radius: 16px;
  margin-bottom: 1.6rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.22);
">
  <p style="
    color: #c4a0ff; font-size: .8rem; letter-spacing: .14em;
    text-transform: uppercase; margin: 0 0 .5rem; font-weight: 600;
  ">Deep Dive · Pooled-fund data 2020–2025</p>
  <h1 style="color: #fff; margin: 0; font-size: 2.5rem; font-weight: 800; line-height: 1.1;">
    🔍 Bias Analysis
  </h1>
  <p style="color: #c4a0ff; margin: .65rem 0 0; font-size: 1.08rem; font-weight: 400;">
    Does the UN systematically discriminate in humanitarian funding?
  </p>
  <p style="color: #7050a8; margin: .35rem 0 0; font-size: .88rem;">
    Machine-learning–backed analysis of geographic, crisis-type, severity, and donor biases.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
render_bias_analysis(full_scored, sev_df, name_map)
