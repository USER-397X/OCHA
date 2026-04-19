"""Geo-Insight — navigation router and global styles."""
import streamlit as st

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
/* Dark sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1b2a !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] span {
    color: #c0d4ec !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1e3a5f !important;
}
/* Bolder metric values */
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}
/* Cleaner dividers */
hr { border-color: #e5e9f0 !important; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation(
    {
        "": [
            st.Page("pages/dashboard.py", title="Crisis Dashboard", icon="🌍", default=True),
        ],
        "Deep Dives": [
            st.Page("pages/bias_analysis.py", title="Bias Analysis", icon="🔍"),
            st.Page("pages/media_attention.py", title="Media Attention", icon="📰"),
            st.Page("pages/stroopwafel.py", title="Stroopwafel EDA", icon="🧇"),
        ],
    },
    position="hidden",
)
pg.run()
