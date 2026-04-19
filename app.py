"""Geo-Insight — navigation router and global styles."""
import streamlit as st

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
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
            st.Page("pages/stroopwafel.py", title="Stroopwafel Notebook", icon="🧇"),
        ],
    },
    position="hidden",
)
pg.run()
