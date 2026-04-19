"""Stroopwafel Notebook — renders stroopwafel.ipynb as a Streamlit page."""
import json
import base64
import streamlit as st
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "stroopwafel.ipynb"

# ── Top navigation ────────────────────────────────────────────────────────────
_nc = st.columns([2, 1, 1, 2])
with _nc[1]:
    st.page_link("pages/dashboard.py", label="Overview", use_container_width=True)
with _nc[2]:
    st.page_link("pages/stroopwafel.py", label="Stroopwafel Notebook", use_container_width=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);
            padding:1.4rem 1.8rem;border-radius:10px;margin-bottom:1rem">
  <h1 style="color:#f0fdf4;margin:0;font-size:1.9rem">🧇 Stroopwafel Notebook</h1>
  <p style="color:#86efac;margin:.5rem 0 0;font-size:1rem">
    Full exploratory analysis — HNO needs · HRP requirements · INFORM severity · FTS flows · CERF allocations
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load and render notebook ──────────────────────────────────────────────────
with open(NOTEBOOK_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def _render_outputs(outputs):
    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            st.code("".join(out.get("text", [])), language=None)
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                img_bytes = base64.b64decode(data["image/png"])
                st.image(img_bytes)
            elif "text/html" in data:
                html = "".join(data["text/html"])
                st.html(html)
            elif "text/plain" in data:
                st.code("".join(data["text/plain"]), language=None)


for cell in nb["cells"]:
    cell_type = cell["cell_type"]
    source = "".join(cell.get("source", []))

    if not source.strip():
        continue

    if cell_type == "markdown":
        st.markdown(source)
    elif cell_type == "code":
        st.code(source, language="python")
        _render_outputs(cell.get("outputs", []))
