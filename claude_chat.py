"""Claude chat assistant for the Geo-Insight dashboard."""
import json
import os
from pathlib import Path
import streamlit as st
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-sonnet-4-6"
PROJECT_ROOT = Path(__file__).parent

NOTEBOOKS = {
    "ocha_eda.ipynb": "main OCHA funding and gap analysis",
    "biases_eda.ipynb": "media and donor bias analysis",
    "stroopwafel.ipynb": "Stroopwafel model EDA",
}

SYSTEM_PROMPT = """You are a humanitarian data analyst assistant embedded in Geo-Insight, \
a dashboard that ranks crises by the gap between documented humanitarian need and CERF/CBPF \
pooled-fund coverage.

Help users understand funding gaps, crisis severity, gap scores, and methodology. \
Be concise and direct — humanitarian analysts are time-constrained.

When a question requires specific data, numbers, or analysis, use the read_notebook tool \
to look it up in the project's Jupyter notebooks rather than guessing or deferring to the user.

Ground your answers in data. Surface uncertainty clearly. Scores support prioritisation \
conversations — they are not automated decisions."""

TOOLS = [
    {
        "name": "read_notebook",
        "description": (
            "Read the content of a project Jupyter notebook — including code, markdown, "
            "and cell outputs — to find data, analysis results, or figures. "
            "Use this whenever a question needs specific numbers or analysis from the project."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_name": {
                    "type": "string",
                    "description": "Notebook filename",
                    "enum": list(NOTEBOOKS.keys()),
                }
            },
            "required": ["notebook_name"],
        },
    }
]

_CSS = """
<style>
/* Smooth card container for the whole chat column */
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-header) {
    background: #f8fafc;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 0.5rem 0.75rem 0.75rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* Chat input rounded pill */
[data-testid="stChatInput"] textarea {
    border-radius: 20px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 0.85rem !important;
    background: #fff !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* Tighter, cleaner chat messages */
[data-testid="stChatMessage"] {
    padding: 0.55rem 0.7rem !important;
    border-radius: 10px !important;
    margin-bottom: 2px !important;
    font-size: 0.87rem !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #eff6ff !important;
}

/* Clear button subtle style */
button[kind="secondary"] {
    border-radius: 8px;
    font-size: 0.8rem;
    color: #94a3b8;
    border-color: #e2e8f0;
}
button[kind="secondary"]:hover {
    color: #ef4444;
    border-color: #fca5a5;
    background: #fff5f5;
}
</style>
"""


def _read_notebook(notebook_name: str) -> str:
    path = PROJECT_ROOT / notebook_name
    if not path.exists():
        return f"Notebook '{notebook_name}' not found."

    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            parts.append(f"[MARKDOWN]\n{source}")
        elif cell_type == "code":
            parts.append(f"[CODE]\n{source}")
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")
                if otype == "stream":
                    text = "".join(output.get("text", []))
                elif otype in ("display_data", "execute_result"):
                    text = "".join(output.get("data", {}).get("text/plain", []))
                else:
                    text = ""
                if text.strip():
                    parts.append(f"[OUTPUT]\n{text.strip()}")

    content = "\n\n".join(parts)
    if len(content) > 60_000:
        content = content[:60_000] + "\n\n... [notebook truncated]"
    return content


def _run_agentic(client: anthropic.Anthropic, messages: list) -> tuple[str, int]:
    """Agentic tool loop; returns (reply_text, notebooks_read_count)."""
    current = list(messages)
    notebooks_read = 0

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=current,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "read_notebook":
                    notebooks_read += 1
                    result = _read_notebook(block.input["notebook_name"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            current = current + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
        else:
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            return text, notebooks_read


def _get_client() -> anthropic.Anthropic | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return anthropic.Anthropic(api_key=api_key) if api_key else None


def render_claude_chat() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # Styled header
    st.markdown("""
<div class="chat-header" style="
  background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5f 100%);
  border-radius: 10px;
  padding: 0.75rem 1rem 0.7rem;
  margin-bottom: 0.75rem;
">
  <p style="margin:0; color:#7eb8f7; font-size:.72rem; font-weight:600;
            letter-spacing:.1em; text-transform:uppercase;">AI Assistant</p>
  <p style="margin:.15rem 0 0; color:#e2e8f0; font-size:.95rem; font-weight:700;">
    Ask Claude
  </p>
  <p style="margin:.2rem 0 0; color:#4e7faa; font-size:.75rem; line-height:1.3;">
    Reads project notebooks to answer specific questions about crisis funding data.
  </p>
</div>
""", unsafe_allow_html=True)

    client = _get_client()
    if client is None:
        st.info("Set `ANTHROPIC_API_KEY` in `.env` to enable the assistant.", icon="🔑")
        return

    if "claude_messages" not in st.session_state:
        st.session_state.claude_messages = []

    history = st.container(height=430, border=False)
    with history:
        if not st.session_state.claude_messages:
            st.markdown(
                "<p style='color:#94a3b8; font-size:.82rem; text-align:center;"
                " margin-top:2rem;'>Ask a question about crisis funding,<br>"
                "gap scores, or methodology.</p>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.claude_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question…", key="claude_chat_input"):
        st.session_state.claude_messages.append({"role": "user", "content": user_input})
        with history:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                try:
                    with st.spinner(""):
                        reply, n_notebooks = _run_agentic(
                            client, st.session_state.claude_messages
                        )
                    st.markdown(reply)
                    if n_notebooks:
                        st.caption(f"📓 Read {n_notebooks} notebook{'s' if n_notebooks > 1 else ''}")
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
                    st.error(reply)
        st.session_state.claude_messages.append({"role": "assistant", "content": reply})

    if st.session_state.get("claude_messages"):
        st.button(
            "Clear conversation",
            use_container_width=True,
            type="secondary",
            key="claude_clear",
            on_click=lambda: st.session_state.update(claude_messages=[]),
        )
