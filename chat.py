"""Gemini chat assistant for the Geo-Insight dashboard."""
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from google import genai
from dotenv import load_dotenv

from scoring import fmt_usd

load_dotenv(Path(__file__).parent / ".env")

MODEL = "gemini-2.0-flash"


def _get_client() -> genai.Client | None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key) if api_key else None


def _build_system_prompt(year_df: pd.DataFrame, year: int) -> str:
    display_cols = [c for c in [
        "country_name", "CRISIS", "TYPE OF CRISIS",
        "INFORM Severity Index", "INFORM Severity category",
        "Trend (last 3 months)", "Pct_Funded",
        "revisedRequirements", "Total_Actual_Funding", "Funding_Gap",
        "gap_score", "In Need",
    ] if c in year_df.columns]

    data = year_df[display_cols].sort_values("gap_score", ascending=False).copy()

    for col, fn in [
        ("revisedRequirements",   fmt_usd),
        ("Total_Actual_Funding",  fmt_usd),
        ("Funding_Gap",           fmt_usd),
        ("Pct_Funded",            lambda x: f"{x:.1f}%"),
        ("INFORM Severity Index", lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"),
        ("gap_score",             lambda x: f"{x:.1f}"),
        ("In Need",               lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "N/A"),
    ]:
        if col in data.columns:
            data[col] = data[col].apply(fn)

    stats = dict(
        n=len(year_df),
        total_gap=fmt_usd(year_df["Funding_Gap"].sum(), 1),
        avg_cov=f"{year_df['Pct_Funded'].mean():.1f}%",
        below_20=int((year_df["Pct_Funded"] < 20).sum()),
        below_10=int((year_df["Pct_Funded"] < 10).sum()),
        avg_sev=f"{year_df['INFORM Severity Index'].mean():.2f}",
    )

    return f"""You are a humanitarian data analyst assistant embedded in the \
Geo-Insight dashboard — a tool that ranks crises by the mismatch between \
documented humanitarian need and CERF/CBPF pooled-fund coverage.

## Your role
Help humanitarian coordinators and donor advisors understand the data, \
interpret gap scores, compare crises, and formulate better questions. \
You are a thinking partner, not a decision-maker.

## Dataset: {year}

### Summary statistics
- Crises in view: {stats["n"]}
- Total funding gap: {stats["total_gap"]}
- Average CERF + CBPF coverage: {stats["avg_cov"]}
- Countries below 20 % funded: {stats["below_20"]}
- Countries below 10 % funded: {stats["below_10"]}
- Average INFORM Severity: {stats["avg_sev"]} / 5.0

### Full crisis data (ranked by Gap Score — highest = most overlooked)
```csv
{data.to_csv(index=False)}
```

## Gap score formula
```
gap_score = (1 − coverage) × (severity / 5) × log(requirements) / log(max_req) × 100
```
- `coverage` = Total_Actual_Funding ÷ revisedRequirements (CERF + CBPF pooled funds only — NOT total FTS funding)
- `severity` = INFORM Severity Index (1–5)
- `scale` = log-normalised requirements

## Hard constraints
1. Ground every answer in the data above. Never invent figures not in the dataset.
2. Coverage is pooled-fund only — do not present these ratios as overall coverage.
3. Surface uncertainty clearly when data is missing, lagged, or ambiguous.
4. Scores are decision-support, not decisions.
5. If a question falls outside the dataset, say so.
6. Keep responses concise — humanitarian analysts are time-constrained.

Tone: direct, professional, honest about uncertainty."""


def _stream_response(client: genai.Client, system_prompt: str, messages: list):
    """Yield text chunks from Gemini."""
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config={"system_instruction": system_prompt},
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def render_chat(year_df: pd.DataFrame, year: int) -> None:
    st.markdown("### 💬 Data Assistant")
    st.caption("Ask about crises, funding gaps, scores, or methodology.")

    client = _get_client()
    if client is None:
        st.info("Set `GEMINI_API_KEY` in `.env` to enable the assistant.", icon="🔑")
        return

    prompt_key = f"_sysprompt_{year}_{len(year_df)}"
    if prompt_key not in st.session_state:
        for k in [k for k in st.session_state if k.startswith("_sysprompt_")]:
            del st.session_state[k]
        with st.spinner("Loading data context…"):
            st.session_state[prompt_key] = _build_system_prompt(year_df, year)
    system_prompt = st.session_state[prompt_key]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    history = st.container(height=520, border=False)
    with history:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question…", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with history:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                try:
                    reply = st.write_stream(
                        _stream_response(client, system_prompt, st.session_state.messages)
                    )
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
                    st.error(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.get("messages"):
        if st.button("Clear chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
