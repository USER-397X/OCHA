"""Claude API chat assistant for the Geo-Insight dashboard."""
import os
import anthropic
import streamlit as st
import pandas as pd

from scoring import fmt_usd

MODEL = "claude-opus-4-7"
MAX_TOKENS = 1024


def _get_client() -> anthropic.Anthropic | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
    return anthropic.Anthropic(api_key=api_key) if api_key else None


def _build_system_prompt(year_df: pd.DataFrame, year: int) -> str:
    """Build a data-grounded system prompt from the current view."""
    display_cols = [c for c in [
        "country_name", "CRISIS", "TYPE OF CRISIS",
        "INFORM Severity Index", "INFORM Severity category",
        "Trend (last 3 months)", "Pct_Funded",
        "revisedRequirements", "Total_Actual_Funding", "Funding_Gap",
        "gap_score", "In Need",
    ] if c in year_df.columns]

    data = year_df[display_cols].sort_values("gap_score", ascending=False).copy()

    # Human-readable formatting
    for col, fn in [
        ("revisedRequirements",  fmt_usd),
        ("Total_Actual_Funding", fmt_usd),
        ("Funding_Gap",          fmt_usd),
        ("Pct_Funded",     lambda x: f"{x:.1f}%"),
        ("INFORM Severity Index", lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"),
        ("gap_score",      lambda x: f"{x:.1f}"),
        ("In Need",        lambda x: f"{x/1e6:.1f}M" if pd.notna(x) else "N/A"),
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
| Component | Source |
|-----------|--------|
| `coverage` | Total_Actual_Funding ÷ revisedRequirements — **CERF + CBPF pooled funds only**, not total FTS funding |
| `severity` | INFORM Severity Index (1–5) |
| `scale` | Log-normalised requirements — large crises score higher but not disproportionately |

Optional **structural neglect multiplier**: upweights crises below 20 % for multiple consecutive years.

## Data sources
- `country_year_severity_funding.csv` — CERF/CBPF allocations vs. HRP requirements, 2020–2025
- `inform_severity_cleaned.csv` — INFORM Severity Index, crisis type, trend signal
- `hpc_hno_{{year}}.csv` — People-in-need figures (when available)

## Hard constraints — always follow
1. **Ground every answer in the data above.** Never invent funding figures, \
population numbers, or severity scores not present in this dataset.
2. **Coverage is pooled-fund only.** Total FTS humanitarian funding is \
substantially higher for most crises — do not present these ratios as \
overall coverage.
3. **Surface uncertainty clearly.** If data is missing, lagged, or ambiguous, say so.
4. **Scores are decision-support, not decisions.** Always note that a \
coordinator must verify context before acting on rankings.
5. **If a question falls outside the dataset**, say so and suggest which \
source would answer it.
6. Keep responses concise — humanitarian analysts are time-constrained.

## Tone
Direct, professional, honest about uncertainty. You are a trusted analyst colleague."""


def _stream_response(client: anthropic.Anthropic, system_prompt: str, messages: list):
    """Yield text chunks from Claude; system prompt is prompt-cached."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def render_chat(year_df: pd.DataFrame, year: int) -> None:
    st.markdown("### 💬 Data Assistant")
    st.caption("Ask about crises, funding gaps, scores, or methodology.")

    client = _get_client()
    if client is None:
        st.info("Set `ANTHROPIC_API_KEY` to enable the assistant.", icon="🔑")
        return

    # Rebuild system prompt only when year or row count changes
    prompt_key = f"_sysprompt_{year}_{len(year_df)}"
    if prompt_key not in st.session_state:
        for k in [k for k in st.session_state if k.startswith("_sysprompt_")]:
            del st.session_state[k]
        with st.spinner("Loading data context…"):
            st.session_state[prompt_key] = _build_system_prompt(year_df, year)
    system_prompt = st.session_state[prompt_key]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Scrollable history
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
                except anthropic.AuthenticationError:
                    reply = "❌ Invalid API key — check `ANTHROPIC_API_KEY`."
                    st.error(reply)
                except anthropic.RateLimitError:
                    reply = "⚠️ Rate limit reached. Wait a moment and retry."
                    st.warning(reply)
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
                    st.error(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.session_state.get("messages"):
        if st.button("Clear chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
