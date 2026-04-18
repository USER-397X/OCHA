"""Gap-score computation and formatting helpers."""
import pandas as pd
import numpy as np

SCORE_LABEL = "Gap Score (0–100)"


def compute_gap_scores(gap_df: pd.DataFrame, use_neglect: bool) -> pd.DataFrame:
    """
    Add a `gap_score` column to every row in gap_df.

    Formula: (1 − coverage) × (severity / 5) × log(requirements) / log(max_req) × 100

    When use_neglect=True, a structural-neglect multiplier upweights crises
    that have been below 20 % coverage for consecutive years:
        multiplier = 1 + 0.3 × log(consec_years + 1) / log(7)
    """
    df = gap_df.copy()

    df["coverage"] = (df["Pct_Funded"] / 100).clip(0, 1)
    df["uncovered"] = 1 - df["coverage"]

    sev = df["INFORM Severity Index"].fillna(df["INFORM Severity Index"].median())
    df["sev_norm"] = (sev / 5).clip(0, 1)

    log_req = np.log1p(df["revisedRequirements"].clip(0))
    df["scale_norm"] = log_req / (log_req.max() or 1)

    df["gap_score"] = (df["uncovered"] * df["sev_norm"] * df["scale_norm"] * 100).round(1)

    if use_neglect:
        df = df.sort_values(["Country_ISO3", "Year"])
        df["_below"] = df["Pct_Funded"] < 20
        df["_consec"] = (
            df.groupby("Country_ISO3")["_below"]
            .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
            * df["_below"]
        )
        df["neglect_mult"] = 1 + 0.3 * np.log1p(df["_consec"]) / np.log1p(6)
        df["gap_score"] = (df["gap_score"] * df["neglect_mult"]).round(1)
        df = df.drop(columns=["_below", "_consec", "neglect_mult"])

    return df


def fmt_usd(val, decimals: int = 0) -> str:
    if pd.isna(val) or val == 0:
        return "N/A"
    if val >= 1e9:
        return f"${val/1e9:.{decimals}f}B"
    return f"${val/1e6:.{decimals}f}M"


def format_rankings_table(top_df: pd.DataFrame) -> pd.DataFrame:
    tbl = top_df[[
        "label", "INFORM Severity Index", "TYPE OF CRISIS",
        "revisedRequirements", "Pct_Funded", "Funding_Gap", "gap_score",
    ]].copy()
    tbl.columns = ["Crisis", "Severity", "Type", "Requirements", "Coverage %", "Gap", "Score"]
    tbl["Requirements"] = tbl["Requirements"].apply(fmt_usd)
    tbl["Gap"] = tbl["Gap"].apply(fmt_usd)
    tbl["Coverage %"] = tbl["Coverage %"].apply(lambda x: f"{x:.1f}%")
    tbl["Severity"] = tbl["Severity"].apply(lambda x: f"{x:.1f}")
    tbl["Score"] = tbl["Score"].apply(lambda x: f"{x:.1f}")
    return tbl.reset_index(drop=True)
