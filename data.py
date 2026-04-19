"""Data loading and caching."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_gap_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "country_year_severity_funding.csv")
    df.columns = df.columns.str.strip()
    df["Pct_Funded"] = pd.to_numeric(df["Pct_Funded"], errors="coerce").fillna(0)
    df["revisedRequirements"] = pd.to_numeric(df["revisedRequirements"], errors="coerce").fillna(0)
    df["Funding_Gap"] = pd.to_numeric(df["Funding_Gap"], errors="coerce").fillna(0)
    df["INFORM Severity Index"] = pd.to_numeric(df["INFORM Severity Index"], errors="coerce")
    return df


@st.cache_data
def load_severity_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "inform_severity_cleaned.csv")
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_hno_pin(year: int) -> pd.DataFrame:
    """Country-level people-in-need figures from the HNO file."""
    path = DATA_DIR / f"hpc_hno_{year}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Country ISO3", "In Need"])
    raw = pd.read_csv(path, header=0, skiprows=[1], low_memory=False)
    raw.columns = raw.columns.str.strip()
    mask = (
        raw["Admin 1 PCode"].isna()
        & (raw["Cluster"].str.upper().str.strip() == "ALL")
        & raw["Category"].isna()
    )
    country = raw[mask][["Country ISO3", "In Need"]].copy()
    country["In Need"] = pd.to_numeric(country["In Need"], errors="coerce")
    return country.dropna(subset=["In Need"]).groupby("Country ISO3", as_index=False)["In Need"].max()


@st.cache_data
def build_name_map(sev_df: pd.DataFrame) -> dict:
    return (
        sev_df[["ISO3", "COUNTRY"]]
        .dropna()
        .drop_duplicates("ISO3")
        .set_index("ISO3")["COUNTRY"]
        .to_dict()
    )


def enrich_year(scored_df: pd.DataFrame, sev_df: pd.DataFrame,
                name_map: dict, year: int, min_sev: float) -> pd.DataFrame:
    """Filter to one year, join crisis names and people-in-need."""
    year_df = scored_df[
        (scored_df["Year"] == year)
        & (scored_df["INFORM Severity Index"] >= min_sev)
    ].copy()

    sev_year = (
        sev_df[sev_df["Year"] == year][
            ["ISO3", "CRISIS", "TYPE OF CRISIS",
             "INFORM Severity category", "Trend (last 3 months)"]
        ]
        .groupby("ISO3", as_index=False).first()
    )
    year_df = year_df.merge(sev_year, left_on="Country_ISO3", right_on="ISO3", how="left")
    year_df["country_name"] = year_df["Country_ISO3"].map(name_map).fillna(year_df["Country_ISO3"])

    pin_df = load_hno_pin(year)
    if not pin_df.empty:
        year_df = year_df.merge(pin_df, left_on="Country_ISO3", right_on="Country ISO3", how="left")
    else:
        year_df["In Need"] = float("nan")

    return year_df
