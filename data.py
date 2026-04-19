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
def load_fts_funding() -> pd.DataFrame:
    """FTS total requirements vs funding, aggregated to country-year.

    Uses fts_requirements_funding_global.csv which includes all funding
    sources (bilateral, pooled, etc.) — not just CERF/CBPF.
    """
    df = pd.read_csv(DATA_DIR / "fts_requirements_funding_global.csv")
    df.columns = df.columns.str.strip()
    df["requirements"] = pd.to_numeric(df["requirements"], errors="coerce")
    df["funding"] = pd.to_numeric(df["funding"], errors="coerce")
    # Aggregate multiple plans per country-year
    agg = (
        df.groupby(["countryCode", "year"], as_index=False)
        .agg(requirements=("requirements", "sum"), funding=("funding", "sum"))
    )
    agg = agg[agg["requirements"] > 0].copy()
    agg["fts_pct_funded"] = (agg["funding"] / agg["requirements"] * 100).clip(0, 100)
    agg["fts_gap_pct"] = 100 - agg["fts_pct_funded"]
    return agg.rename(columns={"countryCode": "Country_ISO3", "year": "Year"})


@st.cache_data
def build_name_map(sev_df: pd.DataFrame) -> dict:
    return (
        sev_df[["ISO3", "COUNTRY"]]
        .dropna()
        .drop_duplicates("ISO3")
        .set_index("ISO3")["COUNTRY"]
        .to_dict()
    )


@st.cache_data
def load_hno_core() -> pd.DataFrame:
    """Return iso3/year dataframe with need_rate, coverage_rate, usd_per_in_need, mismatch."""
    YEARS = [2024, 2025, 2026]
    HNO_COLS = ["Country ISO3", "Population", "In Need", "Targeted", "Cluster", "Category",
                "Admin 1 PCode"]

    frames = []
    for y in YEARS:
        path = DATA_DIR / f"hpc_hno_{y}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, header=0, skiprows=[1], low_memory=False)
        raw.columns = raw.columns.str.strip()
        avail = [c for c in HNO_COLS if c in raw.columns]
        raw = raw[avail].copy()
        mask = (
            raw["Cluster"].str.upper().str.strip() == "ALL"
        ) & (raw["Category"].fillna("").astype(str).str.strip() == "")
        if "Admin 1 PCode" in raw.columns:
            mask = mask & raw["Admin 1 PCode"].isna()
        sub = raw[mask][["Country ISO3", "Population", "In Need", "Targeted"]].copy()
        for c in ["Population", "In Need", "Targeted"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.groupby("Country ISO3", as_index=False).agg(
            population=("Population", "max"),
            in_need=("In Need", "max"),
            targeted=("Targeted", "max"),
        )
        sub["year"] = y
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    hno_df = pd.concat(frames, ignore_index=True).rename(columns={"Country ISO3": "iso3"})

    hrp_path = DATA_DIR / "humanitarian-response-plans.csv"
    hrp = pd.read_csv(hrp_path, header=0, skiprows=[1], low_memory=False)
    hrp.columns = hrp.columns.str.strip()
    hrp["revisedRequirements"] = pd.to_numeric(hrp["revisedRequirements"], errors="coerce").fillna(0)
    hrp["loc_list"] = hrp["locations"].apply(
        lambda x: [p.strip() for p in str(x).split("|") if p.strip()] if pd.notna(x) else [])
    hrp["year_list"] = hrp["years"].apply(
        lambda x: [p.strip() for p in str(x).split("|") if p.strip()] if pd.notna(x) else [])
    hrp["n_locations"] = hrp["loc_list"].map(len)
    hrp_single = hrp[hrp["n_locations"] == 1].copy().explode("year_list")
    hrp_single["year"] = pd.to_numeric(hrp_single["year_list"], errors="coerce")
    hrp_single = hrp_single[hrp_single["year"].isin(YEARS)].copy()
    hrp_single["year"] = hrp_single["year"].astype(int)
    hrp_single["iso3"] = hrp_single["loc_list"].str[0]
    hrp_agg = hrp_single.groupby(["year", "iso3"], as_index=False).agg(
        req_sum=("revisedRequirements", "sum"))

    core = hno_df.merge(hrp_agg, on=["year", "iso3"], how="left")
    core["req_sum"] = core["req_sum"].fillna(0)

    core["need_rate"] = core["in_need"] / core["population"]
    core["coverage_rate"] = core["targeted"] / core["in_need"]
    core["usd_per_in_need"] = core["req_sum"] / core["in_need"]
    for c in ["need_rate", "coverage_rate", "usd_per_in_need"]:
        core.loc[~np.isfinite(core[c]), c] = np.nan
    for raw_col, pct_col in {"need_rate": "need_rate_pct",
                              "usd_per_in_need": "usd_per_in_need_pct"}.items():
        core[pct_col] = core.groupby("year")[raw_col].rank(pct=True, method="average")
    core["mismatch"] = core["need_rate_pct"] - core["usd_per_in_need_pct"]
    return core


@st.cache_data
def load_overlooked(year: int = 2025) -> pd.DataFrame:
    """Crises with high INFORM severity but missing from the humanitarian pipeline."""
    sev = load_severity_df()
    sev_year = sev[sev["Year"] == year].copy()
    sev_year["INFORM Severity Index"] = pd.to_numeric(
        sev_year["INFORM Severity Index"], errors="coerce")
    sev_year = (sev_year.dropna(subset=["INFORM Severity Index", "ISO3"])
                .sort_values("INFORM Severity Index", ascending=False)
                .drop_duplicates(subset=["ISO3"]))
    severe = sev_year[sev_year["INFORM Severity Index"] >= 3.0].copy()

    # HNO presence
    hno_pin = load_hno_pin(year)
    hno_isos = set(hno_pin["Country ISO3"].unique()) if not hno_pin.empty else set()

    # HRP presence — single-country plans with positive requirements
    hrp = pd.read_csv(DATA_DIR / "humanitarian-response-plans.csv",
                      header=0, skiprows=[1], low_memory=False)
    hrp.columns = hrp.columns.str.strip()
    hrp["revisedRequirements"] = pd.to_numeric(
        hrp["revisedRequirements"], errors="coerce").fillna(0)
    hrp["loc_list"] = hrp["locations"].apply(
        lambda x: [p.strip() for p in str(x).split("|") if p.strip()] if pd.notna(x) else [])
    hrp["year_list"] = hrp["years"].apply(
        lambda x: [p.strip() for p in str(x).split("|") if p.strip()] if pd.notna(x) else [])
    hrp_single = hrp[hrp["loc_list"].map(len) == 1].copy().explode("year_list")
    hrp_single["_year"] = pd.to_numeric(hrp_single["year_list"], errors="coerce")
    hrp_isos = set(
        hrp_single[(hrp_single["_year"] == year) & (hrp_single["revisedRequirements"] > 0)]
        ["loc_list"].str[0].dropna().unique()
    )

    severe["has_hno"] = severe["ISO3"].isin(hno_isos)
    severe["has_hrp"] = severe["ISO3"].isin(hrp_isos)

    def classify(row):
        if not row["has_hno"] and not row["has_hrp"]:
            return "Invisible"
        if not row["has_hno"] and row["has_hrp"]:
            return "Undocumented"
        if row["has_hno"] and not row["has_hrp"]:
            return "Unplanned"
        return "In pipeline"

    severe["pipeline_stage"] = severe.apply(classify, axis=1)
    overlooked = severe[severe["pipeline_stage"].isin(["Invisible", "Unplanned"])].copy()
    return overlooked.rename(columns={"ISO3": "Country_ISO3", "COUNTRY": "country_name"})


@st.cache_data
def load_alignment_map() -> pd.DataFrame:
    """Country-level alignment scores + per-cluster detail for the map."""
    from alignment import (load_hno_needs, load_combined_funding,
                           compute_alignment, country_alignment_score)
    needs = load_hno_needs()
    funding = load_combined_funding()
    alignment = compute_alignment(needs, funding)
    scores = country_alignment_score(alignment)
    cluster_detail = (
        alignment.groupby("country")
        .apply(lambda g: "<br>".join(
            f"  {r['cluster']}: {r['alignment_ratio']:.2f}"
            for _, r in g.sort_values("alignment_ratio").iterrows()))
        .reset_index(name="_cluster_detail")
    )
    result = scores.merge(cluster_detail, on="country")
    return result.rename(columns={"country": "Country_ISO3"})


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
