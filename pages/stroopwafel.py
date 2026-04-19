"""Stroopwafel — deep-dive EDA ported from stroopwafel.ipynb."""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style="whitegrid")

DATA_DIR = Path.home() / "Downloads" / "data-20260418T155116Z-3-001" / "data"
YEARS = [2024, 2025, 2026]


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_hdx_csv(path, usecols=None):
    return pd.read_csv(path, skiprows=[1], encoding="utf-8-sig",
                       usecols=usecols, low_memory=False)


def _split_pipe(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split("|") if p.strip()]


def _fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


CLUSTER_NAMES = {
    "ALL": "All Sectors", "WSH": "Water, Sanitation & Hygiene",
    "WASH": "Water, Sanitation & Hygiene", "NUT": "Nutrition",
    "HEA": "Health", "HEALTH": "Health", "EDU": "Education",
    "FSC": "Food Security", "FSL": "Food Security & Livelihoods",
    "FS": "Food Security", "AGR": "Agriculture", "SHL": "Shelter",
    "SHELTER": "Shelter", "NFI": "Non-Food Items",
    "S-NFI": "Shelter & Non-Food Items", "SNFI": "Shelter & Non-Food Items",
    "CCM": "Camp Coordination & Management", "CCCM": "Camp Coordination & Management",
    "SLSC": "Shelter, Land & Site Coordination", "PRO": "Protection",
    "PROT": "Protection", "PROTECTION": "Protection",
    "PRO-GBV": "Protection — GBV", "GBV": "Gender-Based Violence",
    "PRO-CPN": "Protection — Child", "PRO-CP": "Protection — Child",
    "CP": "Child Protection", "PRO-MIN": "Protection — Mine Action",
    "MA": "Mine Action", "PRO-HLP": "Protection — HLP", "HLP": "Housing, Land & Property",
    "LOG": "Logistics", "ETC": "Emergency Telecommunications",
    "TEL": "Emergency Telecommunications", "ERY": "Early Recovery",
    "ER": "Early Recovery", "EREC": "Early Recovery",
    "MPC": "Multi-Purpose Cash", "MPCA": "Multi-Purpose Cash",
    "CVA": "Cash & Voucher Assistance", "MS": "Multi-Sector",
    "CSS": "Coordination & Support", "COORD": "Coordination",
    "RMS": "Refugee & Migrant Support",
}


def _cluster_name(code):
    if pd.isna(code):
        return "Unknown"
    return CLUSTER_NAMES.get(str(code).strip().upper(), str(code))


def _get_primary_driver(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"
    return str(x).split(",")[0].strip()


def _mode_or_first(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


# ── data loading (cached) ────────────────────────────────────────────────────

@st.cache_data
def load_all():
    # HNO
    HNO_COLS = ["Country ISO3", "Description", "Cluster", "Category",
                "Population", "In Need", "Targeted"]
    hno = pd.concat([
        _read_hdx_csv(DATA_DIR / f"hpc_hno_{y}.csv", usecols=HNO_COLS).assign(year=y)
        for y in YEARS
    ], ignore_index=True)
    for c in ["Population", "In Need", "Targeted"]:
        hno[c] = pd.to_numeric(hno[c], errors="coerce")
    hno["cluster_name"] = hno["Cluster"].apply(_cluster_name)
    hno["Cluster"] = hno["Cluster"].astype(str).str.strip()
    hno["Category"] = hno["Category"].fillna("").astype(str).str.strip()

    # HRP
    HRP_COLS = ["code", "startDate", "endDate", "locations", "years",
                "origRequirements", "revisedRequirements"]
    hrp = _read_hdx_csv(DATA_DIR / "humanitarian-response-plans.csv", usecols=HRP_COLS)
    for c in ["origRequirements", "revisedRequirements"]:
        hrp[c] = pd.to_numeric(hrp[c], errors="coerce")
    hrp["startDate"] = pd.to_datetime(hrp["startDate"], errors="coerce")
    hrp["endDate"] = pd.to_datetime(hrp["endDate"], errors="coerce")
    hrp["loc_list"] = hrp["locations"].apply(_split_pipe)
    hrp["year_list"] = hrp["years"].apply(_split_pipe)
    hrp["n_locations"] = hrp["loc_list"].map(len)

    # INFORM
    inform_raw = pd.read_csv(DATA_DIR / "inform_severity_master_2020_2025.csv",
                             encoding="utf-8-sig", low_memory=False)
    inform = inform_raw.iloc[2:].copy()
    inform_cols = {
        "COUNTRY": "country_name", "ISO3": "iso3",
        "TYPE OF CRISIS": "crisis_type", "INFORM Severity Index": "severity_index",
        "INFORM Severity category.1": "severity_category",
        "Trend (last 3 months)": "trend", "Regions": "region",
        "Year": "year", "DRIVERS": "drivers",
        "Complexity of the crisis": "complexity",
        "Operating environment": "operating_env",
    }
    inform = inform[list(inform_cols.keys())].rename(columns=inform_cols)
    for col in ["severity_index", "year", "complexity", "operating_env"]:
        inform[col] = pd.to_numeric(inform[col], errors="coerce")
    inform["primary_driver"] = inform["drivers"].apply(_get_primary_driver)
    inform = inform[~inform["iso3"].str.contains(",", na=False)].copy()
    inform_master = inform.copy()

    # COD population / name map
    COD0_COLS = ["ISO3", "Country"]
    cod0 = _read_hdx_csv(DATA_DIR / "cod_population_admin0.csv", usecols=COD0_COLS)
    cod0 = cod0[~cod0["ISO3"].astype(str).str.startswith("#")].copy()
    name_map = (
        cod0.drop_duplicates("ISO3")[["ISO3", "Country"]]
        .rename(columns={"ISO3": "iso3", "Country": "country"})
    )

    # --- core table ---
    hno_overall = (
        hno.query("Cluster == 'ALL' and Category == ''")
        .rename(columns={"Country ISO3": "iso3", "Population": "population",
                         "In Need": "in_need", "Targeted": "targeted"})
        [["year", "iso3", "population", "in_need", "targeted"]].copy()
    )

    hrp_single = hrp.query("n_locations == 1").copy()
    hrp_single = hrp_single.explode("year_list")
    hrp_single["year"] = pd.to_numeric(hrp_single["year_list"], errors="coerce")
    hrp_single = hrp_single[hrp_single["year"].isin(YEARS)].copy()
    hrp_single["year"] = hrp_single["year"].astype(int)
    hrp_single["iso3"] = hrp_single["loc_list"].str[0]
    hrp_agg = (
        hrp_single
        .assign(revisedRequirements=hrp_single["revisedRequirements"].fillna(0))
        .groupby(["year", "iso3"], as_index=False)
        .agg(req_sum=("revisedRequirements", "sum"),
             req_max=("revisedRequirements", "max"),
             n_plans=("code", "nunique"))
    )

    core = hno_overall.merge(hrp_agg, on=["year", "iso3"], how="left").merge(name_map, on="iso3", how="left")
    core["country"] = core["country"].fillna(core["iso3"])
    for c in ["req_sum", "req_max", "n_plans"]:
        core[c] = core[c].fillna(0)

    # derived metrics
    core = core.copy()
    core["need_rate"] = core["in_need"] / core["population"]
    core["coverage_rate"] = core["targeted"] / core["in_need"]
    core["usd_per_in_need"] = core["req_sum"] / core["in_need"]
    core["usd_per_in_need_max"] = core["req_max"] / core["in_need"]
    core["req_per_capita"] = core["req_sum"] / core["population"]
    core["funding_gap_people"] = core["in_need"] - core["targeted"]
    for c in ["need_rate", "coverage_rate", "usd_per_in_need", "usd_per_in_need_max", "req_per_capita"]:
        core.loc[~np.isfinite(core[c]), c] = np.nan
    core["need_share"] = core.groupby("year")["in_need"].transform(
        lambda s: s / s.sum() if s.sum() else np.nan)
    core["req_share"] = core.groupby("year")["req_sum"].transform(
        lambda s: s / s.sum() if s.sum() else np.nan)
    core["share_gap"] = core["need_share"] - core["req_share"]
    for raw, pct in {"need_rate": "need_rate_pct", "in_need": "in_need_pct",
                     "usd_per_in_need": "usd_per_in_need_pct"}.items():
        core[pct] = core.groupby("year")[raw].rank(pct=True, method="average")
    core["mismatch"] = core["need_rate_pct"] - core["usd_per_in_need_pct"]
    core["log10_in_need"] = np.log10(core["in_need"].where(core["in_need"] > 0))
    core["log10_usd_per_in_need"] = np.log10(core["usd_per_in_need"].where(core["usd_per_in_need"] > 0))

    COUNTRY_NAMES = {
        "SDN": "Sudan", "MMR": "Myanmar", "AFG": "Afghanistan", "YEM": "Yemen",
        "SYR": "Syria", "COD": "DR Congo", "SSD": "South Sudan", "HTI": "Haiti",
        "VEN": "Venezuela", "COL": "Colombia", "NGA": "Nigeria", "MLI": "Mali",
        "ETH": "Ethiopia", "BGD": "Bangladesh", "PSE": "Palestine", "UKR": "Ukraine",
    }
    core["country"] = core["iso3"].map(COUNTRY_NAMES).fillna(core["country"])

    # core_enriched
    inform_join = inform_master[[
        "iso3", "year", "severity_index", "severity_category", "trend",
        "region", "crisis_type", "drivers", "primary_driver",
        "complexity", "operating_env",
    ]].copy()
    inform_join["year"] = pd.to_numeric(inform_join["year"], errors="coerce").astype("Int64")
    inform_2025 = inform_join[inform_join["year"] == 2025].copy()
    inform_2026_proxy = inform_2025.assign(year=pd.array([2026] * len(inform_2025), dtype="Int64"))
    inform_join_ext = pd.concat([inform_join, inform_2026_proxy], ignore_index=True)
    inform_collapsed = (
        inform_join_ext
        .groupby(["iso3", "year"], as_index=False)
        .agg(
            severity_index=("severity_index", "max"),
            severity_category=("severity_category", _mode_or_first),
            trend=("trend", _mode_or_first),
            region=("region", _mode_or_first),
            crisis_type=("crisis_type", lambda s: "|".join(sorted(set(str(x) for x in s.dropna())))),
            drivers=("drivers", _mode_or_first),
            primary_driver=("primary_driver", _mode_or_first),
            complexity=("complexity", "max"),
            operating_env=("operating_env", "max"),
        )
    )
    core_enriched = core.merge(inform_collapsed, on=["iso3", "year"], how="left")
    sev_norm = (core_enriched["severity_index"] / 5.0).clip(lower=0, upper=1)
    core_enriched["mismatch_severity"] = sev_norm.fillna(0) - core_enriched["usd_per_in_need_pct"]
    hrp_first = (
        hrp.query("n_locations == 1")
        .assign(iso3=lambda d: d["loc_list"].str[0],
                start_year=lambda d: d["startDate"].dt.year)
        .dropna(subset=["iso3", "start_year"])
        .groupby("iso3", as_index=False)["start_year"].min()
        .rename(columns={"start_year": "first_hrp_year"})
    )
    core_enriched = core_enriched.merge(hrp_first, on="iso3", how="left")
    core_enriched["years_since_first_response"] = (
        core_enriched["year"] - core_enriched["first_hrp_year"])

    return hno, hrp, hrp_agg, inform, inform_master, name_map, core, core_enriched


@st.cache_data
def load_fts():
    FTS_PATH = DATA_DIR / "fts_outgoing_funding_global.csv"
    fts_raw = pd.read_csv(FTS_PATH, low_memory=False)
    fts_raw["amountUSD"] = pd.to_numeric(fts_raw["amountUSD"], errors="coerce").fillna(0)
    fts_raw["year"] = pd.to_numeric(fts_raw["budgetYear"], errors="coerce")
    fts_raw["year"] = fts_raw["year"].fillna(
        pd.to_numeric(fts_raw["destUsageYearStart"], errors="coerce"))
    fts_raw["year"] = pd.to_numeric(fts_raw["year"], errors="coerce")

    def _classify_channel(row):
        dest_type = str(row.get("destOrganizationTypes", ""))
        dest_org = str(row.get("destOrganization", ""))
        if dest_type != "Pooled Funds":
            return "Bilateral / agency"
        if "Central Emergency Response Fund" in dest_org:
            return "CERF"
        if dest_org.endswith("Humanitarian Fund"):
            if any(x in dest_org for x in ["Women's Peace", "SDG"]):
                return "Other pooled"
            return "CBPF"
        if any(tag in dest_org for tag in [
            "(Eastern, Southern Africa HF)", "(West, Central Africa HF)",
            "(Asia, Pacific HF)", "(Latin America, Caribbean HF)",
        ]):
            return "CBPF"
        return "Other pooled"

    fts_raw["channel"] = fts_raw.apply(_classify_channel, axis=1)
    return fts_raw


@st.cache_data
def build_hrp_fts(core_key=None):
    hno, hrp, hrp_agg, inform, inform_master, name_map, core, core_enriched = load_all()
    COVERAGE_YEARS = [2024, 2025, 2026]
    FTS_PATH = DATA_DIR / "fts_outgoing_funding_global.csv"
    fts_raw = pd.read_csv(FTS_PATH, low_memory=False)
    fts = fts_raw.copy()
    fts["year"] = pd.to_numeric(fts["budgetYear"], errors="coerce")
    fts["year"] = fts["year"].fillna(pd.to_numeric(fts["destUsageYearStart"], errors="coerce"))
    fts = fts.dropna(subset=["year"])
    fts["year"] = fts["year"].astype(int)
    fts = fts[fts["year"].isin(COVERAGE_YEARS)].copy()
    fts["amountUSD"] = pd.to_numeric(fts["amountUSD"], errors="coerce").fillna(0)
    fts = fts[fts["onBoundary"] == "single"].copy()
    fts["iso3"] = fts["destLocations"].astype(str).str.strip()
    fts = fts[fts["iso3"].str.len() == 3].copy()

    funding_by_status = (
        fts.groupby(["iso3", "year", "status"], as_index=False)["amountUSD"].sum()
        .pivot_table(index=["iso3", "year"], columns="status", values="amountUSD", fill_value=0)
        .reset_index()
    )
    for s in ["paid", "commitment", "pledge"]:
        if s not in funding_by_status.columns:
            funding_by_status[s] = 0.0
    funding_by_status = funding_by_status.rename(columns={
        "paid": "fts_paid", "commitment": "fts_commitment", "pledge": "fts_pledge"})
    funding_by_status["fts_funded"] = (
        funding_by_status["fts_paid"] + funding_by_status["fts_commitment"])
    funding_by_status["fts_total_reported"] = (
        funding_by_status["fts_paid"] + funding_by_status["fts_commitment"]
        + funding_by_status["fts_pledge"])

    hrp_fts = (
        core[["iso3", "country", "year", "in_need", "req_sum"]]
        .merge(funding_by_status, on=["iso3", "year"], how="left")
    )
    for c in ["fts_paid", "fts_commitment", "fts_pledge", "fts_funded", "fts_total_reported"]:
        hrp_fts[c] = hrp_fts[c].fillna(0)
    has_hrp = hrp_fts["req_sum"] > 0
    hrp_fts["coverage_ratio"] = np.where(
        has_hrp, hrp_fts["fts_funded"] / hrp_fts["req_sum"], np.nan)
    hrp_fts["coverage_ratio_paid_only"] = np.where(
        has_hrp, hrp_fts["fts_paid"] / hrp_fts["req_sum"], np.nan)
    hrp_fts["funding_gap_usd"] = np.where(
        has_hrp, (hrp_fts["req_sum"] - hrp_fts["fts_funded"]).clip(lower=0), np.nan)
    return hrp_fts


@st.cache_data
def load_cerf(name_map_df):
    CERF_PATH = DATA_DIR / "Data_ CERF Donor Contributions and Allocations - allocations.csv"
    cerf_raw = pd.read_csv(CERF_PATH)
    cerf_raw = cerf_raw.rename(columns={
        "countryCode": "iso3", "totalAmountApproved": "cerf_usd",
        "windowFullName": "window", "agencyName": "agency",
        "emergencyTypeName": "emergency_type",
    })
    cerf_raw["year"] = pd.to_numeric(cerf_raw["year"], errors="coerce").astype("Int64")
    cerf_raw["cerf_usd"] = pd.to_numeric(cerf_raw["cerf_usd"], errors="coerce").fillna(0)
    CERF_YEARS = [2022, 2023, 2024, 2025, 2026]
    cerf_recent = cerf_raw[cerf_raw["year"].isin(CERF_YEARS)].copy()
    cerf_recent["rr_usd"] = np.where(cerf_recent["window"] == "Rapid Response",
                                     cerf_recent["cerf_usd"], 0)
    cerf_recent["ufe_usd"] = np.where(cerf_recent["window"] == "Underfunded Emergencies",
                                      cerf_recent["cerf_usd"], 0)
    cerf_cy = (
        cerf_recent
        .groupby(["iso3", "year"], as_index=False)
        .agg(cerf_total_usd=("cerf_usd", "sum"), cerf_rr_usd=("rr_usd", "sum"),
             cerf_ufe_usd=("ufe_usd", "sum"), n_projects=("projectID", "nunique"),
             n_agencies=("agency", "nunique"))
    )
    cerf_cy["year"] = cerf_cy["year"].astype(int)
    cerf_cy["received_ufe"] = cerf_cy["cerf_ufe_usd"] > 0
    cerf_cy = cerf_cy.merge(name_map_df, on="iso3", how="left")
    cerf_cy["country"] = cerf_cy["country"].fillna(cerf_cy["iso3"])
    return cerf_cy


# ── page ─────────────────────────────────────────────────────────────────────

_nc = st.columns([1, 1, 1, 1, 2])
with _nc[0]:
    st.page_link("pages/dashboard.py", label="🌍 Overview", use_container_width=True)
with _nc[1]:
    st.page_link("pages/media_attention.py", label="📰 Media Attention", use_container_width=True)
with _nc[2]:
    st.page_link("pages/bias_analysis.py", label="🔍 Bias Analysis", use_container_width=True)
with _nc[3]:
    st.page_link("pages/stroopwafel.py", label="🧇 Stroopwafel EDA", use_container_width=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);
            padding:1.4rem 1.8rem;border-radius:10px;margin-bottom:1rem">
  <h1 style="color:#f0fdf4;margin:0;font-size:1.9rem">🧇 Stroopwafel EDA</h1>
  <p style="color:#86efac;margin:.5rem 0 0;font-size:1rem">
    Deep-dive exploratory analysis — HNO needs · HRP requirements · INFORM severity · FTS flows · CERF allocations
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading data…"):
    hno, hrp, hrp_agg, inform, inform_master, name_map_df, core, core_enriched = load_all()

# ── shared metric selector ────────────────────────────────────────────────────
METRIC_OPTIONS = {
    "A — Need Rate": ("need_rate", "RdYlGn_r"),
    "B — Coverage Rate": ("coverage_rate", "RdYlGn"),
    "C — USD / Person in Need": ("usd_per_in_need", "YlOrRd"),
    "D — Mismatch Score": ("mismatch", "RdYlGn_r"),
}
if "selected_metric" not in st.session_state:
    st.session_state["selected_metric"] = list(METRIC_OPTIONS)[0]

st.caption("Select metric — updates heatmap and front-page rankings")
st.radio("", list(METRIC_OPTIONS), horizontal=True, key="selected_metric",
         label_visibility="collapsed")
st.session_state["stroopwafel_core"] = core

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data overview",
    "🌍 INFORM severity",
    "⚖️ Need vs resources",
    "🚨 Invisible crises",
    "💰 Funding coverage",
    "✅ Validation",
])


# ── tab 1: data overview ─────────────────────────────────────────────────────
with tab1:
    st.header("1. Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("HNO")
        st.dataframe(pd.DataFrame({
            "Metric": ["Total records", "Years", "Countries"],
            "Value": [f"{len(hno):,}", str(sorted(hno['year'].unique())),
                      str(hno['Country ISO3'].nunique())],
        }), hide_index=True)

    with col2:
        st.subheader("HRP")
        st.dataframe(pd.DataFrame({
            "Metric": ["Total plans", "Single-country", "Total requirements (USD)"],
            "Value": [f"{len(hrp):,}", f"{(hrp['n_locations'] == 1).sum():,}",
                      f"${hrp['revisedRequirements'].sum()/1e9:.1f}B"],
        }), hide_index=True)

    with col3:
        st.subheader("INFORM")
        st.dataframe(pd.DataFrame({
            "Metric": ["Total records", "Countries", "Years"],
            "Value": [f"{len(inform):,}", str(inform['iso3'].nunique()),
                      str(sorted(inform['year'].dropna().astype(int).unique()))],
        }), hide_index=True)

    st.divider()
    st.subheader("People in Need by cluster (2024–2026 combined)")
    hno_by_cluster = (
        hno.query("Category == ''")
        .groupby(["Cluster", "cluster_name"], as_index=False)
        .agg(in_need=("In Need", "sum"), targeted=("Targeted", "sum"))
        .sort_values("in_need", ascending=False)
    )
    hno_by_cluster = hno_by_cluster[hno_by_cluster["Cluster"] != "ALL"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(hno_by_cluster["cluster_name"], hno_by_cluster["in_need"] / 1e6,
            color="#6366f1", edgecolor="black")
    ax.set_xlabel("People in Need (millions)")
    ax.set_title("Humanitarian Need by Cluster (2024–2026)")
    ax.invert_yaxis()
    for i, v in enumerate(hno_by_cluster["in_need"] / 1e6):
        ax.text(v, i, f"  {v:.1f}M", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── tab 2: INFORM severity ───────────────────────────────────────────────────
with tab2:
    st.header("2. INFORM Severity Coverage")

    # Crisis type distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    crisis_counts = inform["crisis_type"].value_counts().head(6)
    ax.barh(crisis_counts.index, crisis_counts.values, color="#6366f1", edgecolor="black")
    ax.set_xlabel("Number of Records")
    ax.set_title(f"INFORM Severity Data: {len(inform):,} Records by Crisis Type")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Countries & Regions")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    region_counts = (inform.dropna(subset=["region"])
                     .groupby("region").size().sort_values(ascending=True))
    axes[0, 0].barh(region_counts.index, region_counts.values, color="#6366f1", edgecolor="black")
    axes[0, 0].set_title("Records per Region")
    axes[0, 0].set_xlabel("Number of Records")
    for i, v in enumerate(region_counts.values):
        axes[0, 0].text(v, i, f" {v:,}", va="center", fontsize=9)

    countries_per_region = (inform.dropna(subset=["region", "iso3"])
                            .groupby("region")["iso3"].nunique().sort_values(ascending=True))
    axes[0, 1].barh(countries_per_region.index, countries_per_region.values,
                    color="#10b981", edgecolor="black")
    axes[0, 1].set_title("Unique Countries per Region")
    axes[0, 1].set_xlabel("Number of Countries")
    for i, v in enumerate(countries_per_region.values):
        axes[0, 1].text(v, i, f" {v}", va="center", fontsize=9)

    top_countries = (inform.dropna(subset=["severity_index", "country_name"])
                     .groupby("country_name")["severity_index"].mean()
                     .sort_values(ascending=False).head(15).sort_values(ascending=True))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_countries)))
    axes[1, 0].barh(top_countries.index, top_countries.values, color=colors, edgecolor="black")
    axes[1, 0].set_title("Top 15 Countries by Avg Severity Index")
    axes[1, 0].set_xlabel("Mean INFORM Severity Index")
    for i, v in enumerate(top_countries.values):
        axes[1, 0].text(v, i, f" {v:.2f}", va="center", fontsize=9)

    region_order = (inform.dropna(subset=["region", "severity_index"])
                    .groupby("region")["severity_index"].median()
                    .sort_values(ascending=False).index)
    sns.boxplot(data=inform.dropna(subset=["region", "severity_index"]),
                y="region", x="severity_index", order=region_order,
                ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Severity Distribution by Region")
    axes[1, 1].set_xlabel("INFORM Severity Index")
    axes[1, 1].set_ylabel("")

    plt.suptitle(f"INFORM Severity — {inform['iso3'].nunique()} countries, "
                 f"{inform['region'].nunique()} regions",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Trends over time (2020–2025)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    year_counts = inform["year"].value_counts().sort_index()
    axes[0, 0].bar(year_counts.index.astype(int), year_counts.values, color="#6366f1", edgecolor="black")
    axes[0, 0].set_title("Records per Year")
    for x, y in zip(year_counts.index.astype(int), year_counts.values):
        axes[0, 0].text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9)

    sns.boxplot(data=inform.dropna(subset=["severity_index", "year"]),
                x="year", y="severity_index", ax=axes[0, 1], color="#818cf8")
    axes[0, 1].set_title("Severity Distribution by Year")

    mean_sev = inform.groupby("year")["severity_index"].mean().sort_index()
    axes[1, 0].plot(mean_sev.index.astype(int), mean_sev.values, marker="o", color="#4f46e5", linewidth=2)
    axes[1, 0].fill_between(mean_sev.index.astype(int), mean_sev.values, alpha=0.2, color="#6366f1")
    axes[1, 0].set_title("Average Severity Index Over Time")

    cat_by_year = (inform.dropna(subset=["severity_category", "year"])
                   .groupby(["year", "severity_category"]).size().unstack(fill_value=0))
    cat_pct = cat_by_year.div(cat_by_year.sum(axis=1), axis=0) * 100
    cat_pct.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="viridis",
                 edgecolor="black", width=0.8)
    axes[1, 1].set_title("Severity Category Share by Year (%)")
    axes[1, 1].legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Crisis drivers: complexity vs operating environment")

    plot_df = inform.dropna(subset=["complexity", "operating_env", "region"]).copy()
    regions = sorted(plot_df["region"].unique())
    palette = dict(zip(regions, sns.color_palette("tab10", n_colors=len(regions))))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].scatter(plot_df["complexity"], plot_df["operating_env"],
                    s=40, alpha=0.5, color="#6366f1", edgecolor="black", linewidth=0.3)
    if plot_df[["complexity", "operating_env"]].dropna().shape[0] > 1:
        z = np.polyfit(plot_df["complexity"].dropna(), plot_df["operating_env"].dropna(), 1)
        xs = np.linspace(plot_df["complexity"].min(), plot_df["complexity"].max(), 100)
        axes[0].plot(xs, np.polyval(z, xs), color="red", linewidth=2, linestyle="--",
                     label=f"Fit (slope={z[0]:.2f})")
        axes[0].legend()
    axes[0].set_title("Complexity vs Operating Environment")
    axes[0].set_xlabel("Complexity")
    axes[0].set_ylabel("Operating Environment")

    agg = (plot_df.groupby("region")
           .agg(complexity=("complexity", "mean"), operating_env=("operating_env", "mean"),
                severity=("severity_index", "mean"), n=("iso3", "count"))
           .reset_index())
    sev = agg["severity"].fillna(agg["severity"].mean())
    sizes = 100 + (sev - sev.min()) / max(sev.max() - sev.min(), 1e-9) * 600
    for i, row in agg.iterrows():
        axes[1].scatter(row["complexity"], row["operating_env"], s=sizes.iloc[i],
                        color=palette.get(row["region"], "#888"), alpha=0.8,
                        edgecolor="black", linewidth=1)
        axes[1].annotate(row["region"], (row["complexity"], row["operating_env"]),
                         xytext=(8, 4), textcoords="offset points", fontsize=9, fontweight="bold")
    axes[1].set_title("Regional Means (bubble = mean severity)")
    axes[1].set_xlabel("Mean Complexity")
    axes[1].set_ylabel("Mean Operating Environment")

    year_df2 = plot_df.dropna(subset=["year", "complexity"]).copy()
    year_df2["year"] = year_df2["year"].astype(int)
    sns.boxplot(data=year_df2, x="year", y="complexity", ax=axes[2], color="#818cf8", width=0.6)
    mean_by_year = year_df2.groupby("year")["complexity"].mean().sort_index()
    axes[2].plot(range(len(mean_by_year)), mean_by_year.values, color="red", marker="o",
                 linewidth=2, label="Mean", zorder=5)
    axes[2].set_title("Complexity by Year")
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── tab 3: need vs resources ─────────────────────────────────────────────────
with tab3:
    st.header("3. Need vs Resources")

    st.subheader("Key metrics summary")
    metrics_summary = core[["need_rate", "coverage_rate", "usd_per_in_need", "mismatch"]].describe().round(3)
    st.dataframe(metrics_summary)

    st.divider()
    year_sel = st.select_slider("Year", options=YEARS, value=YEARS[-1], key="nvr_year")
    n_labels = st.slider("Countries to label", 3, 12, 7, key="nvr_labels")

    df_year = core[core["year"] == year_sel].copy()
    if not df_year.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            df_year["need_rate"] * 100, df_year["usd_per_in_need"],
            s=df_year["in_need"] / 1e6 * 5,
            c=df_year["mismatch"], cmap="RdYlGn_r", alpha=0.7,
            edgecolors="black", linewidth=0.5,
        )
        for _, row in df_year.nlargest(n_labels, "mismatch").iterrows():
            ax.annotate(row["country"], (row["need_rate"] * 100, row["usd_per_in_need"]),
                        fontsize=9, color="#8b0000", fontweight="bold")
        for _, row in df_year.nsmallest(n_labels, "mismatch").iterrows():
            ax.annotate(row["country"], (row["need_rate"] * 100, row["usd_per_in_need"]),
                        fontsize=9, color="#006400", fontweight="bold")
        median_usd = df_year["usd_per_in_need"].median()
        ax.axhline(median_usd, color="gray", linestyle="--", alpha=0.5,
                   label=f"Median: ${median_usd:.0f}/person")
        ax.set_xlabel("Need Rate (% of population in need)", fontsize=12)
        ax.set_ylabel("USD Requested per Person in Need", fontsize=12)
        ax.set_title(f"Need vs Resource Allocation ({year_sel})\n"
                     "Bubble size = people in need | Red = underserved | Green = over-resourced",
                     fontsize=13)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Mismatch Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Most underserved ({year_sel})**")
            st.dataframe(
                df_year.nlargest(n_labels, "mismatch")[
                    ["country", "need_rate", "usd_per_in_need", "mismatch"]
                ].round(3).reset_index(drop=True))
        with col_b:
            st.markdown(f"**Most over-resourced ({year_sel})**")
            st.dataframe(
                df_year.nsmallest(n_labels, "mismatch")[
                    ["country", "need_rate", "usd_per_in_need", "mismatch"]
                ].round(3).reset_index(drop=True))

    st.divider()
    st.subheader("Top 5 underserved crises per year")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        df_yr = core[core["year"] == year].nlargest(5, "mismatch")
        norm = Normalize(vmin=df_yr["mismatch"].min(), vmax=df_yr["mismatch"].max())
        bar_clrs = plt.cm.Reds(0.3 + 0.6 * norm(df_yr["mismatch"]))
        bars = ax.barh(df_yr["country"], df_yr["mismatch"], color=bar_clrs, edgecolor="black")
        ax.set_xlabel("Mismatch Score")
        ax.set_title(f"Top 5 Underserved ({year})")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        for bar, val in zip(bars, df_yr["mismatch"]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)
    plt.suptitle("Most Underserved Crises by Year", fontsize=14, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Country × Year heatmap")

    metric_label = st.session_state.get("selected_metric", list(METRIC_OPTIONS)[0])
    col_name, cmap_name = METRIC_OPTIONS[metric_label]

    hm_df = core[["country", "year", col_name]].dropna(subset=[col_name])
    if not hm_df.empty:
        pivot_hm = hm_df.pivot_table(index="country", columns="year", values=col_name, aggfunc="mean")
        pivot_hm = pivot_hm.reindex(sorted(pivot_hm.index))

        fig_h, ax_h = plt.subplots(figsize=(max(6, len(pivot_hm.columns) * 1.8),
                                            max(4, len(pivot_hm) * 0.35)))
        sns.heatmap(
            pivot_hm, ax=ax_h, cmap=cmap_name, annot=True,
            fmt=".2g", linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"shrink": 0.7},
        )
        ax_h.set_title(metric_label, fontsize=12)
        ax_h.set_xlabel("")
        ax_h.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)


# ── tab 4: invisible crises ──────────────────────────────────────────────────
with tab4:
    st.header("4. Invisible Crises — Pipeline Gap Analysis")
    st.markdown("""
Countries classified by their position in the humanitarian pipeline:
- **Invisible** — high severity, no HNO, no HRP
- **Undocumented** — no HNO but HRP exists
- **Unplanned** — HNO exists but no HRP
- **In pipeline** — both HNO and HRP
""")

    sev_floor = st.slider("INFORM severity floor", 2.0, 4.5, 3.0, 0.5, key="inv_floor")
    ANALYSIS_YEARS_INV = [2024, 2025]

    sev_df = (
        inform_master.dropna(subset=["severity_index", "iso3", "year"])
        .assign(year=lambda d: d["year"].astype(int))
        .query("year in @ANALYSIS_YEARS_INV")
        .sort_values(["iso3", "year", "severity_index"], ascending=[True, True, False])
        .drop_duplicates(subset=["iso3", "year"], keep="first")
        [["iso3", "year", "country_name", "severity_index", "severity_category",
          "primary_driver", "crisis_type", "region", "trend"]]
    )
    severe = sev_df[sev_df["severity_index"] >= sev_floor].copy()

    hno_clean = hno.copy()
    hno_clean["Cluster"] = hno_clean["Cluster"].astype(str).str.strip()
    hno_clean["Category"] = hno_clean["Category"].fillna("").astype(str).str.strip()
    hno_present = (
        hno_clean.query("Cluster == 'ALL' and Category == ''")
        .rename(columns={"Country ISO3": "iso3", "In Need": "in_need"})
        .assign(in_need=lambda d: pd.to_numeric(d["in_need"], errors="coerce"))
        .dropna(subset=["in_need"]).query("in_need > 0")
        [["iso3", "year", "in_need"]].drop_duplicates(subset=["iso3", "year"])
    )
    hno_present["has_hno"] = True

    hrp_present = (hrp_agg.query("req_sum > 0")[["iso3", "year", "req_sum"]]
                   .drop_duplicates(subset=["iso3", "year"]))
    hrp_present["has_hrp"] = True

    overlap = (severe
               .merge(hno_present, on=["iso3", "year"], how="left")
               .merge(hrp_present, on=["iso3", "year"], how="left"))
    overlap["has_hno"] = overlap["has_hno"].fillna(False)
    overlap["has_hrp"] = overlap["has_hrp"].fillna(False)
    overlap["in_need"] = overlap["in_need"].fillna(0)
    overlap["req_sum"] = overlap["req_sum"].fillna(0)

    def _classify(row):
        if not row["has_hno"] and not row["has_hrp"]:
            return "Invisible"
        if not row["has_hno"] and row["has_hrp"]:
            return "Undocumented"
        if row["has_hno"] and not row["has_hrp"]:
            return "Unplanned"
        return "In pipeline"

    overlap["pipeline_stage"] = overlap.apply(_classify, axis=1)
    overlap = overlap.merge(name_map_df, on="iso3", how="left")
    overlap["country"] = overlap["country"].fillna(overlap["country_name"]).fillna(overlap["iso3"])

    stage_summary = (
        overlap.groupby(["year", "pipeline_stage"], as_index=False)
        .agg(n_countries=("iso3", "nunique"))
        .pivot(index="year", columns="pipeline_stage", values="n_countries")
        .fillna(0).astype(int)
    )
    for col in ["Invisible", "Undocumented", "In pipeline"]:
        if col not in stage_summary.columns:
            stage_summary[col] = 0

    st.subheader("Pipeline stage breakdown")
    st.dataframe(stage_summary)

    STAGE_COLORS = {
        "Invisible": "#b91c1c", "Unplanned": "#ef4444",
        "Undocumented": "#f59e0b", "In pipeline": "#10b981",
    }

    overlooked = (
        overlap.query("pipeline_stage in ['Invisible', 'Unplanned']")
        .sort_values(["year", "severity_index"], ascending=[False, False])
        [["year", "iso3", "country", "severity_index", "severity_category",
          "primary_driver", "trend", "pipeline_stage", "has_hno", "has_hrp", "in_need"]]
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cols_present = [c for c in stage_summary.columns if c in STAGE_COLORS]
    stage_summary[cols_present].plot(
        kind="bar", stacked=True, ax=axes[0],
        color=[STAGE_COLORS[c] for c in cols_present], edgecolor="black")
    axes[0].set_title(f"Severe crises (INFORM ≥ {sev_floor}) by pipeline stage")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Number of countries")
    axes[0].legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].tick_params(axis="x", rotation=0)

    if not overlooked.empty:
        latest_year = overlooked["year"].max()
        latest_ol = (overlooked[overlooked["year"] == latest_year]
                     .sort_values("severity_index", ascending=True).tail(15))
        bar_clrs2 = latest_ol["pipeline_stage"].map(STAGE_COLORS)
        axes[1].barh(latest_ol["country"], latest_ol["severity_index"],
                     color=bar_clrs2, edgecolor="black")
        axes[1].axvline(sev_floor, color="black", linestyle="--", linewidth=0.7)
        axes[1].set_title(f"Overlooked countries in {latest_year}")
        axes[1].set_xlabel("INFORM Severity Index")
        axes[1].legend(handles=[Patch(color=STAGE_COLORS[s], label=s)
                                 for s in ["Invisible", "Unplanned"]], loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Overlooked countries table")
    st.dataframe(overlooked, use_container_width=True)


# ── tab 5: funding coverage ──────────────────────────────────────────────────
with tab5:
    st.header("5. HRP Funding Coverage (FTS)")

    with st.spinner("Loading FTS data…"):
        hrp_fts = build_hrp_fts()

    year_cov = st.select_slider("Year", options=YEARS, value=YEARS[-1], key="cov_year")
    latest_view = (hrp_fts.query("year == @year_cov and req_sum > 0")
                   .sort_values("coverage_ratio", ascending=True).copy())

    if latest_view.empty:
        st.warning("No data for this year.")
    else:
        med_cov = latest_view["coverage_ratio"].median()
        st.metric("Countries with HRP", len(latest_view))
        st.metric("Median coverage (paid + committed)",
                  f"{med_cov:.1%}")

        st.subheader("Least-funded HRPs")
        bottom = latest_view.head(15).copy()
        bottom["in_need_fmt"] = bottom["in_need"].apply(_fmt)
        bottom["req_sum_fmt"] = bottom["req_sum"].apply(lambda x: "$" + _fmt(x))
        bottom["fts_paid_fmt"] = bottom["fts_paid"].apply(lambda x: "$" + _fmt(x))
        bottom["coverage_fmt"] = (bottom["coverage_ratio"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            bottom[["country", "in_need_fmt", "req_sum_fmt", "fts_paid_fmt", "coverage_fmt"]]
            .rename(columns={"country": "Country", "in_need_fmt": "In Need",
                             "req_sum_fmt": "HRP Ask", "fts_paid_fmt": "Paid",
                             "coverage_fmt": "Coverage"}),
            hide_index=True, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        rank_df = latest_view.sort_values("coverage_ratio", ascending=True)
        bar_clrs3 = ["#b91c1c" if r < 0.25 else "#ef4444" if r < 0.50
                     else "#f59e0b" if r < 0.75 else "#10b981"
                     for r in rank_df["coverage_ratio"]]
        axes[0].barh(rank_df["country"], rank_df["coverage_ratio"],
                     color=bar_clrs3, edgecolor="black")
        axes[0].axvline(1.0, color="black", linestyle="--", linewidth=0.7, label="100%")
        axes[0].axvline(0.50, color="gray", linestyle=":", linewidth=0.7, label="50%")
        axes[0].set_xlabel("Coverage (paid + committed) / HRP requirement")
        axes[0].set_title(f"HRP coverage by country, {year_cov}")
        axes[0].legend()
        axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        worst10 = latest_view.head(10).sort_values("coverage_ratio", ascending=True)
        allocated = worst10["fts_total_reported"] / 1e9
        gap = (worst10["req_sum"] - worst10["fts_total_reported"]).clip(lower=0) / 1e9
        y_pos = np.arange(len(worst10))
        axes[1].barh(y_pos, allocated, color="#2563eb", edgecolor="black", label="Allocated")
        axes[1].barh(y_pos, gap, left=allocated, color="#e5e7eb", edgecolor="black", label="Gap")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(worst10["country"])
        axes[1].set_xlabel("USD (billions)")
        axes[1].set_title(f"10 least-funded HRPs, {year_cov}")
        axes[1].legend()

        alloc_total = worst10["fts_total_reported"].replace(0, np.nan)
        paid_frac = worst10["fts_paid"] / alloc_total
        commit_frac = worst10["fts_commitment"] / alloc_total
        pledge_frac = worst10["fts_pledge"] / alloc_total
        axes[2].barh(y_pos, paid_frac, color="#10b981", edgecolor="black", label="Paid")
        axes[2].barh(y_pos, commit_frac, left=paid_frac, color="#60a5fa",
                     edgecolor="black", label="Committed")
        axes[2].barh(y_pos, pledge_frac, left=paid_frac + commit_frac, color="#fbbf24",
                     edgecolor="black", label="Pledged")
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(worst10["country"])
        axes[2].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[2].set_title(f"Composition of allocated funding, {year_cov}")
        axes[2].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Severity vs HRP Coverage")

        sev_data = core_enriched[["iso3", "year", "severity_index"]].dropna(subset=["severity_index"])
        df_sc = (hrp_fts.query("year == @year_cov and req_sum > 0")
                 .merge(sev_data, on=["iso3", "year"], how="left")
                 .dropna(subset=["severity_index"]).copy())
        if not df_sc.empty:
            df_sc["cov_ratio"] = df_sc["fts_total_reported"] / df_sc["req_sum"]
            df_sc["sev_pct"] = df_sc["severity_index"].rank(pct=True, method="average")
            df_sc["cov_pct"] = df_sc["cov_ratio"].rank(pct=True, method="average")
            df_sc["mismatch_sc"] = (df_sc["sev_pct"] + (1 - df_sc["cov_pct"])) / 2
            max_alloc = df_sc["fts_total_reported"].max()
            df_sc["bsize"] = 40 + (df_sc["fts_total_reported"] / max_alloc) * 900 if max_alloc > 0 else 80

            n_lbl_sc = st.slider("Countries to label", 3, 12, 7, key="sc_labels")
            fig, ax = plt.subplots(figsize=(12, 8))
            sc = ax.scatter(df_sc["severity_index"], df_sc["cov_ratio"] * 100,
                            s=df_sc["bsize"], c=df_sc["mismatch_sc"], cmap="RdYlGn_r",
                            alpha=0.8, edgecolors="black", linewidth=0.5, vmin=0, vmax=1)
            for _, row in df_sc.nlargest(n_lbl_sc, "mismatch_sc").iterrows():
                ax.annotate(row["country"], (row["severity_index"], row["cov_ratio"] * 100),
                            fontsize=9, color="#8b0000", fontweight="bold",
                            xytext=(4, 4), textcoords="offset points")
            for _, row in df_sc.nsmallest(n_lbl_sc, "mismatch_sc").iterrows():
                ax.annotate(row["country"], (row["severity_index"], row["cov_ratio"] * 100),
                            fontsize=9, color="#006400", fontweight="bold",
                            xytext=(4, 4), textcoords="offset points")
            ax.axvline(df_sc["severity_index"].median(), color="gray", linestyle="--", alpha=0.5)
            ax.axhline(df_sc["cov_ratio"].median() * 100, color="gray", linestyle=":", alpha=0.5)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_xlabel("INFORM Severity Index (0–5)", fontsize=12)
            ax.set_ylabel("HRP coverage ratio (allocated / requested)", fontsize=12)
            ax.set_title(f"Severity vs HRP Coverage ({year_cov})\n"
                         "Bubble size = allocated USD | Red = forgotten | Green = best-served")
            plt.colorbar(sc, ax=ax, label="Mismatch (0=best, 1=worst)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ── tab 6: validation ────────────────────────────────────────────────────────
with tab6:
    st.header("6. Validation — Mismatch vs CERF UFE List")
    st.markdown("""
CERF's **Underfunded Emergencies (UFE)** window is OCHA's own formal designation of
neglected crises. If our mismatch score captures the same signal, UFE-recipient
countries should cluster at the high end of our distribution.

We compare **2026 mismatch score** against **2025 CERF UFE designations**.
""")

    with st.spinner("Loading CERF data…"):
        hrp_fts_val = build_hrp_fts()
        cerf_cy = load_cerf(name_map_df)

    MISMATCH_YEAR = 2026
    CERF_YEAR = 2025

    sev_v = core_enriched[["iso3", "year", "severity_index"]].dropna(subset=["severity_index"])
    mis = (hrp_fts_val.query("year == @MISMATCH_YEAR and req_sum > 0")
           .merge(sev_v, on=["iso3", "year"], how="left")
           .dropna(subset=["severity_index"]).copy())

    if mis.empty:
        st.warning("No mismatch data for 2026.")
    else:
        mis["allocated_usd"] = mis["fts_total_reported"]
        mis["cov_ratio"] = mis["allocated_usd"] / mis["req_sum"]
        mis["sev_pct"] = mis["severity_index"].rank(pct=True, method="average")
        mis["cov_pct"] = mis["cov_ratio"].rank(pct=True, method="average")
        mis["mismatch"] = (mis["sev_pct"] + (1 - mis["cov_pct"])) / 2

        cerf_slice = cerf_cy[cerf_cy["year"] == CERF_YEAR][
            ["iso3", "received_ufe", "cerf_ufe_usd", "cerf_total_usd"]].copy()
        val_df = mis.merge(cerf_slice, on="iso3", how="left")
        val_df["received_ufe"] = val_df["received_ufe"].fillna(False).astype(bool)
        val_df["cerf_ufe_usd"] = val_df["cerf_ufe_usd"].fillna(0)
        val_df["cerf_total_usd"] = val_df["cerf_total_usd"].fillna(0)

        n_total = len(val_df)
        n_ufe = int(val_df["received_ufe"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Countries in analysis", n_total)
        c2.metric(f"Received CERF UFE ({CERF_YEAR})", n_ufe)
        c3.metric("Did not", n_total - n_ufe)

        ufe_scores = val_df.loc[val_df["received_ufe"], "mismatch"].values
        no_ufe_scores = val_df.loc[~val_df["received_ufe"], "mismatch"].values

        if len(ufe_scores) > 0 and len(no_ufe_scores) > 0:
            r_pb, p_pb = stats.pointbiserialr(val_df["received_ufe"].astype(int), val_df["mismatch"])
            u_stat, p_u = stats.mannwhitneyu(ufe_scores, no_ufe_scores, alternative="greater")
            med_ufe = np.median(ufe_scores)
            med_no = np.median(no_ufe_scores)

            c4, c5 = st.columns(2)
            c4.metric("Median mismatch — UFE group", f"{med_ufe:.3f}")
            c5.metric("Median mismatch — non-UFE group", f"{med_no:.3f}")
            st.write(f"Point-biserial correlation: **r = {r_pb:+.3f}** (p = {p_pb:.4f})")
            st.write(f"Mann–Whitney U (one-sided): **U = {u_stat:.0f}** (p = {p_u:.4f})")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        box_data = [no_ufe_scores, ufe_scores]
        bp = axes[0].boxplot(
            box_data,
            labels=[f"No UFE\n(n={len(no_ufe_scores)})", f"UFE recipient\n(n={len(ufe_scores)})"],
            widths=0.55, patch_artist=True, showfliers=False)
        for patch, colour in zip(bp["boxes"], ["#d1d5db", "#fca5a5"]):
            patch.set_facecolor(colour)
            patch.set_edgecolor("black")
        rng = np.random.default_rng(0)
        for scores, xpos in [(no_ufe_scores, 1), (ufe_scores, 2)]:
            jitter = rng.uniform(-0.12, 0.12, len(scores))
            axes[0].scatter(np.full(len(scores), xpos) + jitter, scores,
                            s=40, alpha=0.6, color="#374151", edgecolor="white",
                            linewidth=0.4, zorder=3)
        axes[0].set_ylabel("Mismatch score")
        axes[0].set_title(f"Mismatch by CERF UFE status\n"
                          f"Mismatch ({MISMATCH_YEAR}) vs UFE ({CERF_YEAR})")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].grid(axis="y", alpha=0.3)

        ufe_only = val_df[val_df["received_ufe"]].copy()
        if not ufe_only.empty:
            axes[1].scatter(ufe_only["cerf_ufe_usd"] / 1e6, ufe_only["mismatch"],
                            s=90, c=ufe_only["mismatch"], cmap="RdYlGn_r",
                            vmin=0, vmax=1, alpha=0.85, edgecolor="black", linewidth=0.5)
            for _, row in ufe_only.iterrows():
                axes[1].annotate(row["country"], (row["cerf_ufe_usd"] / 1e6, row["mismatch"]),
                                 fontsize=8, ha="left", va="bottom",
                                 xytext=(4, 4), textcoords="offset points")
        axes[1].set_xlabel(f"CERF UFE allocation, {CERF_YEAR} (USD millions)")
        axes[1].set_ylabel("Mismatch (2026) score")
        axes[1].set_title("Among UFE recipients: bigger allocation ↔ higher mismatch?")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"CERF UFE recipients ({CERF_YEAR}), ranked by mismatch")
            show = (val_df[val_df["received_ufe"]]
                    .sort_values("mismatch", ascending=False)
                    [["country", "severity_index", "cov_ratio", "mismatch", "cerf_ufe_usd"]]
                    .copy())
            show["severity_index"] = show["severity_index"].round(2)
            show["cov_ratio"] = (show["cov_ratio"] * 100).round(1).astype(str) + "%"
            show["mismatch"] = show["mismatch"].round(3)
            show["cerf_ufe_usd"] = show["cerf_ufe_usd"].apply(lambda x: "$" + _fmt(x))
            show.columns = ["Country", "Severity", "Coverage", "Mismatch", f"UFE ({CERF_YEAR})"]
            st.dataframe(show.reset_index(drop=True), hide_index=True)
        with col_right:
            st.subheader(f"High-mismatch countries NOT in UFE ({CERF_YEAR})")
            miss = (val_df[~val_df["received_ufe"]].nlargest(10, "mismatch")
                    [["country", "severity_index", "cov_ratio", "mismatch", "cerf_total_usd"]]
                    .copy())
            miss["severity_index"] = miss["severity_index"].round(2)
            miss["cov_ratio"] = (miss["cov_ratio"] * 100).round(1).astype(str) + "%"
            miss["mismatch"] = miss["mismatch"].round(3)
            miss["cerf_total_usd"] = miss["cerf_total_usd"].apply(lambda x: "$" + _fmt(x))
            miss.columns = ["Country", "Severity", "Coverage", "Mismatch", f"Any CERF ({CERF_YEAR})"]
            st.dataframe(miss.reset_index(drop=True), hide_index=True)
