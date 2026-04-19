"""
Supply-Demand Alignment Analysis
=================================
Computes alignment ratios (funding share / need share) per country-sector,
then rolls up to country-level alignment scores.

NOTE: The HNO dataset lacks a year column. Where multiple national-level
entries exist per country-cluster, we take the *mean* People-in-Need as a
representative estimate.  Funding (CERF + CBPF) is summed across all years.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Cluster code ↔ full-name mapping ────────────────────────────────────────
CLUSTER_CODE_TO_NAME = {
    "EDU": "Education",
    "FSC": "Food Security",
    "HEA": "Health",
    "NUT": "Nutrition",
    "PRO": "Protection",
    "SHL": "Shelter/NFI",
    "WSH": "WASH",
    "CCM": "CCCM",
    "ERY": "Early Recovery",
    "MS":  "Multi-Sector",
    "MPC": "Multipurpose Cash",
}

# CBPF PooledFundName → ISO3
CBPF_COUNTRY_TO_ISO3 = {
    "Afghanistan": "AFG", "Bangladesh": "BGD", "Burkina Faso": "BFA",
    "CAR": "CAF", "Chad (RhPF)": "TCD", "Colombia": "COL",
    "Colombia (RhPF)": "COL", "DRC": "COD", "Ethiopia": "ETH",
    "Fiji": "FJI", "Haiti": "HTI", "Haiti (RhPF)": "HTI",
    "Iraq": "IRQ", "Jordan": "JOR", "Kenya": "KEN", "Lebanon": "LBN",
    "Mali": "MLI", "Mozambique (RhPF)": "MOZ", "Myanmar": "MMR",
    "Niger": "NER", "Nigeria": "NGA", "Pakistan": "PAK",
    "Somalia": "SOM", "South Sudan": "SSD", "Sudan": "SDN",
    "Syria": "SYR", "Syria Cross border": "SYR", "Uganda": "UGA",
    "Ukraine": "UKR", "Venezuela": "VEN", "Yemen": "YEM", "oPt": "PSE",
}


# ── 1. Load HNO needs ───────────────────────────────────────────────────────
def load_hno_needs() -> pd.DataFrame:
    """Return DataFrame with columns [country, cluster, pin] at national level."""
    df = pd.read_csv(DATA_DIR / "hno_comb_cleaned.csv", low_memory=False)
    df.columns = df.columns.str.strip()

    # Keep national-level rows (no admin breakdown) with empty Category
    mask = (
        df["Admin 1 PCode"].isna()
        & df["Admin 2 PCode"].isna()
        & df["Admin 3 PCode"].isna()
        & df["Category"].isna()
    )
    nat = df[mask][["Country ISO3", "Cluster", "In Need"]].copy()
    nat = nat[nat["Country ISO3"] != "#country+code"]
    nat["In Need"] = pd.to_numeric(nat["In Need"], errors="coerce")
    nat = nat.dropna(subset=["In Need"])
    nat = nat[nat["In Need"] > 0]

    # Drop clusters we don't want
    nat = nat[~nat["Cluster"].isin(["Multipurpose Cash", "Early Recovery"])]

    # For Protection: prefer "Protection (overall)" over sub-types when duplicated
    # We simply average across duplicate entries per country-cluster (multi-year)
    needs = (
        nat.groupby(["Country ISO3", "Cluster"], as_index=False)["In Need"]
        .mean()
        .rename(columns={"Country ISO3": "country", "Cluster": "cluster", "In Need": "pin"})
    )
    return needs


# ── 2. Load CERF funding ────────────────────────────────────────────────────
def load_cerf() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cerf_allocations_cleaned.csv", low_memory=False)
    df.columns = df.columns.str.strip()
    df["totalAmountApproved"] = pd.to_numeric(df["totalAmountApproved"], errors="coerce").fillna(0)
    df["cluster_name"] = df["Cluster"].map(CLUSTER_CODE_TO_NAME)
    df = df[~df["cluster_name"].isin(["Multipurpose Cash", "Early Recovery"])]
    out = (
        df.groupby(["countryCode", "cluster_name"], as_index=False)["totalAmountApproved"]
        .sum()
        .rename(columns={"countryCode": "country", "cluster_name": "cluster",
                         "totalAmountApproved": "funding"})
    )
    return out.dropna(subset=["cluster"])


# ── 3. Load CBPF funding ────────────────────────────────────────────────────
def load_cbpf() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "cbpf_allocations_clean.csv", low_memory=False)
    df.columns = df.columns.str.strip()
    df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce").fillna(0)
    df["ClusterPercentage"] = pd.to_numeric(df["ClusterPercentage"], errors="coerce").fillna(100)
    # Weighted budget for this cluster
    df["cluster_funding"] = df["Budget"] * df["ClusterPercentage"] / 100.0
    df["country"] = df["PooledFundName"].map(CBPF_COUNTRY_TO_ISO3)
    df["cluster_name"] = df["Cluster"].map(CLUSTER_CODE_TO_NAME)
    df = df[~df["cluster_name"].isin(["Multipurpose Cash", "Early Recovery"])]
    out = (
        df.groupby(["country", "cluster_name"], as_index=False)["cluster_funding"]
        .sum()
        .rename(columns={"cluster_name": "cluster", "cluster_funding": "funding"})
    )
    return out.dropna(subset=["country", "cluster"])


# ── 4. Combine funding ──────────────────────────────────────────────────────
def load_combined_funding() -> pd.DataFrame:
    cerf = load_cerf()
    cbpf = load_cbpf()
    combined = pd.concat([cerf, cbpf], ignore_index=True)
    return combined.groupby(["country", "cluster"], as_index=False)["funding"].sum()


# ── 5. Compute alignment ratios ─────────────────────────────────────────────
def compute_alignment(needs: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """
    Merge needs and funding, compute shares and alignment ratios.
    Returns DataFrame with: country, cluster, pin, funding, need_share,
    funding_share, alignment_ratio.
    """
    merged = needs.merge(funding, on=["country", "cluster"], how="inner")

    # Compute shares within each country
    country_totals = merged.groupby("country").agg(
        total_pin=("pin", "sum"),
        total_funding=("funding", "sum"),
    )
    merged = merged.join(country_totals, on="country")
    merged["need_share"] = merged["pin"] / merged["total_pin"]
    merged["funding_share"] = merged["funding"] / merged["total_funding"]
    merged["alignment_ratio"] = merged["funding_share"] / merged["need_share"]

    # Drop countries with only 1 sector (ratios are trivially 1.0)
    sector_counts = merged.groupby("country")["cluster"].transform("count")
    merged = merged[sector_counts >= 2].copy()

    return merged.drop(columns=["total_pin", "total_funding"])


# ── 6. Country-level alignment score ────────────────────────────────────────
def country_alignment_score(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Needs-weighted log-deviation formula → single score per country.
    alignment_score = exp(-Σ |log(ratio)| × need_share)
    """
    df = alignment_df.copy()
    # Clamp ratio to avoid log(0)
    df["ratio_clamped"] = df["alignment_ratio"].clip(lower=1e-6)
    df["abs_log_dev"] = np.abs(np.log(df["ratio_clamped"]))
    df["weighted_dev"] = df["abs_log_dev"] * df["need_share"]
    misalignment = df.groupby("country")["weighted_dev"].sum().reset_index()
    misalignment.columns = ["country", "misalignment"]
    misalignment["alignment_score"] = np.exp(-misalignment["misalignment"])
    return misalignment


# ── 7. Plotting ─────────────────────────────────────────────────────────────

def plot_country_heatmap(alignment_df: pd.DataFrame) -> None:
    """Heatmap: country × cluster with alignment ratio as colour."""
    pivot = alignment_df.pivot_table(
        index="country", columns="cluster", values="alignment_ratio"
    )
    # Sort by mean alignment (most misaligned at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.45)))
    norm = mcolors.LogNorm(vmin=0.05, vmax=20)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red = under, green = over

    sns.heatmap(
        pivot, ax=ax, cmap=cmap, norm=norm, center=1.0,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Alignment Ratio (1.0 = proportional)"},
    )
    ax.set_title("Funding–Need Alignment by Country and Sector\n"
                 "(ratio = funding share ÷ need share; 1.0 = fair share)",
                 fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "alignment_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'alignment_heatmap.png'}")


def plot_global_sector_alignment(alignment_df: pd.DataFrame) -> None:
    """
    Bar chart: average alignment ratio per sector across all countries,
    weighted by each country-sector's need share in the global total.
    """
    df = alignment_df.copy()
    global_pin = df["pin"].sum()
    df["global_need_weight"] = df["pin"] / global_pin

    # Weighted mean alignment ratio per cluster
    df["weighted_ratio"] = df["alignment_ratio"] * df["global_need_weight"]
    sector_stats = df.groupby("cluster").agg(
        weighted_ratio=("weighted_ratio", "sum"),
        global_weight=("global_need_weight", "sum"),
    )
    sector_stats["mean_alignment"] = sector_stats["weighted_ratio"] / sector_stats["global_weight"]
    sector_stats = sector_stats.sort_values("mean_alignment")

    colors = ["#d32f2f" if v < 1 else "#388e3c" for v in sector_stats["mean_alignment"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sector_stats.index, sector_stats["mean_alignment"], color=colors, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Proportional (1.0)")
    ax.set_xlabel("Needs-Weighted Mean Alignment Ratio")
    ax.set_title("Global Sector Alignment: Which Clusters Are Systematically\n"
                 "Underfunded or Overfunded Relative to Need?", fontsize=13)
    ax.legend()

    # Annotate bars
    for bar, val in zip(bars, sector_stats["mean_alignment"]):
        label = f"{val:.2f}"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "global_sector_alignment.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'global_sector_alignment.png'}")


def plot_sector_boxplot(alignment_df: pd.DataFrame) -> None:
    """Box plot showing distribution of alignment ratios per sector across countries."""
    df = alignment_df.copy()
    df["log_ratio"] = np.log2(df["alignment_ratio"].clip(lower=1e-6))

    order = df.groupby("cluster")["log_ratio"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="cluster", y="log_ratio", order=order, ax=ax,
                palette="RdYlGn", showfliers=True)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("log₂(Alignment Ratio)\n← underfunded | overfunded →")
    ax.set_xlabel("")
    ax.set_title("Distribution of Alignment Ratios by Sector Across Countries", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sector_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'sector_boxplot.png'}")


def plot_country_scores(scores: pd.DataFrame) -> None:
    """Horizontal bar chart of country-level alignment scores."""
    scores = scores.sort_values("alignment_score")
    colors = plt.cm.RdYlGn(scores["alignment_score"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(scores) * 0.35)))
    ax.barh(scores["country"], scores["alignment_score"], color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Alignment Score (1.0 = perfectly proportional)")
    ax.set_title("Country-Level Funding Alignment Score\n"
                 "(higher = funding more proportional to need)", fontsize=13)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.5)

    for i, (_, row) in enumerate(scores.iterrows()):
        ax.text(row["alignment_score"] + 0.01, i, f"{row['alignment_score']:.2f}",
                va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "country_alignment_scores.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'country_alignment_scores.png'}")


def plot_scatter_ratio_vs_need(alignment_df: pd.DataFrame) -> None:
    """Scatter: need share (x) vs alignment ratio (y) per sector, coloured by cluster."""
    df = alignment_df.copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    clusters = df["cluster"].unique()
    palette = sns.color_palette("tab10", n_colors=len(clusters))
    cluster_colors = dict(zip(sorted(clusters), palette))

    for cluster, group in df.groupby("cluster"):
        ax.scatter(group["need_share"], group["alignment_ratio"],
                   label=cluster, alpha=0.7, s=50, color=cluster_colors[cluster])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Fair share (1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("Need Share (fraction of country's total PiN)")
    ax.set_ylabel("Alignment Ratio (log scale)")
    ax.set_title("Need Share vs Alignment Ratio by Sector", fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter_need_vs_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'scatter_need_vs_alignment.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    needs = load_hno_needs()
    funding = load_combined_funding()

    print(f"  HNO needs: {len(needs)} country-cluster pairs")
    print(f"  Combined funding: {len(funding)} country-cluster pairs")

    print("\nComputing alignment ratios...")
    alignment = compute_alignment(needs, funding)
    print(f"  Matched: {len(alignment)} country-cluster pairs "
          f"across {alignment['country'].nunique()} countries")

    print("\nComputing country-level scores...")
    scores = country_alignment_score(alignment)
    print(scores.sort_values("alignment_score").to_string(index=False))

    print("\nGenerating plots...")
    plot_country_heatmap(alignment)
    plot_global_sector_alignment(alignment)
    plot_sector_boxplot(alignment)
    plot_country_scores(scores)
    plot_scatter_ratio_vs_need(alignment)

    # Print summary table: most under/overfunded sectors globally
    print("\n── Global Sector Summary ──")
    global_pin = alignment["pin"].sum()
    alignment["global_weight"] = alignment["pin"] / global_pin
    alignment["w_ratio"] = alignment["alignment_ratio"] * alignment["global_weight"]
    sector_summary = alignment.groupby("cluster").agg(
        w_ratio=("w_ratio", "sum"),
        weight=("global_weight", "sum"),
        n_countries=("country", "nunique"),
    )
    sector_summary["mean_alignment"] = sector_summary["w_ratio"] / sector_summary["weight"]
    sector_summary = sector_summary.sort_values("mean_alignment")
    for _, row in sector_summary.iterrows():
        status = "UNDER" if row["mean_alignment"] < 1 else "OVER "
        print(f"  {status}  {row.name:<20s}  ratio={row['mean_alignment']:.3f}  "
              f"(across {int(row['n_countries'])} countries)")

    print(f"\nDone. All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
