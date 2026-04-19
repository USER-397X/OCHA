import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from google.cloud import bigquery

PROJECT = "ocha-493723"
client  = bigquery.Client(project=PROJECT)

CRISES = [
    {"name": "Ukraine",   "location": "Ukraine",    "local_lang": "Ukrainian",
     "local_srclc": "uk", "onset": "2022-02-24",
     "start": "20210801", "end": "20220801", "inform": 3.8},
    {"name": "Sudan",     "location": "Sudan",      "local_lang": "Arabic",
     "local_srclc": "ar", "onset": "2023-04-15",
     "start": "20221001", "end": "20231001", "inform": 4.9},
    {"name": "Palestine", "location": "Palestine",  "local_lang": "Arabic",
     "local_srclc": "ar", "onset": "2023-10-07",
     "start": "20230401", "end": "20240401", "inform": 4.5},
    {"name": "DR Congo",  "location": "Congo",      "local_lang": "French",
     "local_srclc": "fr", "onset": "2024-01-23",
     "start": "20230701", "end": "20240701", "inform": 4.3},
]

def fetch(location, lang_sql, start, end, label):
    print(f"  Querying {label}...", flush=True)
    q = f"""
    SELECT
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
      COUNTIF(V2Locations LIKE '%{location}%')                  AS article_count,
      COUNT(*)                                                    AS all_articles
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE DATE BETWEEN {start}000000 AND {end}235959
      AND ({lang_sql})
    GROUP BY date
    ORDER BY date
    """
    df = client.query(q).to_dataframe()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["frac"] = df["article_count"] / df["all_articles"].replace(0, np.nan)
    print(f"    {len(df)} rows, max frac={df['frac'].max():.5f}", flush=True)
    return df

EN_SQL    = "TranslationInfo = '' OR TranslationInfo IS NULL"
data = {}

for c in CRISES:
    local_sql = f"TranslationInfo LIKE '%srclc:{c['local_srclc']}%'"
    data[(c["name"], "en")]    = fetch(c["location"], EN_SQL,    c["start"], c["end"],
                                        f"{c['name']}/English")
    data[(c["name"], "local")] = fetch(c["location"], local_sql, c["start"], c["end"],
                                        f"{c['name']}/{c['local_lang']}")

# ─────────────────────────────────────────────────────────────
# Figure 1: time series 2×2
# ─────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False, sharey=False)
axes = axes.flatten()

for ax, c in zip(axes, CRISES):
    onset    = pd.Timestamp(c["onset"], tz="UTC")
    df_en    = data.get((c["name"], "en"))
    df_local = data.get((c["name"], "local"))

    if df_en is not None:
        ax.plot(df_en["date"],    df_en["frac"]    * 100,
                color="#1f77b4", linewidth=0.9, label="English (global)")
    if df_local is not None:
        ax.plot(df_local["date"], df_local["frac"] * 100,
                color="#ff7f0e", linewidth=0.9, label=f"{c['local_lang']} (local)")

    ax.axvline(onset, color="red", linestyle="--", linewidth=1.2)
    ax.text(onset, ax.get_ylim()[1] * 0.95, "  onset", color="red", fontsize=7, va="top")
    ax.set_title(f"{c['name']}  (INFORM {c['inform']})", fontsize=10)
    ax.set_ylabel("% of language's news", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

fig1.suptitle("Global vs local media coverage of humanitarian crises\n"
              "(coverage fraction = daily articles about crisis / total articles in that language)  [BigQuery]",
              fontsize=10)
fig1.tight_layout()
fig1.savefig("scratch/media_bias_fig1_timeseries.png", dpi=150)
print("Saved media_bias_fig1_timeseries.png")

# ─────────────────────────────────────────────────────────────
# Figure 2: visibility scatter
# ─────────────────────────────────────────────────────────────
def post_mean(df, onset_str, days=30):
    if df is None or df.empty:
        return np.nan
    onset  = pd.Timestamp(onset_str, tz="UTC")
    window = df[(df["date"] >= onset) & (df["date"] < onset + pd.Timedelta(days=days))]
    return np.nan if len(window) < 5 else window["frac"].mean()

en_vals    = [post_mean(data.get((c["name"], "en")),    c["onset"]) for c in CRISES]
local_vals = [post_mean(data.get((c["name"], "local")), c["onset"]) for c in CRISES]

fig2, ax2 = plt.subplots(figsize=(7, 6))
valid = [(e, l, c["inform"], c["name"])
         for e, l, c in zip(en_vals, local_vals, CRISES)
         if not (np.isnan(e) or np.isnan(l))]

if valid:
    ev, lv, iv, nv = zip(*valid)
    sizes = [(i / 5) ** 2 * 1000 for i in iv]
    ax2.scatter(ev, lv, s=sizes, color="#333333", alpha=0.7, zorder=3)
    for x, y, name in zip(ev, lv, nv):
        ax2.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    lim_min = min(min(ev), min(lv)) * 0.4
    lim_max = max(max(ev), max(lv)) * 3
    diag = np.array([lim_min, lim_max])
    ax2.plot(diag, diag, "k--", linewidth=0.8, alpha=0.4, label="Equal coverage")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("English coverage fraction — 30-day post-crisis mean", fontsize=10)
    ax2.set_ylabel("Local-language coverage fraction — 30-day post-crisis mean", fontsize=10)
    ax2.set_title("Visibility map: global vs local media attention\n"
                  "(dot size ∝ INFORM Severity;  above diagonal = ignored by West)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", linewidth=0.3, alpha=0.4)
else:
    ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
             transform=ax2.transAxes)

fig2.tight_layout()
fig2.savefig("scratch/media_bias_fig2_scatter.png", dpi=150)
print("Saved media_bias_fig2_scatter.png")

# ─────────────────────────────────────────────────────────────
# Figure 3: INFORM severity vs English coverage scatter
# The money shot: higher severity ≠ more Western attention
# ─────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))

valid3 = [(c["inform"], e, c["name"])
          for c, e in zip(CRISES, en_vals) if not np.isnan(e)]

if valid3:
    informs, eng_fracs, names = zip(*valid3)
    colours3 = ["#d62728" if n == "Ukraine" else "#1f77b4" for n in names]
    ax3.scatter(informs, [f * 100 for f in eng_fracs],
                s=200, c=colours3, zorder=3, edgecolors="white", linewidth=1.5)
    for inf, frac, name in zip(informs, eng_fracs, names):
        ax3.annotate(name, (inf, frac * 100),
                     textcoords="offset points", xytext=(8, 4), fontsize=11,
                     fontweight="bold")

    # Dashed trend line (if coverage tracked severity, it would slope up)
    inform_range = np.array([min(informs) - 0.2, max(informs) + 0.2])
    max_frac_pct = max(f * 100 for f in eng_fracs)
    # Hypothetical "fair" line: proportional to INFORM severity
    fair_slope = max_frac_pct / max(informs)
    ax3.plot(inform_range, fair_slope * inform_range,
             "k--", linewidth=0.8, alpha=0.35, label="Expected if coverage ∝ severity")

    ax3.set_xlabel("INFORM Severity Index  (higher = worse humanitarian crisis)", fontsize=11)
    ax3.set_ylabel("English media coverage — 30-day post-onset\n(% of all English GDELT articles)", fontsize=10)
    ax3.set_title("Western media attention vs humanitarian severity\n"
                  "Higher severity does NOT predict more English coverage  [GDELT via BigQuery]",
                  fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(linewidth=0.3, alpha=0.4)
    ax3.set_yscale("log")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))

fig3.tight_layout()
fig3.savefig("scratch/media_bias_fig3_bias.png", dpi=150)
print("Saved media_bias_fig3_bias.png")

print("\n=== Bias summary ===")
print(f"{'Crisis':<14} {'INFORM':>6} {'English%':>10} {'Local%':>10} {'Bias(log10)':>12}")
for c, e, l in sorted(zip(CRISES, en_vals, local_vals), key=lambda x: -x[0]["inform"]):
    ep = f"{e*100:.4f}%" if not np.isnan(e) else "  n/a"
    lp = f"{l*100:.4f}%" if not np.isnan(l) else "  n/a"
    ratio_str = f"{np.log10(e/l):+.2f}" if not (np.isnan(e) or np.isnan(l) or l == 0) else "  n/a"
    print(f"{c['name']:<14} {c['inform']:>6.1f} {ep:>10} {lp:>10} {ratio_str:>12}")
