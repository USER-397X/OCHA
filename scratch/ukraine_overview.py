"""
GDELT via BigQuery — Ukraine data overview.
Replaces the original gdeltdoc-based ukraine_overview.py.

Sections:
  1. Raw article volume        (COUNTIF match / COUNT total)
  2. Normalized volume         (frac = match / total)
  3. Sentiment / tone          (AVG of first V2Tone field)
  4. Language breakdown        (group by srclc language code)
  5. Source-country breakdown  (group by first V2Locations country)
  6. Article listing           (DocumentIdentifier + tone sample)
  7. Theme filter              (V2Themes LIKE)

Run:  python scratch/ukraine_overview.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from google.cloud import bigquery

PROJECT  = "ocha-493723"
client   = bigquery.Client(project=PROJECT)
INVASION = pd.Timestamp("2022-02-24")
START    = "20220101"
END      = "20230101"
DATE_RANGE = f"DATE BETWEEN {START}000000 AND {END}235959"
EN_FILTER  = "TranslationInfo = '' OR TranslationInfo IS NULL"

def q(sql):
    df = client.query(sql).to_dataframe()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def sec(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ───────────────────────────────────────────────────────────────
sec("SECTION 1 — Raw article volume")
print("  What: Actual daily count of English articles mentioning 'Ukraine'.")
print("  Use:  Absolute intensity signal.")

df_raw = q(f"""
SELECT
  PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
  COUNTIF(V2Locations LIKE '%Ukraine%') AS article_count,
  COUNT(*) AS all_articles
FROM `gdelt-bq.gdeltv2.gkg`
WHERE {DATE_RANGE} AND ({EN_FILTER})
GROUP BY date ORDER BY date
""")
df_raw = df_raw[df_raw["article_count"] > 0]
pre  = df_raw.loc[df_raw["date"] < INVASION, "article_count"].mean()
peak = df_raw["article_count"].max()
print(f"  Pre-invasion daily mean : {pre:.0f} articles")
print(f"  Post-invasion peak      : {peak:.0f} articles")
print(f"  Spike ratio             : {peak/pre:.1f}x baseline")


# ───────────────────────────────────────────────────────────────
sec("SECTION 2 — Normalized volume (coverage fraction)")
print("  What: Ukraine articles as fraction of all English GDELT articles.")
print("  Use:  Controls for total news-volume fluctuations.")

df_raw["frac"] = df_raw["article_count"] / df_raw["all_articles"]
pre_pct  = df_raw.loc[df_raw["date"] < INVASION, "frac"].mean() * 100
peak_pct = df_raw["frac"].max() * 100
print(f"  Pre-invasion share  : {pre_pct:.4f}% of all English news")
print(f"  Post-invasion peak  : {peak_pct:.4f}% of all English news")


# ───────────────────────────────────────────────────────────────
sec("SECTION 3 — Tone / sentiment")
print("  What: Average tone score per day (V2Tone, field 1).")
print("  Use:  Detect escalation and framing shifts.")

df_tone = q(f"""
SELECT
  PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_ORDINAL(1)] AS FLOAT64)) AS avg_tone
FROM `gdelt-bq.gdeltv2.gkg`
WHERE {DATE_RANGE}
  AND ({EN_FILTER})
  AND V2Locations LIKE '%Ukraine%'
  AND V2Tone IS NOT NULL AND V2Tone != ''
GROUP BY date ORDER BY date
""")
print(df_tone["avg_tone"].describe().round(3).to_string())
worst = df_tone.loc[df_tone["avg_tone"].idxmin()]
print(f"  Most negative day: {worst['date'].date()}  tone={worst['avg_tone']:.2f}")


# ───────────────────────────────────────────────────────────────
sec("SECTION 4 — Language breakdown")
print("  What: Article counts split by source language (via TranslationInfo srclc code).")
print("  Use:  Which language communities are covering Ukraine?")

df_lang_raw = q(f"""
SELECT
  IFNULL(REGEXP_EXTRACT(TranslationInfo, r'srclc:([a-z]+)'), 'en') AS lang,
  COUNT(*) AS articles
FROM `gdelt-bq.gdeltv2.gkg`
WHERE {DATE_RANGE}
  AND V2Locations LIKE '%Ukraine%'
GROUP BY lang ORDER BY articles DESC
LIMIT 15
""")
print("\n  Top 15 languages by total article volume:")
print(df_lang_raw.to_string(index=False))


# ───────────────────────────────────────────────────────────────
sec("SECTION 5 — Source-country breakdown")
print("  What: Articles grouped by source country (first location's FIPS code).")

df_sc = q(f"""
SELECT
  REGEXP_EXTRACT(SPLIT(V2Locations, ';')[SAFE_ORDINAL(1)], r'#([A-Z]{{2}})#') AS country,
  COUNT(*) AS articles
FROM `gdelt-bq.gdeltv2.gkg`
WHERE {DATE_RANGE}
  AND ({EN_FILTER})
  AND V2Locations LIKE '%Ukraine%'
GROUP BY country
HAVING country IS NOT NULL
ORDER BY articles DESC
LIMIT 15
""")
print("\n  Top 15 source countries (FIPS codes):")
print(df_sc.to_string(index=False))


# ───────────────────────────────────────────────────────────────
sec("SECTION 6 — Article listing (invasion day)")
print("  What: Sample articles from 2022-02-24 (DocumentIdentifier = URL).")

df_art = q(f"""
SELECT
  DocumentIdentifier AS url,
  SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_ORDINAL(1)] AS FLOAT64) AS tone,
  SourceCommonName AS domain
FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN 20220224000000 AND 20220224235959
  AND ({EN_FILTER})
  AND V2Locations LIKE '%Ukraine%'
ORDER BY tone
LIMIT 20
""")
print(f"\n  Articles on invasion day: {len(df_art)}")
print(df_art[["domain", "tone"]].head(10).to_string(index=False))


# ───────────────────────────────────────────────────────────────
sec("SECTION 7 — Theme filter (GKG themes)")
print("  What: Articles tagged with specific GDELT GKG themes.")

for theme in ["ARMEDCONFLICT", "UNGP_HUMANITARIAN_AID", "REFUGEE"]:
    df_t = q(f"""
    SELECT
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
      COUNT(*) AS articles
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE DATE BETWEEN 20220201000000 AND 20220601235959
      AND V2Locations LIKE '%Ukraine%'
      AND REGEXP_CONTAINS(Themes, r'(^|;){theme}(;|$)')
    GROUP BY date ORDER BY date
    """)
    peak_t = df_t["articles"].max() if not df_t.empty else 0
    total_t = df_t["articles"].sum() if not df_t.empty else 0
    print(f"  [{theme:30s}]  peak={peak_t:.0f}  total={total_t:.0f}")


# ───────────────────────────────────────────────────────────────
sec("BUILDING SUMMARY FIGURE — 4-panel overview")

fig, axes = plt.subplots(4, 1, figsize=(13, 18), sharex=False)
fig.suptitle("GDELT (BigQuery) — Ukraine Data Overview\n(2022-01-01 to 2023-01-01)",
             fontsize=13, fontweight="bold", y=0.99)

def add_vline(ax):
    ax.axvline(INVASION, color="red", linestyle="--", linewidth=1.2)
    ax.text(INVASION + pd.Timedelta(days=2), ax.get_ylim()[1] * 0.88,
            "24 Feb\ninvasion", color="red", fontsize=7, va="top")

ax = axes[0]
ax.fill_between(df_raw["date"], df_raw["article_count"], alpha=0.2, color="#1f77b4")
ax.plot(df_raw["date"], df_raw["article_count"], color="#1f77b4", lw=0.9)
add_vline(ax)
ax.set_title("1. Raw article count", loc="left", fontsize=9)
ax.set_ylabel("Articles / day")

ax = axes[1]
ax.plot(df_tone["date"], df_tone["avg_tone"], color="#d62728", lw=0.9)
ax.axhline(0, color="grey", lw=0.5, linestyle=":")
add_vline(ax)
ax.set_title("2. Average tone — more negative = crisis framing", loc="left", fontsize=9)
ax.set_ylabel("Tone score")

ax = axes[2]
ax.text(0.5, 0.5, "Language breakdown: see Section 4 output above\n"
        "(BigQuery aggregated — time series not shown here)",
        ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
ax.set_title("3. Language breakdown", loc="left", fontsize=9)

ax = axes[3]
ax.text(0.5, 0.5, "Source-country breakdown: see Section 5 output above\n"
        "(BigQuery aggregated — time series not shown here)",
        ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
ax.set_title("4. Source-country breakdown", loc="left", fontsize=9)

for ax in axes[:2]:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("scratch/ukraine_overview.png", dpi=150)
print("\nSaved scratch/ukraine_overview.png")

print("""
DATA DIMENSIONS available for any country/crisis (BigQuery GKG):
  1.  Raw count         → COUNTIF(V2Locations LIKE '%Country%') per day
  2.  Normalized        → article_count / all_articles
  3.  Tone              → AVG(SPLIT(V2Tone,',')[1]) per day
  4.  Language split    → GROUP BY REGEXP_EXTRACT(TranslationInfo, 'srclc:XX')
  5.  Source country    → REGEXP_EXTRACT first V2Locations FIPS code
  6.  Article listing   → SELECT DocumentIdentifier (URL) LIMIT N
  7.  Theme filter      → REGEXP_CONTAINS(Themes, '(^|;)THEME(;|$)')
  8.  Tone filter       → HAVING avg_tone < threshold
""")
