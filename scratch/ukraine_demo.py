import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from google.cloud import bigquery

PROJECT = "ocha-493723"
client  = bigquery.Client(project=PROJECT)

print("Fetching Ukraine / English article counts from GDELT BigQuery...")

df = client.query("""
    SELECT
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
      COUNTIF(V2Locations LIKE '%Ukraine%') AS article_count,
      COUNT(*) AS all_articles
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE DATE BETWEEN 20210601000000 AND 20230601235959
      AND (TranslationInfo = '' OR TranslationInfo IS NULL)
    GROUP BY date
    ORDER BY date
""").to_dataframe()

df["date"] = pd.to_datetime(df["date"])
df = df[df["article_count"] > 0].reset_index(drop=True)
print(f"Got {len(df)} rows. Peak: {df['article_count'].max():,} articles/day")

invasion_date = pd.Timestamp("2022-02-24")
baseline_mean = df.loc[df["date"] < invasion_date, "article_count"].mean()
peak_row      = df.loc[df["article_count"].idxmax()]
ratio         = peak_row["article_count"] / baseline_mean

print(f"\nPeak date:              {peak_row['date'].date()}")
print(f"Peak article count:     {int(peak_row['article_count']):,}")
print(f"Peak / baseline ratio:  {ratio:.1f}x")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["article_count"], color="#111111", linewidth=0.8)
ax.axvline(invasion_date, color="red", linestyle="--", linewidth=1.4)
ax.text(invasion_date, ax.get_ylim()[1] * 0.97,
        "Invasion (24 Feb 2022)", color="red", fontsize=9, ha="left", va="top")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right")
ax.set_xlabel("Date")
ax.set_ylabel("Daily article count (raw)")
ax.set_title("GDELT English-language articles mentioning 'Ukraine'  [via BigQuery]")
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig("scratch/ukraine_demo.png", dpi=150)
print("\nSaved scratch/ukraine_demo.png")
