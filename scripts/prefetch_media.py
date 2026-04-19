"""Pre-fetch GDELT media attention for all crisis countries via BigQuery.

Scans gdelt-bq.gdeltv2.gkg once per country (English articles only).
Saves results to data/media/{ISO3}.csv so the Streamlit app has data
immediately on startup without hitting the API live.

Usage:
    uv run scripts/prefetch_media.py                    # all countries, 4 workers
    uv run scripts/prefetch_media.py --workers 8        # faster, more BigQuery slots
    uv run scripts/prefetch_media.py --iso3 SDN,UKR     # specific countries only
    uv run scripts/prefetch_media.py --force            # re-fetch even if cached
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from google.cloud import bigquery

from media import MEDIA_DIR, _load_csv, _save_csv

DATA_DIR = Path(__file__).parent.parent / "data"
PROJECT  = "ocha-493723"

DEFAULT_START = "2021-01-01"
DEFAULT_END   = "2026-03-31"
DEFAULT_WORKERS = 4

# Countries where the first word of the COUNTRY name is ambiguous or misleading.
# Value: the search keyword(s) to use in the V2Locations LIKE clause.
_KEYWORD_OVERRIDES: dict[str, str] = {
    "COD": "Congo Democratic",      # avoid matching Republic of Congo
    "COG": "Congo Brazzaville",
    "PRK": "North Korea",
    "KOR": "South Korea",
    "SSD": "South Sudan",
    "PSE": "Palestine",
    "CAF": "Central African",
    "TLS": "Timor-Leste",
    "TCD": "Chad",                  # COUNTRY = "Chad" but split()[0] = "Chad" ✓
    "GNB": "Guinea-Bissau",
    "GNQ": "Equatorial Guinea",
    "SLV": "El Salvador",
    "DOM": "Dominican Republic",
    "LAO": "Laos",
}


def _country_list() -> list[tuple[str, str]]:
    """Return [(iso3, keyword), ...] for all single-ISO3 crisis countries."""
    gap = pd.read_csv(DATA_DIR / "country_year_severity_funding.csv")
    sev = pd.read_csv(DATA_DIR / "inform_severity_cleaned.csv")
    fts = pd.read_csv(DATA_DIR / "fts_requirements_funding_global.csv")
    gap.columns = gap.columns.str.strip()
    sev.columns = sev.columns.str.strip()
    fts.columns = fts.columns.str.strip()

    name_map: dict[str, str] = (
        sev[["ISO3", "COUNTRY"]].dropna().drop_duplicates("ISO3")
        .set_index("ISO3")["COUNTRY"].to_dict()
    )

    all_iso3 = (
        set(gap["Country_ISO3"].dropna().unique())
        | set(sev["ISO3"].dropna().unique())
        | set(fts["countryCode"].dropna().unique())
    )

    results = []
    for iso3 in sorted(all_iso3):
        if "," in iso3 or " " in iso3:
            continue  # skip multi-country stub entries
        keyword = _KEYWORD_OVERRIDES.get(iso3) or (
            name_map[iso3].split()[0] if iso3 in name_map else None
        )
        if keyword:
            results.append((iso3, keyword))

    return results


def _fetch_bq(
    keyword: str,
    start: str,
    end: str,
    client: bigquery.Client,
) -> pd.DataFrame:
    """Query GDELT BigQuery for daily English article counts mentioning *keyword*."""
    start_int = start.replace("-", "")
    end_int   = end.replace("-", "")
    kw_safe   = keyword.replace("'", "''")   # basic SQL-injection guard

    # For multi-word keywords, require all words to appear
    like_clauses = " AND ".join(
        f"V2Locations LIKE '%{word.replace(chr(39), chr(39)+chr(39))}%'"
        for word in kw_safe.split()
    )

    query = f"""
    SELECT
      PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
      COUNTIF({like_clauses})  AS articles,
      COUNT(*)                  AS total
    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE DATE BETWEEN {start_int}000000 AND {end_int}235959
      AND (TranslationInfo = '' OR TranslationInfo IS NULL)
    GROUP BY date
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["frac"] = df["articles"] / df["total"].replace(0, np.nan)
    return df[["date", "articles", "frac"]].sort_values("date").reset_index(drop=True)


def _needs_fetch(iso3: str, end: str, force: bool) -> bool:
    if force:
        return True
    existing = _load_csv(iso3)
    if existing is None or existing.empty:
        return True
    return existing["date"].max().date().isoformat() < end


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start",   default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel BigQuery queries (default: {DEFAULT_WORKERS})")
    parser.add_argument("--iso3",    default="",
                        help="Comma-separated ISO3 codes to fetch (default: all)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-fetch even when CSV already covers the date range")
    args = parser.parse_args()

    all_countries = _country_list()

    if args.iso3:
        target = {c.strip().upper() for c in args.iso3.split(",")}
        all_countries = [(iso3, kw) for iso3, kw in all_countries if iso3 in target]
        missing = target - {iso3 for iso3, _ in all_countries}
        if missing:
            print(f"WARNING: unknown ISO3 codes (no name mapping): {', '.join(sorted(missing))}")

    to_fetch = [
        (iso3, kw) for iso3, kw in all_countries
        if _needs_fetch(iso3, args.end, args.force)
    ]
    skipped = len(all_countries) - len(to_fetch)

    print(f"Countries total : {len(all_countries)}")
    print(f"Already cached  : {skipped}")
    print(f"To fetch        : {len(to_fetch)}")
    print(f"Date range      : {args.start} → {args.end}")
    print(f"Workers         : {args.workers}")
    print()

    if not to_fetch:
        print("Nothing to fetch — all countries already cached.")
        return

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=PROJECT)

    def _worker(item: tuple[str, str]) -> tuple[str, str, int, str | None]:
        iso3, keyword = item
        try:
            new_df = _fetch_bq(keyword, args.start, args.end, client)
            existing = _load_csv(iso3)
            if existing is not None and not existing.empty:
                new_df = (
                    pd.concat([existing, new_df])
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            _save_csv(iso3, new_df)
            return iso3, keyword, len(new_df), None
        except Exception as exc:
            return iso3, keyword, 0, str(exc)

    done = 0
    total = len(to_fetch)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, item): item for item in to_fetch}
        for fut in as_completed(futs):
            iso3, keyword, n_rows, err = fut.result()
            done += 1
            status = "ERROR" if err else "OK   "
            detail = err if err else f"{n_rows} rows"
            print(f"  {status} [{done:3d}/{total}] {iso3:8s} ({keyword}): {detail}")

    errors = sum(1 for f in futs if f.result()[3] is not None)
    print(f"\nDone. {total - errors}/{total} countries fetched successfully.")
    if errors:
        print(f"{errors} errors — re-run with --force to retry failed countries.")


if __name__ == "__main__":
    main()
