"""Pre-fetch GDELT media attention data for all crisis countries.

Run from the repo root before the demo:
    uv run scripts/prefetch_media.py

Demo prep — to make the refresh button visible during a live demo, pass an
--end-date that is ~30 days in the past. The app will then detect stale data
and show the "New data available" refresh button, which fetches the missing
30 days live in front of the audience:
    uv run scripts/prefetch_media.py --end-date 2026-03-20

Skips countries whose CSV is already up to date (unless --force is passed).
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from media import MEDIA_DIR, _fetch_gdelt, _load_csv, _save_csv

DATA_DIR = Path(__file__).parent.parent / "data"
_INTER_CALL_SLEEP = 15


def _country_list() -> list[tuple[str, str]]:
    """Return [(iso3, country_name), ...] from the main funding CSV."""
    df = pd.read_csv(DATA_DIR / "country_year_severity_funding.csv")
    sev = pd.read_csv(DATA_DIR / "inform_severity_cleaned.csv")
    sev.columns = sev.columns.str.strip()
    name_map = (
        sev[["ISO3", "COUNTRY"]].dropna().drop_duplicates("ISO3")
        .set_index("ISO3")["COUNTRY"].to_dict()
    )
    iso3s = df["Country_ISO3"].dropna().unique()
    return [(iso3, name_map.get(iso3, iso3)) for iso3 in sorted(iso3s)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end-date", default=date.today().isoformat(),
                        help="End date for fetch window (YYYY-MM-DD). Default: today.")
    parser.add_argument("--months", type=int, default=12,
                        help="Months of history to fetch. Default: 12.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if CSV already exists.")
    args = parser.parse_args()

    end = args.end_date
    start = (date.fromisoformat(end) - timedelta(days=args.months * 31)).isoformat()
    countries = _country_list()

    print(f"Pre-fetching {len(countries)} countries  [{start} → {end}]")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    for i, (iso3, name) in enumerate(countries, 1):
        existing = _load_csv(iso3)
        if not args.force and existing is not None and not existing.empty:
            max_date = existing["date"].max().date().isoformat()
            if max_date >= end:
                print(f"  [{i}/{len(countries)}] {iso3} ({name}) — skip (up to {max_date})")
                continue

        print(f"  [{i}/{len(countries)}] {iso3} ({name}) … ", end="", flush=True)
        df = _fetch_gdelt(name, start, end)
        if df is None or df.empty:
            print("no data")
        else:
            _save_csv(iso3, df)
            print(f"{len(df)} rows")
        time.sleep(_INTER_CALL_SLEEP)

    print("Done.")


if __name__ == "__main__":
    main()
