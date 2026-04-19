"""GDELT media attention fetching with hybrid CSV cache."""
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from gdeltdoc import Filters, GdeltDoc
from gdeltdoc.errors import RateLimitError

MEDIA_DIR = Path(__file__).parent / "data" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

_STALE_DAYS = 1
_RETRY_SLEEP = 120
_INTER_CALL_SLEEP = 15


def _load_csv(iso3: str) -> pd.DataFrame | None:
    path = MEDIA_DIR / f"{iso3}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def _save_csv(iso3: str, df: pd.DataFrame) -> None:
    df.to_csv(MEDIA_DIR / f"{iso3}.csv", index=False)


def _process_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise GDELT timelinevolraw response to date / articles / frac."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
    art_col = next(c for c in df.columns if "article count" in c.lower())
    all_col = next(c for c in df.columns if "all articles" in c.lower())
    df = df.rename(columns={date_col: "date", art_col: "articles", all_col: "total"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["frac"] = df["articles"] / df["total"].replace(0, np.nan)
    return df[["date", "articles", "frac"]].sort_values("date").reset_index(drop=True)


def _fetch_gdelt(keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Call GDELT API with one retry on rate-limit. Raises on failure."""
    gd = GdeltDoc()
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            f = Filters(keyword=keyword, language="English",
                        start_date=start_date, end_date=end_date)
            raw = gd.timeline_search("timelinevolraw", f)
            if raw is not None and not raw.empty:
                return _process_raw(raw)
            raise RuntimeError(f"GDELT returned empty result for keyword={keyword!r} ({start_date} – {end_date})")
        except RateLimitError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(_RETRY_SLEEP)
            continue
        except Exception as exc:
            last_exc = exc
            break
    raise RuntimeError(f"GDELT fetch failed for {keyword!r}") from last_exc


def is_stale(iso3: str) -> bool:
    """True if the cached CSV is missing or older than _STALE_DAYS."""
    df = _load_csv(iso3)
    if df is None or df.empty:
        return True
    cutoff = pd.Timestamp(date.today() - timedelta(days=_STALE_DAYS), tz="UTC")
    return df["date"].max() < cutoff


def _gap_fill_and_save(iso3: str, country_name: str) -> None:
    """Fetch missing days from GDELT and append to the CSV."""
    existing = _load_csv(iso3)
    if existing is None or existing.empty:
        start = (date.today() - timedelta(days=365)).isoformat()
    else:
        start = (existing["date"].max().date() + timedelta(days=1)).isoformat()
    end = date.today().isoformat()
    if start >= end:
        return
    new_data = _fetch_gdelt(country_name, start, end)
    if new_data is None or new_data.empty:
        return
    combined = pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
    combined = combined.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    _save_csv(iso3, combined)


@st.cache_data(ttl=3600)
def get_annual_media_map(year: int) -> pd.DataFrame:
    """Return a DataFrame with Country_ISO3 and media_frac_pct (annual mean daily frac × 100).

    Only includes countries that have cached CSVs with data for *year*.
    """
    rows = []
    for csv_path in MEDIA_DIR.glob("*.csv"):
        iso3 = csv_path.stem
        df = _load_csv(iso3)
        if df is None or df.empty:
            continue
        year_df = df[df["date"].dt.year == year]
        if year_df.empty:
            continue
        rows.append({
            "Country_ISO3": iso3,
            "media_frac_pct": year_df["frac"].mean() * 100,
        })
    if not rows:
        return pd.DataFrame(columns=["Country_ISO3", "media_frac_pct"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_media_attention(iso3: str, country_name: str, year: int = 2026) -> pd.DataFrame:
    """Load cached media attention data for a country, sliced to *year*.

    For the current year: last 12 months of data.
    For past years: Jan 1 – Dec 31 of that year.
    Does NOT auto-fetch stale data — call _gap_fill_and_save() first if needed.
    Raises RuntimeError if no data is available.
    """
    current_year = date.today().year
    df = _load_csv(iso3)
    if df is None or df.empty:
        if year == current_year:
            start = (date.today() - timedelta(days=365)).isoformat()
            end   = date.today().isoformat()
        else:
            start = f"{year}-01-01"
            end   = f"{year}-12-31"
        df = _fetch_gdelt(country_name, start, end)
        _save_csv(iso3, df)

    if year == current_year:
        cutoff = pd.Timestamp(date.today() - timedelta(days=365), tz="UTC")
        df = df[df["date"] >= cutoff].copy()
    else:
        df = df[
            (df["date"] >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
            (df["date"] <= pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"))
        ].copy()

    if df.empty:
        raise RuntimeError(
            f"No GDELT data for {country_name} ({iso3}) in {year}. "
            "Try clicking refresh or switching to a different year."
        )
    df["rolling_7d"] = df["frac"].rolling(7, min_periods=1).mean() * 100
    df["frac_pct"]   = df["frac"] * 100
    return df.reset_index(drop=True)
