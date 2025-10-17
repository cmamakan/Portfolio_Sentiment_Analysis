#LIBRAIRY

import os
import re
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

# news_ingest.py
"""
Fetch recent financial news for a list of US tickers from two sources:
1) NewsAPI (multi-publisher aggregator)
2) GDELT 2.0 DOC API (web-scale news)

Output: a deduplicated CSV with minimal fields:
date (UTC, YYYY-MM-DD), ticker, title, description, url, source, domain, provider

Usage (example):
    export NEWSAPI_KEY="YOUR_KEY_HERE"
    python news_ingest.py --start 2023-01-01 --end 2025-09-30 --out articles.csv
"""

import os
import re
import time
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

# -----------------------------
# CONFIG
# -----------------------------

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "V", "JNJ", "GOOGL"]

COMPANY_NAMES = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon.com",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "GOOGL": "Alphabet Inc",
}

# Optional: add a few sector keywords to reduce noise per ticker (keep it short)
SECTOR_KEYWORDS = {
    "NVDA": "(chip OR gpu OR semiconductor)",
    "TSLA": "(ev OR battery OR autopilot)",
    "JPM": "(bank OR lending OR credit)",
    "V": "(payments OR card OR network)",
}

# API keys (NewsAPI)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # set with: export NEWSAPI_KEY="..."
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# GDELT DOC API
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

# -----------------------------
# HELPERS
# -----------------------------

def normalize_url(url: str) -> str:
    """Normalize URL for deduplication: lower-case host, drop query/fragment."""
    try:
        parsed = urlparse(url)
        # drop query & fragment, lower host
        new_netloc = parsed.netloc.lower()
        clean = parsed._replace(query="", fragment="", netloc=new_netloc)
        # Some sites include tracking in path; leave as-is (too aggressive otherwise)
        return urlunparse(clean)
    except Exception:
        return url or ""

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def to_iso_date(date_str: str) -> str:
    """
    Convert various incoming datetimes to UTC date (YYYY-MM-DD).
    - NewsAPI uses ISO 8601 like '2024-05-02T12:34:56Z'
    - GDELT seendate is 'YYYYMMDDHHMMSS'
    """
    if not date_str:
        return ""
    # GDELT format
    if re.match(r"^\d{14}$", date_str):
        dt = datetime.strptime(date_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    # ISO-like (NewsAPI)
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # Try fallback
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return ""

def make_query_for_ticker(ticker: str) -> str:
    """
    Build a simple query string:
    ("Company Name" OR TICKER) AND optional sector keywords
    """
    base = f'("{COMPANY_NAMES.get(ticker, ticker)}" OR {ticker})'
    extra = SECTOR_KEYWORDS.get(ticker)
    if extra:
        return f"{base} AND {extra}"
    return base

# -----------------------------
# FETCHERS
# -----------------------------

def fetch_newsapi_for_ticker(ticker: str, start: str, end: str, page_size: int = 100, max_pages: int = 5) -> pd.DataFrame:
    """
    Pull articles for one ticker using NewsAPI /v2/everything
    Fields kept: date, ticker, title, description, url, source, domain, provider='newsapi'
    """
    rows = []
    if not NEWSAPI_KEY:
        print("[NewsAPI] No NEWSAPI_KEY env var set, skipping NewsAPI.")
        return pd.DataFrame(rows)

    query = make_query_for_ticker(ticker)
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                NEWSAPI_ENDPOINT,
                params={
                    "q": query,
                    "from": start,
                    "to": end,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "page": page,
                },
                headers={"X-Api-Key": NEWSAPI_KEY},
                timeout=30,
            )
            data = resp.json()
            arts = data.get("articles", [])
            for a in arts:
                url = a.get("url")
                rows.append({
                    "date": to_iso_date(a.get("publishedAt")),
                    "ticker": ticker,
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "url": url,
                    "source": (a.get("source") or {}).get("name"),
                    "domain": extract_domain(url),
                    "provider": "newsapi",
                })
            # stop if fewer than page_size
            if len(arts) < page_size:
                break
            time.sleep(0.25)  # be gentle
        except Exception as e:
            print(f"[NewsAPI] {ticker} page {page} error: {e}")
            break

    return pd.DataFrame(rows)


def fetch_gdelt_for_ticker(ticker: str, start: str, end: str, max_records: int = 250) -> pd.DataFrame:
    """
    Pull articles for one ticker using GDELT DOC API (mode=ArtList, JSON)
    GDELT needs datetime like: YYYYMMDDHHMMSS; we pass date-only boundaries extended to full-day.
    """
    rows = []
    startdt = start.replace("-", "") + "000000"
    enddt = end.replace("-", "") + "235959"
    query = make_query_for_ticker(ticker)

    try:
        resp = requests.get(
            GDELT_DOC_ENDPOINT,
            params={
                "query": query,
                "mode": "ArtList",
                "format": "JSON",
                "maxrecords": max_records,
                "startdatetime": startdt,
                "enddatetime": enddt,
            },
            timeout=45,
        )
        data = resp.json()
        arts = data.get("articles", [])
        for a in arts:
            url = a.get("url")
            rows.append({
                "date": to_iso_date(a.get("seendate")),
                "ticker": ticker,
                "title": a.get("title"),
                "description": a.get("snippet") or a.get("socialimage") or None,  # often empty; keep minimal
                "url": url,
                "source": a.get("sourceCommonName") or a.get("domain"),
                "domain": extract_domain(url),
                "provider": "gdelt",
            })
    except Exception as e:
        print(f"[GDELT] {ticker} error: {e}")

    return pd.DataFrame(rows)

# -----------------------------
# DEDUPLICATION
# -----------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, robust dedup:
      1) Normalize URL; drop duplicates by normalized URL
      2) Then drop duplicates by (ticker, date, normalized title)
    """
    if df.empty:
        return df

    df = df.copy()
    # Normalize URL
    df["url_norm"] = df["url"].map(normalize_url)

    # Normalize title (lower & strip extra spaces)
    def norm_title(x):
        if not isinstance(x, str):
            return ""
        # collapse whitespace, lowercase
        return re.sub(r"\s+", " ", x).strip().lower()

    df["title_norm"] = df["title"].map(norm_title)

    # Step 1: unique by normalized URL
    df = df.drop_duplicates(subset=["url_norm"], keep="first")

    # Step 2: unique by (ticker, date, title_norm)
    df = df.drop_duplicates(subset=["ticker", "date", "title_norm"], keep="first")

    # Clean helper cols
    df = df.drop(columns=["url_norm", "title_norm"])
    return df

# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="articles.csv", help="Output CSV path")
    parser.add_argument("--tickers", nargs="*", default=TICKERS, help="Override tickers list")
    args = parser.parse_args()

    all_rows = []

    # Fetch per ticker
    for tkr in args.tickers:
        print(f"Fetching NewsAPI for {tkr} ...")
        df1 = fetch_newsapi_for_ticker(tkr, args.start, args.end)
        print(f"  -> {len(df1)} rows")

        print(f"Fetching GDELT for {tkr} ...")
        df2 = fetch_gdelt_for_ticker(tkr, args.start, args.end)
        print(f"  -> {len(df2)} rows")

        merged = pd.concat([df1, df2], ignore_index=True)
        # keep only rows with date + title + url + ticker
        merged = merged.dropna(subset=["date", "title", "url", "ticker"])
        all_rows.append(merged)

    if not all_rows:
        print("No data fetched.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total before dedup: {len(df)}")

    df = deduplicate(df)
    print(f"Total after dedup: {len(df)}")

    # Sort and save
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    # Final schema: date, ticker, title, description, url, source, domain, provider
    keep_cols = ["date", "ticker", "title", "description", "url", "source", "domain", "provider"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df[keep_cols]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved: {args.out} (rows: {len(df)})")

if __name__ == "__main__":
    main()
