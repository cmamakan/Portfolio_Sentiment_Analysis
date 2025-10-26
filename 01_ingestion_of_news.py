"""
This script collects financial news headlines from two ple APIs : NewsAPI and GDELT
and prepares them for sentiment analysis. 

It fetches articles related to selected tickers, cleans and normalizes the data 
(dates, sources, tickers, duplicates), and saves a unified CSV file.

Purpose:
To build a clean dataset of short, impactful headlines that will later be scored 
with FinBERT for sentiment analysis.

Output:
articles_API_GDELT and articles_API_NewsAPI (after ajustments)
  → contains columns like [date, ticker, title]
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

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "JPM", "V", "JNJ", "GOOGL"]

COMPANY_NAMES = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "GOOGL": "Alphabet Inc",
}

SECTOR_KEYWORDS = {
    "NVDA": "(chip OR gpu OR semiconductor)",
    "TSLA": "(ev OR battery OR autopilot)",
    "JPM": "(bank OR lending OR credit)",
    "V": "(payments OR card OR network)",
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse(p._replace(query="", fragment="", netloc=p.netloc.lower()))
    except Exception:
        return url or ""

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def to_iso_date(s: str) -> str:
    if not s:
        return ""
    if re.match(r"^\d{14}$", s):  # GDELT
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).strftime("%Y-%m-%d")
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return ""

# -------- Queries ----------
def make_query_for_ticker_newsapi(ticker: str) -> str:
    base_name = f'"{COMPANY_NAMES.get(ticker, ticker)}"'
    ticker_token = f" OR {ticker}" if len(ticker) >= 1 else ""
    base = f"({base_name}{ticker_token})"
    extra = SECTOR_KEYWORDS.get(ticker)
    return f"{base} AND {extra}" if extra else base

def _needs_quotes(company: str) -> bool:
    # Quote only multi-word company names; single-word names (e.g., Visa) left unquoted for GDELT.
    return " " in company.strip()

def make_query_for_ticker_gdelt(ticker: str) -> str:
    """
    GDELT-safe:
      - Use company name (quoted only if multi-word).
      - Add ticker only if len>=3.
      - Wrap OR groups in parentheses.
      - If there is an OR in the base, wrap ( ... ) before AND.
    """
    company = COMPANY_NAMES.get(ticker, ticker).strip()
    tokens = []
    # company token
    tokens.append(f'"{company}"' if _needs_quotes(company) else company)
    # ticker token (avoid 'V', 'GE', etc.)
    if len(ticker) >= 3:
        tokens.append(ticker)

    if len(tokens) == 1:
        base = tokens[0]
    else:
        base = "(" + " OR ".join(tokens) + ")"  # OR group must be in ()

    extra = SECTOR_KEYWORDS.get(ticker)
    if extra:
        ext = extra.strip()
        if " OR " in ext and not ext.startswith("("):
            ext = f"({ext})"
        return f"{base} AND {ext}"
    return base

# -------- Fetchers ----------
def fetch_newsapi_for_ticker(ticker: str, start: str, end: str, page_size: int = 100, max_pages: int = 3) -> pd.DataFrame:
    rows = []
    if not NEWSAPI_KEY:
        print("[NewsAPI] No NEWSAPI_KEY env var set, skipping NewsAPI.")
        return pd.DataFrame(rows)
    q = make_query_for_ticker_newsapi(ticker)
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                NEWSAPI_ENDPOINT,
                params={
                    "q": q, "from": start, "to": end, "language": "en",
                    "sortBy": "publishedAt", "pageSize": page_size, "page": page
                },
                headers={"X-Api-Key": NEWSAPI_KEY},
                timeout=30,
            )
            if r.status_code != 200:
                print(f"[NewsAPI] {ticker} HTTP {r.status_code}: {r.text[:200]}")
                break
            data = r.json()
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
            if len(arts) < page_size:
                break
            time.sleep(0.25)
        except Exception as e:
            print(f"[NewsAPI] {ticker} page {page} error: {e}")
            break
    return pd.DataFrame(rows)

def fetch_gdelt_for_ticker(ticker: str, start: str, end: str, max_records: int = 250) -> pd.DataFrame:
    rows = []
    startdt = start.replace("-", "") + "000000"
    enddt = end.replace("-", "") + "235959"
    q = make_query_for_ticker_gdelt(ticker)
    try:
        r = requests.get(
            GDELT_DOC_ENDPOINT,
            params={
                "query": q, "mode": "ArtList", "format": "JSON",
                "maxrecords": max_records, "startdatetime": startdt, "enddatetime": enddt
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=45,
        )
        if r.status_code != 200:
            print(f"[GDELT] {ticker} HTTP {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=["date","ticker","title","description","url","source","domain","provider"])
        try:
            content = r.text.lstrip("\ufeff").strip()
            data = json.loads(content)
        except Exception:
            print(f"[GDELT] {ticker} Non-JSON response (first 200 chars): {r.text[:200]}")
            return pd.DataFrame(columns=["date","ticker","title","description","url","source","domain","provider"])
        arts = data.get("articles", [])
        for a in arts:
            url = a.get("url")
            rows.append({
                "date": to_iso_date(a.get("seendate")),
                "ticker": ticker,
                "title": a.get("title"),
                "description": a.get("snippet") or a.get("socialimage") or None,
                "url": url,
                "source": a.get("sourceCommonName") or a.get("domain"),
                "domain": extract_domain(url),
                "provider": "gdelt",
            })
    except Exception as e:
        print(f"[GDELT] {ticker} error: {e}")
    return pd.DataFrame(rows)

# -------- Dedup ----------
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["url_norm"] = df["url"].map(normalize_url)
    def norm_title(x):
        if not isinstance(x, str):
            return ""
        return re.sub(r"\s+", " ", x).strip().lower()
    df["title_norm"] = df["title"].map(norm_title)
    df = df.drop_duplicates(subset=["url_norm"], keep="first")
    df = df.drop_duplicates(subset=["ticker", "date", "title_norm"], keep="first")
    return df.drop(columns=["url_norm", "title_norm"])

# -------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="articles.csv")
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    all_rows = []
    for tkr in args.tickers:
        print(f"Fetching NewsAPI for {tkr} ...")
        df1 = fetch_newsapi_for_ticker(tkr, args.start, args.end)
        print(f"  -> {len(df1)} rows")

        print(f"Fetching GDELT for {tkr} ...")
        df2 = fetch_gdelt_for_ticker(tkr, args.start, args.end)
        print(f"  -> {len(df2)} rows")

        merged = pd.concat([df1, df2], ignore_index=True)
        for col in ["date","title","url","ticker","description","source","domain","provider"]:
            if col not in merged.columns:
                merged[col] = None
        merged = merged.dropna(subset=["date","title","url","ticker"])
        all_rows.append(merged)

    if not all_rows:
        print("No data fetched.")
        return

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total before dedup: {len(df)}")
    df = deduplicate(df)
    print(f"Total after dedup: {len(df)}")
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)
    keep = ["date","ticker","title","description","url","source","domain","provider"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = df[keep]
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved: {args.out} (rows: {len(df)})")

if __name__ == "__main__":
    main()
