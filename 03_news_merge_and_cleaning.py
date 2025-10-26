"""
Merges and cleans the two original article datasets (historical + recent)
into one consistent CSV file.
Keeps only [date, ticker, title, domain].
Removes rows with invalid dates, duplicates, empty rows,
and any title containing fewer than 20 ASCII letters (A–Z).

This filter excludes all titles containing accented or non-English characters
(e.g., ñ, é, è, à, ç, ù, …) to ensure compatibility with English-only models
like FinBERT or TextBlob.

Inputs:
- articles_API_GDELT.csv
- articles_API_NewsAPI.csv

Output:
- articles_merge_of_APIs
"""

import re
import pandas as pd

def safe_read_csv(path: str) -> pd.DataFrame:
    """Robust CSV reader that skips bad lines and ignores encoding errors."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip", encoding_errors="ignore")
        print(f"[OK] Loaded {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()

# Only plain ASCII letters A–Z (strict English)
LATIN_RE = re.compile(r"[A-Za-z]")

def count_ascii_letters(text: str) -> int:
    """Count number of ASCII letters in a string."""
    if not isinstance(text, str):
        return 0
    return len(LATIN_RE.findall(text))

def merge_articles(
    hist_path="articles_API_GDELT.csv",
    recent_path="articles_API_NewsAPI.csv",
    out_path="articles_API_NewsAPI.csv",
    min_ascii_letters=20
):
    print(">> Loading sources…")
    df_hist = safe_read_csv(hist_path)
    df_recent = safe_read_csv(recent_path)

    df = pd.concat([df_hist, df_recent], ignore_index=True)

    # 1) Drop fully empty rows
    before = len(df)
    df.dropna(how="all", inplace=True)
    print(f"[CLEAN] Dropped fully empty rows: {before - len(df)}")

    # 2) Keep only necessary columns
    keep_cols = [c for c in ["date", "ticker", "title", "domain"] if c in df.columns]
    if not keep_cols:
        raise ValueError("None of the expected columns ['date','ticker','title','domain'] are present.")
    df = df[keep_cols].copy()

    # 3) Validate / normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df.dropna(subset=["date"], inplace=True)
        print(f"[CLEAN] Dropped invalid dates: {before - len(df)}")
        df["date"] = df["date"].dt.date

    # 4) Remove titles with <20 ASCII letters (A–Z only)
    if "title" in df.columns:
        df["ascii_letters"] = df["title"].apply(count_ascii_letters)
        before = len(df)
        df = df[df["ascii_letters"] >= min_ascii_letters].copy()
        print(f"[CLEAN] Dropped titles with <{min_ascii_letters} English letters: {before - len(df)}")
        df.drop(columns=["ascii_letters"], inplace=True)
    else:
        raise ValueError("Expected 'title' column not found.")

    # 5) Normalize ticker (uppercase)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # 6) Drop duplicates (same date + title)
    subset = [c for c in ["date", "title"] if c in df.columns]
    if subset:
        before = len(df)
        df.drop_duplicates(subset=subset, inplace=True)
        print(f"[CLEAN] Dropped duplicates on {subset}: {before - len(df)}")

    # 7) Save final clean file
    print(f"[OK] Final rows: {len(df)}")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved → {out_path}")

if __name__ == "__main__":
    merge_articles()
