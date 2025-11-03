"""
Merges and cleans the two original article datasets (historical + recent)
into one consistent CSV file.
Keeps only [date, ticker, title, domain].
Removes rows with invalid dates, duplicates, empty rows,
removes titles containing any non-ASCII char (accents, ñ, …),
removes titles that contain the standalone word 'y' (Spanish conjunction),
and any title containing fewer than N ASCII letters (A–Z).

Inputs:
- articles_API_GDELT.csv
- articles_API_NewsAPI.csv

Output:
- articles_merge_of_APIs.csv
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

# Regex utilitaires
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")      # lettres ASCII A-Z uniquement
NON_ASCII_RE    = re.compile(r"[^\x00-\x7F]")  # tout caractère non ASCII
SPANISH_Y_RE    = re.compile(r"\b[yY]\b")      # mot isolé 'y' (espagnol)

def count_ascii_letters(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(ASCII_LETTER_RE.findall(text))

def is_ascii_only(text: str) -> bool:
    """True si le texte ne contient QUE des caractères ASCII (0x00–0x7F)."""
    if not isinstance(text, str):
        return False
    return NON_ASCII_RE.search(text) is None

def merge_articles(
    hist_path="articles_API_GDELT.csv",
    recent_path="articles_API_NewsAPI.csv",
    out_path="articles_merge_of_APIs.csv",
    min_ascii_letters=20,
    drop_if_contains_standalone_y=True
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

    # 3) Strip whitespace + drop empty titles
    if "title" in df.columns:
        df["title"] = df["title"].astype(str).str.strip()
        before = len(df)
        df = df[df["title"].str.len() > 0]
        print(f"[CLEAN] Dropped empty titles: {before - len(df)}")
    else:
        raise ValueError("Expected 'title' column not found.")

    # 4) Validate / normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df.dropna(subset=["date"], inplace=True)
        print(f"[CLEAN] Dropped invalid dates: {before - len(df)}")
        df["date"] = df["date"].dt.date

    # 5) Remove non-ASCII titles (accents, ñ, etc.)
    before = len(df)
    df = df[df["title"].apply(is_ascii_only)].copy()
    print(f"[CLEAN] Dropped titles containing non-ASCII chars: {before - len(df)}")

    # 6) Optionnel : enlever les titres contenant le mot isolé 'y' (espagnol)
    if drop_if_contains_standalone_y:
        before = len(df)
        df = df[~df["title"].str.contains(SPANISH_Y_RE, na=False)]
        print(f"[CLEAN] Dropped titles containing standalone 'y': {before - len(df)}")

    # 7) Remove titles with < N ASCII letters (A–Z)
    df["ascii_letters"] = df["title"].apply(count_ascii_letters)
    before = len(df)
    df = df[df["ascii_letters"] >= min_ascii_letters].copy()
    print(f"[CLEAN] Dropped titles with <{min_ascii_letters} ASCII letters: {before - len(df)}")
    df.drop(columns=["ascii_letters"], inplace=True)

    # 8) Normalize ticker (uppercase)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # 9) Drop duplicates (same date + title)
    subset = [c for c in ["date", "title"] if c in df.columns]
    if subset:
        before = len(df)
        df.drop_duplicates(subset=subset, inplace=True)
        print(f"[CLEAN] Dropped duplicates on {subset}: {before - len(df)}")

    # 10) Save final clean file
    print(f"[OK] Final rows: {len(df)}")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved → {out_path}")

if __name__ == "__main__":
    merge_articles()
