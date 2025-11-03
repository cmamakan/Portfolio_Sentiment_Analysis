"""
Performs sentiment analysis on financial news headlines using FinBERT,
with an automatic fallback to TextBlob if FinBERT is unavailable.
It cleans and scores all articles (train and test), then aggregates
daily sentiment by date and ticker.
Finally, it merges the sentiment data with market prices to prepare
datasets for further modeling and analysis.

Purpose:
To generate clean sentiment signals and merge them with price data.

Input:
- articles_merge_of_APIs.csv
- prices_long.csv                          → used only for the final merge step

Outputs:
- sentiment_daily.csv   → daily average sentiment 
- sentiment_prices.csv        → merged prices + sentiment
"""

# pip install transformers torch pandas numpy tqdm
# (optionnel fallback) pip install textblob

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ---- 1) Charger FinBERT (finance) ----
USE_FINBERT = True
analyzer = None
try:
    from transformers import pipeline
    analyzer = pipeline(
        task="sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        return_all_scores=True,
        truncation=True,
    )
except Exception as e:
    USE_FINBERT = False
    print("[WARN] FinBERT indisponible => fallback TextBlob (moins précis).", e)
    try:
        from textblob import TextBlob
    except Exception as ee:
        raise RuntimeError("Ni FinBERT ni TextBlob disponibles. Installe au moins l'un des deux.") from ee


def finbert_signed_score(text: str) -> float:
    """
    Score signé ~ [-1, +1] via FinBERT : s = P(pos) - P(neg)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    outs = analyzer(text[:512])  # tronquage simple
    triplet = {d['label'].lower(): d['score'] for d in outs[0]}
    p_pos = triplet.get('positive', 0.0)
    p_neg = triplet.get('negative', 0.0)
    return float(p_pos - p_neg)


def textblob_signed_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)  # [-1, 1]


def compute_sentiment_series(df: pd.DataFrame, text_cols=("title", "content")) -> pd.Series:
    """
    Calcule un score de sentiment article-level pour chaque ligne.
    Concatène les colonnes textuelles disponibles (title/content…).
    """
    use_cols = [c for c in text_cols if c in df.columns]
    scores = []
    it = tqdm(df.itertuples(index=False), total=len(df), desc="Scoring articles")
    for row in it:
        row_dict = row._asdict() if hasattr(row, "_asdict") else None
        # fallback si namedtuple non standard
        if row_dict is None:
            row_dict = {c: v for c, v in zip(df.columns, row)}

        txt_parts = []
        for c in use_cols:
            val = row_dict.get(c, None)
            if isinstance(val, str) and val.strip():
                txt_parts.append(val)
        text = " / ".join(txt_parts) if txt_parts else ""

        if USE_FINBERT:
            s = finbert_signed_score(text)
        else:
            s = textblob_signed_score(text)
        scores.append(s)
    return pd.Series(scores, index=df.index, name="sentiment")


# ---- 2) Utilitaires nettoyage & agrégation ----
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Garde colonnes clés si présentes
    - Drop duplicates simples (date,ticker,title)
    - Drop content vide si colonne présente
    - Normalise date -> date (YYYY-MM-DD)
    - Uppercase ticker
    """
    keep = [c for c in ["date", "ticker", "title", "content", "source", "url", "domain"] if c in df.columns]
    df = df[keep].copy()

    subset_cols = [c for c in ["date", "ticker", "title"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    if "content" in df.columns:
        # on garde les lignes utilisant au moins title ou content ; si content existe, on évite les NaN massifs
        df = df[~(df["content"].isna() & (~df.get("title", pd.Series([])).astype(str).str.strip().astype(bool)))]

    # normaliser date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège (date, ticker) par moyenne du score 'sentiment'.
    """
    out = (
        df.groupby(["date", "ticker"], as_index=False)["sentiment"]
          .mean()
          .rename(columns={"sentiment": "sentiment_mean"})
    )
    return out


# ---- 3) Pipeline principal (un seul fichier articles mergé) ----
def run_pipeline_one(
    articles_path="articles_merge_of_APIs.csv",
    prices_long_path="prices_long.csv",
    out_daily="sentiment_daily.csv",
    out_join="sentiment_prices.csv",
    text_cols=("title", "content"),
):
    # 1) Load & clean
    print(">> Loading articles")
    df = pd.read_csv(articles_path)
    df = basic_clean(df)

    # 2) Sentiment per article
    print(">> Scoring articles…")
    df["sentiment"] = compute_sentiment_series(df, text_cols=text_cols)

    # 3) Aggregate daily (date,ticker)
    daily = aggregate_daily(df)
    daily.to_csv(out_daily, index=False)
    print(f"[OK] Saved {out_daily} ({len(daily)})")

    # 4) Join with prices
    print(">> Joining with prices_long.csv")
    prices = pd.read_csv(prices_long_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    joined = prices.merge(daily, on=["date", "ticker"], how="left")
    joined.to_csv(out_join, index=False)
    print(f"[OK] Saved {out_join} ({len(joined)})")


def parse_args():
    p = argparse.ArgumentParser(description="03_news_merge_and_cleaning: sentiment + merge with prices")
    p.add_argument("--articles_path", default="articles_zfull.csv")
    p.add_argument("--prices_long_path", default="prices_long.csv")
    p.add_argument("--out_daily", default="daily_sentiment.csv")
    p.add_argument("--out_join", default="sentiment_prices.csv")
    p.add_argument(
        "--text_cols",
        nargs="*",
        default=["title", "content"],
        help="Colonnes textuelles à utiliser (ex: --text_cols title content) ; pour plus rapide: --text_cols title",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_one(
        articles_path=args.articles_path,
        prices_long_path=args.prices_long_path,
        out_daily=args.out_daily,
        out_join=args.out_join,
        text_cols=tuple(args.text_cols),
    )