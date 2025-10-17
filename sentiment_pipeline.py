# pip install transformers torch pandas numpy tqdm
# (optionnel fallback) pip install textblob

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- 1) Charger FinBERT (finance) ----
USE_FINBERT = True
analyzer = None
try:
    from transformers import pipeline
    analyzer = pipeline(
        task="sentiment-analysis",
        model="ProsusAI/finbert",      # modèle finance
        tokenizer="ProsusAI/finbert",
        return_all_scores=True,         # pour récupérer P(pos), P(neu), P(neg)
        truncation=True
    )
except Exception as e:
    USE_FINBERT = False
    print("[WARN] FinBERT indisponible => fallback TextBlob (moins précis).", e)
    from textblob import TextBlob

def finbert_signed_score(text: str) -> float:
    """
    Renvoie un score signé ~ [-1, +1] à partir des proba FinBERT.
    Heuristique simple : s = P(pos) - P(neg).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    outs = analyzer(text[:512])  # tronque pour rester compatible
    # outs: [[{'label': 'positive', 'score': 0.7}, {'label': 'neutral', ...}, {'label': 'negative', ...}]]
    triplet = {d['label'].lower(): d['score'] for d in outs[0]}
    p_pos = triplet.get('positive', 0.0)
    p_neg = triplet.get('negative', 0.0)
    return float(p_pos - p_neg)

def textblob_signed_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)  # [-1, 1]

def compute_sentiment_series(df: pd.DataFrame, text_cols=("title","content")) -> pd.Series:
    """
    Calcule un score de sentiment (article-level) pour chaque ligne du DF.
    Combine title + content si dispo. Retourne une Series float index-alignée.
    """
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring articles"):
        txt_parts = []
        for c in text_cols:
            if c in df.columns and isinstance(row[c], str):
                txt_parts.append(row[c])
        text = " / ".join(txt_parts) if txt_parts else ""
        if USE_FINBERT:
            s = finbert_signed_score(text)
        else:
            s = textblob_signed_score(text)
        scores.append(s)
    return pd.Series(scores, index=df.index, name="sentiment")

# ---- 2) Utilitaires nettoyage & agrégation ----
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # garde colonnes clés si existent
    keep = [c for c in ["date","ticker","title","content","source","url","domain"] if c in df.columns]
    df = df[keep].copy()
    # drop doublons naïfs sur title+date+ticker s'ils existent
    subset_cols = [c for c in ["date","ticker","title"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    # drop textes vides
    if "content" in df.columns:
        df = df.dropna(subset=["content"])
    # normaliser date -> date (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # ticker uppercase si présent
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège au niveau (date, ticker) en moyenne de sentiment.
    """
    out = (
        df.groupby(["date","ticker"], as_index=False)["sentiment"]
          .mean()
          .rename(columns={"sentiment":"sentiment_mean"})
    )
    return out

# ---- 3) Pipeline principal train/test ----
def run_pipeline(
    train_path="articles_zhistorical_010123_310825.csv",
    test_path="articles_zfull.csv",
    prices_long_path="prices_long.csv",
    out_train="daily_sentiment_train.csv",
    out_test="daily_sentiment_test.csv",
    out_join_train="train_sentiment_prices.csv",
    out_join_test="test_sentiment_prices.csv"
):
    # a) Charger articles
    print(">> Loading articles (train/test)")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # b) Nettoyage minimal
    train = basic_clean(train)
    test  = basic_clean(test)

    # c) Scoring article-level
    print(">> Scoring TRAIN articles…")
    train["sentiment"] = compute_sentiment_series(train)
    print(">> Scoring TEST articles…")
    test["sentiment"]  = compute_sentiment_series(test)

    # d) Agrégation daily (date,ticker)
    daily_train = aggregate_daily(train)
    daily_test  = aggregate_daily(test)

    # e) Sauvegarde daily
    daily_train.to_csv(out_train, index=False)
    daily_test.to_csv(out_test, index=False)
    print(f"[OK] Saved {out_train} ({len(daily_train)})")
    print(f"[OK] Saved {out_test} ({len(daily_test)})")

    # f) Joindre aux PRIX (prices_long.csv)
    #    On merge sur (date, ticker) pour préparer corrélations & next-day returns ensuite
    print(">> Joining with prices_long.csv")
    prices = pd.read_csv(prices_long_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    train_join = prices.merge(daily_train, on=["date","ticker"], how="left")
    test_join  = prices.merge(daily_test,  on=["date","ticker"], how="left")

    train_join.to_csv(out_join_train, index=False)
    test_join.to_csv(out_join_test, index=False)
    print(f"[OK] Saved {out_join_train} ({len(train_join)})")
    print(f"[OK] Saved {out_join_test} ({len(test_join)})")

if __name__ == "__main__":
    run_pipeline()
