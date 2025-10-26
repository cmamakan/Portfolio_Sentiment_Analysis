#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Sentiment (PySpark + FinBERT)
- Lecture articles train/test (CSV)
- Nettoyage minimal & normalisation (date, ticker)
- Scoring sentiment (title+content) via FinBERT (transformers) avec pandas UDF (vectorisé)
- Agrégation quotidienne (date, ticker) = moyenne
- Join avec prices_long.csv
- Sauvegarde CSV (partitions fusionnées)

Exécution :
spark-submit spark_finbert_pipeline.py \
  --train_path articles_zhistorical_010123_310825.csv \
  --test_path  articles_zfull.csv \
  --prices_long_path prices_long.csv \
  --out_train daily_sentiment_train.csv \
  --out_test  daily_sentiment_test.csv \
  --out_join_train train_sentiment_prices.csv \
  --out_join_test  test_sentiment_prices.csv

Dépendances (à installer dans TON environnement) :
pip install pyspark==3.5.1 pyarrow==14.0.1 pandas>=2.0 numpy tqdm \
            torch>=2.2 transformers>=4.44 tokenizers>=0.15 textblob>=0.18
# (optionnel) python -m textblob.download_corpora
# (si GPU CUDA: installe torch depuis l’index CUDA adapté)
"""

import os
import sys
import argparse

from pyspark.sql import SparkSession, functions as F, types as T


# ---------- 1) Config Spark ----------
def build_spark(app_name="SparkFinBERT"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # pandas UDF
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------- 2) Chargement FinBERT côté worker (lazy global) ----------
_analyzer = None
_USE_FINBERT = True

def _load_analyzer():
    """
    Chargé une seule fois par worker, au premier appel du UDF.
    Si FinBERT indispo, fallback TextBlob.
    """
    global _analyzer, _USE_FINBERT
    if _analyzer is not None:
        return _analyzer, _USE_FINBERT
    try:
        from transformers import pipeline
        _analyzer = pipeline(
            task="sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
        )
        _USE_FINBERT = True
    except Exception as e:
        _USE_FINBERT = False
        print(f"[WARN][worker] FinBERT indisponible, fallback TextBlob: {e}")
        from textblob import TextBlob
        _analyzer = TextBlob
    return _analyzer, _USE_FINBERT


# ---------- 3) pandas UDF de scoring ----------
@F.pandas_udf(T.DoubleType())   # <<< IMPORTANT: évite l’erreur de session
def finbert_signed_score_udf(text_series):
    """
    Reçoit une pandas.Series de textes, renvoie une pandas.Series de floats [-1, 1].
    Heuristique FinBERT : P(positive) - P(negative).
    """
    import pandas as pd
    analyzer, use_finbert = _load_analyzer()
    out = []
    if use_finbert:
        batch = [ (t if isinstance(t, str) and t.strip() else "")[:512] for t in text_series.tolist() ]
        results = analyzer(batch)
        for outs in results:
            # outs: [{'label': 'positive', 'score': x}, {'label': 'neutral', ...}, {'label': 'negative', ...}]
            if not outs or not isinstance(outs, list):
                out.append(0.0)
                continue
            triplet = {d.get("label", "").lower(): float(d.get("score", 0.0)) for d in outs}
            p_pos = triplet.get("positive", 0.0)
            p_neg = triplet.get("negative", 0.0)
            out.append(float(p_pos - p_neg))
    else:
        # Fallback TextBlob
        for t in text_series.tolist():
            if not isinstance(t, str) or not t.strip():
                out.append(0.0)
            else:
                s = analyzer(t).sentiment.polarity  # [-1, 1]
                out.append(float(s))
    return pd.Series(out, dtype="float64")


# ---------- 4) Utilitaires DataFrame ----------
def basic_clean_spark(df):
    # garde colonnes clés si existent
    keep_cols = [c for c in ["date","ticker","title","content","source","url","domain"] if c in df.columns]
    df = df.select(*keep_cols)

    # drop textes vides si "content" existe
    if "content" in df.columns:
        df = df.filter(F.col("content").isNotNull() & (F.length(F.col("content")) > 0))

    # normaliser date (YYYY-MM-DD) -> to_date
    df = df.withColumn("date", F.to_date(F.col("date").cast("string")))

    # ticker uppercase si présent
    if "ticker" in df.columns:
        df = df.withColumn("ticker", F.upper(F.col("ticker").cast("string")))

    # suppression de doublons naïfs (date, ticker, title) si dispo
    subset = [c for c in ["date","ticker","title"] if c in df.columns]
    if subset:
        df = df.dropDuplicates(subset)

    return df


def aggregate_daily_spark(df_scored):
    # Agrège moyenne sur (date, ticker)
    return (
        df_scored.groupBy("date", "ticker")
        .agg(F.avg(F.col("sentiment")).alias("sentiment_mean"))
        .orderBy("date", "ticker")
    )


def write_single_csv(df, path):
    """
    Écrit en un dossier CSV (Spark) avec un part-*.csv.
    Renomme manuellement après si tu veux un seul fichier plat.
    """
    (df.coalesce(1)
       .write
       .option("header", True)
       .mode("overwrite")
       .csv(path))
    print(f"[OK] Saved folder: {path} (Spark CSV). "
          "Note: Spark écrit un dossier avec part-*.csv ; renomme si besoin.")


# ---------- 5) Pipeline principal ----------
def run_pipeline(
    spark,
    train_path,
    test_path,
    prices_long_path,
    out_train,
    out_test,
    out_join_train,
    out_join_test
):
    # a) Charger articles
    print(">> Loading articles (train/test)")
    train = (spark.read.option("header", True).option("multiLine", True).csv(train_path))
    test  = (spark.read.option("header", True).option("multiLine", True).csv(test_path))

    # b) Nettoyage minimal
    train = basic_clean_spark(train)
    test  = basic_clean_spark(test)

    # c) Combiner title+content (concat_ws ignore les NULL)
    def with_combined_text(df):
        cols = [c for c in ["title","content"] if c in df.columns]
        if not cols:
            return df.withColumn("combined_text", F.lit(""))
        return df.withColumn("combined_text", F.concat_ws(" / ", *[F.col(c) for c in cols]))
    train = with_combined_text(train)
    test  = with_combined_text(test)

    # d) Scoring article-level
    print(">> Scoring TRAIN articles…")
    train_scored = train.withColumn("sentiment", finbert_signed_score_udf(F.col("combined_text")))
    print(">> Scoring TEST articles…")
    test_scored  = test.withColumn("sentiment",  finbert_signed_score_udf(F.col("combined_text")))

    # e) Agrégation daily
    daily_train = aggregate_daily_spark(train_scored)
    daily_test  = aggregate_daily_spark(test_scored)

    # f) Sauvegarde daily
    write_single_csv(daily_train, out_train)
    write_single_csv(daily_test,  out_test)

    # g) Join avec PRIX
    print(">> Joining with prices_long.csv")
    prices = (spark.read.option("header", True).csv(prices_long_path))
    prices = (prices
              .withColumn("date", F.to_date(F.col("date").cast("string")))
              .withColumn("ticker", F.upper(F.col("ticker").cast("string"))))

    train_join = (prices.join(daily_train, on=["date","ticker"], how="left"))
    test_join  = (prices.join(daily_test,  on=["date","ticker"], how="left"))

    write_single_csv(train_join, out_join_train)
    write_single_csv(test_join,  out_join_test)

    print("[DONE] Pipeline terminé.")


# ---------- 6) Entrée CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="PySpark + FinBERT sentiment pipeline")
    p.add_argument("--train_path", default="articles_zhistorical_010123_310825.csv")
    p.add_argument("--test_path",  default="articles_zfull.csv")
    p.add_argument("--prices_long_path", default="prices_long.csv")
    p.add_argument("--out_train", default="daily_sentiment_train.csv")
    p.add_argument("--out_test",  default="daily_sentiment_test.csv")
    p.add_argument("--out_join_train", default="train_sentiment_prices.csv")
    p.add_argument("--out_join_test",  default="test_sentiment_prices.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spark = build_spark("SparkFinBERT")
    try:
        run_pipeline(
            spark,
            train_path=args.train_path,
            test_path=args.test_path,
            prices_long_path=args.prices_long_path,
            out_train=args.out_train,
            out_test=args.out_test,
            out_join_train=args.out_join_train,
            out_join_test=args.out_join_test
        )
    finally:
        spark.stop()
