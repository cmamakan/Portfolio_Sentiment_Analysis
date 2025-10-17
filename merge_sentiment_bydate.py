import pandas as pd

df = pd.read_csv("test_sentiment_prices.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker","date"])

# Rendements journaliers par ticker (à partir d'adj_close)
df["return"] = df.groupby("ticker")["adj_close"].pct_change()

# Aligner "sentiment_t" avec "return_{t+1}" (effet lendemain)
df["sentiment_t"] = df["sentiment_mean"]
df["return_t1"]   = df.groupby("ticker")["return"].shift(-1)

# Garder juste la fenêtre où on a du sentiment
out = df.dropna(subset=["sentiment_t"])
out.to_csv("test_sentiment_for_model.csv", index=False)

print(out[["date","ticker","sentiment_t","return_t1"]].head())