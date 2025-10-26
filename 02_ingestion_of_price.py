"""
Downloads daily adjusted close prices and volumes
for a selected list of tickers using the Yahoo Finance API (yfinance).

The script prepares the data in both long and wide formats
and also downloads the 13-week Treasury Bill (^IRX) as a proxy
for the risk-free rate, used later in portfolio evaluation.

Purpose:
To collect clean and consistent financial market data 
aligned with the news sentiment dataset by date and ticker.

Outputs:
- prices_long.csv  → [date, ticker, adj_close, volume]
- prices_wide.csv  → daily adjusted close prices in wide format
- risk_free_IRX.csv → daily 13-week T-Bill rates (Adj Close)
"""

import pandas as pd
import yfinance as yf

# === Paramètres ===
TICKERS = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","JPM","V","JNJ","GOOGL","SPY"]  # SPY = benchmark
START = "2023-01-01"
END = "2025-10-05"  # jusqu'au 5 octobre 2025, cohérent avec tes fichiers de news

print(f"Downloading daily prices from {START} to {END} for {len(TICKERS)} tickers...")

# === Téléchargement des prix (quotidien) ===
data = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    interval="1d",
    group_by="ticker",
    auto_adjust=False,
    threads=True,
)

# --- Format LONG : date, ticker, adj_close, volume
rows = []
for tkr in TICKERS:
    if tkr not in data.columns.get_level_values(0):
        print(f"[WARN] No data for {tkr} in requested range.")
        continue

    df_t = data[tkr][["Adj Close", "Volume"]].copy()
    df_t = df_t.rename(columns={"Adj Close": "adj_close", "Volume": "volume"})
    df_t["ticker"] = tkr
    df_t = df_t.reset_index().rename(columns={"Date": "date"})
    rows.append(df_t)

long_df = pd.concat(rows, ignore_index=True)
long_df = long_df.dropna(subset=["adj_close"])
long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

# --- Sauvegarde LONG
long_df = long_df[["date","ticker","adj_close","volume"]].sort_values(["date","ticker"])
long_df.to_csv("prices_long.csv", index=False)
print(f"✅ Saved prices_long.csv  (rows: {len(long_df)})")

# --- Format WIDE (Adj Close seulement)
wide_df = long_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
wide_df.to_csv("prices_wide.csv")
print(f"✅ Saved prices_wide.csv  (rows: {len(wide_df)}, cols: {len(wide_df.columns)})")

# --- Vérifications rapides
print("\nFirst/last dates:", wide_df.index.min(), "→", wide_df.index.max())
print("Missing counts per ticker (Adj Close):")
print(wide_df.isna().sum())

# === Taux sans risque (T-Bill 13 semaines) ===
_rf = yf.download("^IRX", start=START, end=END, interval="1d", auto_adjust=False)

# Récupère proprement la colonne 'Adj Close' quelle que soit la forme retournée
if isinstance(_rf, pd.DataFrame):
    rf = _rf.get("Adj Close")
    if isinstance(rf, pd.DataFrame):  # si encore DataFrame (cas rare)
        rf = rf.squeeze()
else:
    rf = _rf  # fallback

# Nommer la Series et nettoyer
rf = rf.astype("float64").dropna()
rf.name = "IRX_adj_close"
rf.index = rf.index.date

rf.to_csv("risk_free_IRX.csv")
print(f"✅ Saved risk_free_IRX.csv  (rows: {len(rf)})")