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
- prices_long.csv  → [date, ticker, adj_close, volume]  (SPY excluded)
- prices_wide.csv  → daily adjusted close prices in wide format (SPY kept)
- risk_free_IRX.csv → daily 13-week T-Bill rates (Adj Close)
"""

import pandas as pd
import yfinance as yf

# === Parameters ===
TICKERS = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","JPM","V","JNJ","GOOGL","SPY"]  # SPY kept only for wide format
START = "2023-01-01"
END = "2025-10-05"

print(f"Downloading daily prices from {START} to {END} for {len(TICKERS)} tickers...")

# === Download data ===
data = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    interval="1d",
    group_by="ticker",
    auto_adjust=False,
    threads=True,
)

# --- LONG format (without SPY)
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

# --- Remove SPY from prices_long only
before = len(long_df)
long_df = long_df[long_df["ticker"] != "SPY"]
after = len(long_df)
print(f"🧹 Removed {before - after} SPY rows from prices_long.csv")

# --- Save LONG (SPY excluded)
long_df = long_df[["date","ticker","adj_close","volume"]].sort_values(["date","ticker"])
long_df.to_csv("prices_long.csv", index=False)
print(f"✅ Saved prices_long.csv  (rows: {len(long_df)})")

# --- WIDE format (SPY kept)
wide_rows = []
for tkr in TICKERS:
    if tkr not in data.columns.get_level_values(0):
        continue
    df_t = data[tkr][["Adj Close"]].copy()
    df_t = df_t.rename(columns={"Adj Close": tkr})
    df_t = df_t.reset_index().rename(columns={"Date": "date"})
    wide_rows.append(df_t.set_index("date"))

wide_df = pd.concat(wide_rows, axis=1)
wide_df.to_csv("prices_wide.csv")
print(f"✅ Saved prices_wide.csv  (rows: {len(wide_df)}, cols: {len(wide_df.columns)})")

# --- Quick checks
print("\nFirst/last dates:", wide_df.index.min().date(), "→", wide_df.index.max().date())
print("Missing counts per ticker (Adj Close):")
print(wide_df.isna().sum())

# === Risk-free rate (13-week T-Bill) ===
_rf = yf.download("^IRX", start=START, end=END, interval="1d", auto_adjust=False)

if isinstance(_rf, pd.DataFrame):
    rf = _rf.get("Adj Close")
    if isinstance(rf, pd.DataFrame):
        rf = rf.squeeze()
else:
    rf = _rf

rf = rf.astype("float64").dropna()
rf.name = "IRX_adj_close"
rf.index = rf.index.date
rf.to_csv("risk_free_IRX.csv")
print(f"✅ Saved risk_free_IRX.csv  (rows: {len(rf)})")
