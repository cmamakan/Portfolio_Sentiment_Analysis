"""
06_bl_portfolios.py
Black–Litterman (Q = LSTM predictions) vs Equal-Weight vs Sentiment-only

Inputs:
- prices_long.csv
- prediction_lstm.csv
- daily_sentiment.csv
- risk_free_IRX.csv  (optional; improves Sharpe/Sortino via excess returns)

Outputs:
- weights_LSTM.csv
- bt_LSTM.csv
- outputs/nav_compare_LSTM.png
- outputs/corr_matrix_LSTM.png
- outputs/metrics_bt_LSTM.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIG ===============
MODEL = "LSTM"
PRICES_CSV = "prices_long.csv"
PREDS_FILE = "prediction_lstm.csv"
SENT_CSV = "daily_sentiment.csv"
RISKFREE_CSV = "risk_free_IRX.csv"   # optional

LAM = 1.0   # prior risk aversion
TAU = 0.05  # prior uncertainty
K   = 0.5   # view uncertainty

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# =============== UTILS ===============
def normalize_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def get_price_col(df):
    for c in ["adj_close", "close", "price", "px", "last", "value"]:
        if c in df.columns:
            return c
    raise KeyError("No price column found (expected adj_close/close/price/px/last/value).")

def get_pred_col(df):
    for c in ["y_pred_return_t1", "ret_pred", "pred", "y_pred", "return_pred", "ret"]:
        if c in df.columns:
            return c
    raise KeyError("No predicted return column found (e.g., y_pred_return_t1/ret_pred/pred).")

def mv_weights_from_mu(Sigma, mu, long_only=True):
    Sig = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    try:
        w = np.linalg.solve(Sig, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(Sig) @ mu
    if long_only:
        w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w) / len(w)) if s <= 0 else (w / s)

def bl_posterior_mu(Sigma, pi, Q, tau, Omega_inv):
    A = np.linalg.inv(tau * Sigma) + Omega_inv
    b = np.linalg.inv(tau * Sigma) @ pi + Omega_inv @ Q
    return np.linalg.solve(A, b)

def Omega_inv_from(S_sub):
    diag = K * np.diag(TAU * S_sub) + 1e-8
    return np.diag(1.0 / diag)

# =============== 1) LOAD DATA ===============
prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
preds  = pd.read_csv(PREDS_FILE,  parse_dates=["date"])
sent   = pd.read_csv(SENT_CSV,    parse_dates=["date"])

prices = normalize_cols(prices)
preds  = normalize_cols(preds)
sent   = normalize_cols(sent)

prices["date"] = prices["date"].dt.date
preds["date"]  = preds["date"].dt.date
sent["date"]   = sent["date"].dt.date

prices["ticker"] = prices["ticker"].astype(str).str.upper()
preds["ticker"]  = preds["ticker"].astype(str).str.upper()
sent["ticker"]   = sent["ticker"].astype(str).str.upper()

price_col = get_price_col(prices)
pred_col  = get_pred_col(preds)

# Optional: risk-free daily series aligned by date (for excess returns in metrics)
rf_series = None
if Path(RISKFREE_CSV).exists():
    rf = pd.read_csv(RISKFREE_CSV, parse_dates=["date"])
    rf = normalize_cols(rf)
    rf["date"] = rf["date"].dt.date
    # convert annual % to daily decimal (252 trading days)
    rf["rf_daily"] = rf["irx_adj_close"] / 100.0 / 252.0
    rf_series = rf.set_index("date")["rf_daily"]
    print(f"✅ Risk-free loaded ({len(rf_series)} rows)")
else:
    print("ℹ️ risk_free_IRX.csv not found → metrics will use rf=0.")

# =============== 2) COVARIANCE (TRAIN) & PRIOR ===============
dmin_te, dmax_te = preds["date"].min(), preds["date"].max()

ret_all = (
    prices.sort_values(["ticker","date"])
          .assign(ret=lambda d: d.groupby("ticker")[price_col].pct_change())
)

ret_train = ret_all[ret_all["date"] < dmin_te]
tickers_common = sorted(set(prices["ticker"]) & set(preds["ticker"]))
R = ret_train.pivot(index="date", columns="ticker", values="ret")
R = R[tickers_common].fillna(0.0)

Sigma = R.cov().values
w_mkt = np.ones(len(tickers_common)) / len(tickers_common)
pi = LAM * Sigma.dot(w_mkt)

# --- correlation heatmap (clear) ---
corr = R.corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(tickers_common)), tickers_common, rotation=90)
plt.yticks(range(len(tickers_common)), tickers_common)
plt.title("Correlation matrix between assets (train window)")
plt.tight_layout()
corr_png = os.path.join(OUTDIR, f"corr_matrix_{MODEL}.png")
plt.savefig(corr_png, dpi=150)
plt.close()
print(f"✅ Saved correlation heatmap → {corr_png}")

# =============== 3) DAILY WEIGHTS (BL / EW / Sentiment) ===============
rows = []
for d, g in preds.groupby("date", sort=True):
    tkrs = list(g["ticker"].unique())
    tkrs = [t for t in tkrs if t in tickers_common]
    if not tkrs:
        continue
    idx = [tickers_common.index(t) for t in tkrs]
    S_d  = Sigma[np.ix_(idx, idx)]
    pi_d = pi[idx]
    Q_d  = g.set_index("ticker")[pred_col].reindex(tkrs).fillna(0.0).values.astype(float)

    Omega_inv_d = Omega_inv_from(S_d)
    mu_star = bl_posterior_mu(S_d, pi_d, Q_d, TAU, Omega_inv_d)

    w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)
    w_ew = np.ones(len(tkrs)) / len(tkrs)

    s_today = (
        sent[sent["date"] == d]
        .drop_duplicates("ticker")
        .set_index("ticker")["sentiment_mean"]
        .reindex(tkrs)
        .fillna(0.0)
        .values
        .astype(float)
    )
    s_pos = np.clip(s_today, 0, None)
    w_s = s_pos / s_pos.sum() if s_pos.sum() > 0 else np.zeros_like(s_pos)

    for t, w1, w2, w3 in zip(tkrs, w_bl, w_ew, w_s):
        rows.append({"date": d, "ticker": t, "w_BL": w1, "w_EW": w2, "w_Sent": w3})

weights = pd.DataFrame(rows).sort_values(["date", "ticker"])
weights_out = f"weights_{MODEL}.csv"
weights.to_csv(weights_out, index=False)
print(f"✅ Saved {weights_out} ({len(weights)} rows)")

# =============== 4) BACKTEST (TEST WINDOW) ===============
ret_test = ret_all[(ret_all["date"] >= dmin_te) & (ret_all["date"] <= dmax_te)]
ret_test = ret_test.pivot(index="date", columns="ticker", values="ret").fillna(0.0)

def run_bt(col):
    W = weights.pivot(index="date", columns="ticker", values=col).reindex(ret_test.index).fillna(0.0)
    common = [c for c in W.columns if c in ret_test.columns]
    return (W[common] * ret_test[common]).sum(axis=1)

bt = pd.DataFrame({
    "ret_BL":   run_bt("w_BL"),
    "ret_EW":   run_bt("w_EW"),
    "ret_Sent": run_bt("w_Sent"),
}).fillna(0.0)

for c in bt.columns:
    bt["cum_" + c.split("_")[1]] = (1.0 + bt[c]).cumprod()

bt_out = f"bt_{MODEL}.csv"
bt.to_csv(bt_out, index=True)
print(f"✅ Saved {bt_out} ({len(bt)} rows)")

# =============== 5) PERFORMANCE METRICS (with optional risk-free) ===============
def compute_metrics(df, col, rf_daily_series=None):
    r = df[col].copy()
    if rf_daily_series is not None:
        # align daily risk-free to df index; fill forward for missing dates
        rf_aligned = pd.Series(rf_daily_series).reindex(df.index).fillna(method="ffill").fillna(0.0)
        r = r - rf_aligned
    mean = r.mean()
    std = r.std()
    downside = r[r < 0].std()
    sharpe = mean / std if std > 0 else np.nan
    sortino = mean / downside if downside > 0 else np.nan
    cum = (1.0 + r).cumprod()
    # CAGR on trading days (assume 252)
    cagr = cum.iloc[-1] ** (252 / max(len(r),1)) - 1
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    maxdd = dd.min()
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Volatility": std,
        "Days": len(r)
    }

rf_aligned = None
if rf_series is not None:
    # Convert bt.index (date) to Series index for alignment
    rf_aligned = rf_series

metrics = []
for col in ["ret_BL", "ret_EW", "ret_Sent"]:
    m = compute_metrics(bt, col, rf_daily_series=rf_aligned)
    m["Portfolio"] = col.split("_")[1]
    metrics.append(m)

metrics_df = pd.DataFrame(metrics).set_index("Portfolio")
metrics_out = os.path.join(OUTDIR, f"metrics_bt_{MODEL}.csv")
metrics_df.to_csv(metrics_out)
print(f"✅ Saved metrics table → {metrics_out}")
print("\n=== PERFORMANCE METRICS (excess returns if rf available) ===")
print(metrics_df.round(4))

# =============== 6) PLOT NAVs (3 strategies) ===============
plt.figure(figsize=(10,6))
plt.plot(bt.index, bt["cum_BL"],   label=f"Black–Litterman ({MODEL})", linewidth=2)
plt.plot(bt.index, bt["cum_EW"],   label="Equal Weight", linewidth=2)
plt.plot(bt.index, bt["cum_Sent"], label="Sentiment-only", linewidth=2)
plt.title(f"Portfolio NAVs Comparison ({MODEL})")
plt.xlabel("Date"); plt.ylabel("Cumulative Value")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
png_out = os.path.join(OUTDIR, f"nav_compare_{MODEL}.png")
plt.savefig(png_out, dpi=150)
plt.close()
print(f"✅ Saved NAV plot → {png_out}")
