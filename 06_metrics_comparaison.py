# make_metrics_from_predictions.py
# Rebuild metrics & backtests from prediction files (no training needed)
# Inputs:
#   - sentiment_prices.csv
#   - prediction_rnn.csv / prediction_lstm.csv / prediction_gru.csv
# Outputs:
#   - bt_RNN.csv / bt_LSTM.csv / bt_GRU.csv
#   - recap_metrics_models.csv

import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

DATA_CSV = "sentiment_prices.csv"
PRED_FILES = {
    "RNN":  "prediction_rnn.csv",
    "LSTM": "prediction_lstm.csv",
    "GRU":  "prediction_gru.csv",
}

# ---- Helpers
def sharpe(r):  mu, sd = r.mean(), r.std(ddof=1); return np.nan if sd==0 else (mu/sd)*np.sqrt(252)
def sortino(r): d=r[r<0]; sd=d.std(ddof=1); return np.nan if sd==0 else (r.mean()/sd)*np.sqrt(252)
def maxdd(cum): peak=cum.cummax(); dd=cum/peak-1.0; return dd.min()

def mv_weights_from_mu(Sigma, mu, long_only=True):
    S = Sigma + 1e-6*np.eye(Sigma.shape[0])
    w = np.linalg.solve(S, mu)
    if long_only: w = np.clip(w,0,None)
    s = w.sum()
    return (np.ones_like(w)/len(w)) if s<=0 else (w/s)

def bl_mu(Sigma, pi, Q, tau, Omega_inv):
    A = np.linalg.inv(tau*Sigma) + Omega_inv
    b = np.linalg.inv(tau*Sigma)@pi + Omega_inv@Q
    return np.linalg.solve(A,b)

def Omega_inv_from(S, TAU=0.05, K=0.5):
    diag = K*np.diag(TAU*S) + 1e-8
    return np.diag(1.0/diag)

# ---- Load base data
df = pd.read_csv(DATA_CSV, parse_dates=["date"])
df["date"] = df["date"].dt.date
df["ticker"] = df["ticker"].str.upper()
df = df.sort_values(["ticker","date"])

# daily returns
ret_all = df.copy()
ret_all["ret"] = ret_all.groupby("ticker", observed=True)["adj_close"].pct_change()

# TRAIN (<2025) for Sigma; TEST (>=2025) for backtest
cut = pd.to_datetime("2025-01-01").date()
ret_train = (ret_all[ret_all["date"] <  cut]
             .groupby(["date","ticker"], as_index=False)["ret"].last())
ret_test  = (ret_all[ret_all["date"] >= cut]
             .groupby(["date","ticker"], as_index=False)["ret"].last())

R = (ret_train.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
               .sort_index().fillna(0.0))
assets = list(R.columns)
if len(assets) == 0:
    raise RuntimeError("No assets found in TRAIN period to compute Sigma.")
Sigma = R.cov().values
w_mkt = np.ones(len(assets))/len(assets)
LAM = 1.0
TAU = 0.05
K   = 0.5
pi = LAM * Sigma.dot(w_mkt)

RET_TEST = (ret_test.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
                     .sort_index().fillna(0.0))

# ---- Iterate models
rows_summary = []
for model_name, pred_path in PRED_FILES.items():
    if not Path(pred_path).exists():
        print(f"[WARN] Missing {pred_path} — skipping {model_name}")
        continue

    P = pd.read_csv(pred_path, parse_dates=["date"])
    P["date"] = P["date"].dt.date
    P["ticker"] = P["ticker"].str.upper()

    # ----- Regression metrics (global, over all tickers)
    # drop NA + keep aligned arrays
    p_clean = P.dropna(subset=["y_true_return_t1","y_pred_return_t1"]).copy()
    y_true = p_clean["y_true_return_t1"].to_numpy(float)
    y_pred = p_clean["y_pred_return_t1"].to_numpy(float)

    if len(y_true) == 0:
        print(f"[WARN] No rows to score for {model_name}")
        mse = rmse = mae = r2 = mape = np.nan
    else:
        mse  = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)

    # ----- Black–Litterman weights from predictions (views = predicted next-day returns)
    rows = []
    for d, g in p_clean.groupby("date"):
        tkrs_in_pred = list(g["ticker"].unique())

        # keep only assets present in Sigma universe
        tkrs = [t for t in tkrs_in_pred if t in assets]
        if not tkrs: 
            continue
        idx = [assets.index(t) for t in tkrs]
        S_d  = Sigma[np.ix_(idx, idx)]
        pi_d = pi[idx]
        Q_d = (g.drop_duplicates("ticker")
                 .set_index("ticker")["y_pred_return_t1"]
                 .reindex(tkrs).fillna(0.0).to_numpy(float))
        Omega_inv = Omega_inv_from(S_d, TAU=TAU, K=K)
        mu_star = bl_mu(S_d, pi_d, Q_d, TAU, Omega_inv)
        w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)

        for t, w in zip(tkrs, w_bl):
            rows.append({"date": d, "ticker": t, f"w_BL_{model_name}": w})

    W = (pd.DataFrame(rows)
           .groupby(["date","ticker"], as_index=False)
           .last())  # unique (date,ticker)

    # Align weights to test returns calendar
    wtab = (W.pivot_table(index="date", columns="ticker",
                          values=f"w_BL_{model_name}", aggfunc="last")
              .reindex(RET_TEST.index)
              .fillna(0.0))
    common = [c for c in wtab.columns if c in RET_TEST.columns]
    port = (wtab[common] * RET_TEST[common]).sum(axis=1)

    bt = pd.DataFrame({f"ret_BL_{model_name}": port})
    bt[f"cum_BL_{model_name}"] = (1 + bt[f"ret_BL_{model_name}"]).cumprod()
    bt.to_csv(f"bt_{model_name}.csv")

    shar = sharpe(bt[f"ret_BL_{model_name}"])
    sort = sortino(bt[f"ret_BL_{model_name}"])
    cagr = bt[f"cum_BL_{model_name}"].iloc[-1] - 1 if len(bt) else np.nan
    mdd  = maxdd(bt[f"cum_BL_{model_name}"]) if len(bt) else np.nan
    days = len(bt)

    rows_summary.append({
        "Model": model_name,
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape,
        "Sharpe": shar, "Sortino": sort, "CAGR": cagr, "MaxDD": mdd, "Days": days
    })

# ---- Save recap
if rows_summary:
    recap = pd.DataFrame(rows_summary).sort_values("Sharpe", ascending=False)
    recap.to_csv("recap_metrics_models.csv", index=False)
    print("\nSaved: recap_metrics_models.csv")
    print(recap.round(6))
else:
    print("[ERROR] No models were processed. Check prediction files exist and are non-empty.")