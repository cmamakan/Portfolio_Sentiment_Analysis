# -*- coding: utf-8 -*-
"""
06_bl_portofolios.py
Black–Litterman (Q = prédictions du modèle choisi) vs Equal-Weight vs Sentiment-only

Fichiers utilisés (conformes à ta capture):
- prices_long.csv
- prediction_gru.csv / prediction_lstm.csv / prediction_rnn.csv
- prediction_summary_metrics.csv (optionnel pour auto-pick)
- sentiment_daily.csv (optionnel)
- risk_free_IRX.csv (optionnel, non nécessaire)

Sorties:
- weights_{MODEL}.csv
- bt_{MODEL}.csv
- outputs/nav_compare_{MODEL}.png
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIG ===============
MODEL = "GRU"   # "GRU" | "LSTM" | "RNN"
AUTO_PICK_BEST = True  # True: lit prediction_summary_metrics.csv et choisit le meilleur modèle

PRICES_CSV = "prices_long.csv"
PREDS_BY_MODEL = {
    "GRU":  "prediction_gru.csv",
    "LSTM": "prediction_lstm.csv",
    "RNN":  "prediction_rnn.csv",
}
SENT_CSV_OPT = "sentiment_daily.csv"  # optionnel
METRICS_SUMMARY = "prediction_summary_metrics.csv"  # optionnel (pour AUTO_PICK_BEST)

# Hyperparamètres BL
LAM = 1.0   # échelle du prior
TAU = 0.05  # incertitude sur Sigma
K   = 0.5   # incertitude des vues (plus grand => moins confiant dans Q)

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# =============== UTILS ===============
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def get_price_col(df: pd.DataFrame) -> str:
    for c in ["adj_close", "close", "price", "px", "last", "value"]:
        if c in df.columns:
            return c
    raise KeyError("Colonne prix non trouvée (attendu: adj_close/close/price/px/last/value).")

def get_pred_col(df: pd.DataFrame) -> str:
    for c in ["y_pred_return_t1", "ret_pred", "pred", "y_pred", "return_pred", "ret"]:
        if c in df.columns:
            return c
    raise KeyError("Colonne de retour prédit non trouvée (ex: y_pred_return_t1/ret_pred/pred).")

def mv_weights_from_mu(Sigma: np.ndarray, mu: np.ndarray, long_only: bool = True) -> np.ndarray:
    Sig = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    try:
        w = np.linalg.solve(Sig, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(Sig) @ mu
    if long_only:
        w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w) / len(w)) if s <= 0 else (w / s)

def bl_posterior_mu(Sigma: np.ndarray, pi: np.ndarray, Q: np.ndarray, tau: float, Omega_inv: np.ndarray) -> np.ndarray:
    A = np.linalg.inv(tau * Sigma) + Omega_inv
    b = np.linalg.inv(tau * Sigma) @ pi + Omega_inv @ Q
    return np.linalg.solve(A, b)

def Omega_inv_from(S_sub: np.ndarray) -> np.ndarray:
    diag = K * np.diag(TAU * S_sub) + 1e-8
    return np.diag(1.0 / diag)

def choose_best_model(default_model: str) -> str:
    if not AUTO_PICK_BEST or not Path(METRICS_SUMMARY).exists():
        return default_model
    df = pd.read_csv(METRICS_SUMMARY)
    # colonnes possibles : Model, Sharpe, Sortino, CAGR, etc.
    cols = [c.lower() for c in df.columns]
    # normalise
    df.columns = cols
    # priorité Sharpe, sinon CAGR, sinon garde défaut
    key = "sharpe" if "sharpe" in cols else ("cagr" if "cagr" in cols else None)
    if key is None:
        return default_model
    df = df.sort_values(key, ascending=False)
    candidate = str(df.iloc[0]["model"]).upper() if "model" in cols else default_model
    candidate = candidate if candidate in {"GRU","LSTM","RNN"} else default_model
    return candidate

# =============== 1) LECTURE ===============
# modèle choisi
MODEL = choose_best_model(MODEL)
preds_file = PREDS_BY_MODEL[MODEL]
if not Path(preds_file).exists():
    # fallback: prend le 1er fichier existant
    for m, f in PREDS_BY_MODEL.items():
        if Path(f).exists():
            MODEL = m
            preds_file = f
            break
if not Path(preds_file).exists():
    raise FileNotFoundError(f"Aucun fichier de prédiction trouvé parmi: {list(PREDS_BY_MODEL.values())}")

# PRIX
if not Path(PRICES_CSV).exists():
    raise FileNotFoundError(f"Introuvable: {PRICES_CSV}")
prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
prices = normalize_cols(prices)
prices["date"] = prices["date"].dt.date
prices["ticker"] = prices["ticker"].astype(str).str.upper()
price_col = get_price_col(prices)

# PREDS (modèle)
preds = pd.read_csv(preds_file, parse_dates=["date"])
preds = normalize_cols(preds)
preds["date"] = preds["date"].dt.date
preds["ticker"] = preds["ticker"].astype(str).str.upper()
pred_col = get_pred_col(preds)

# SENTIMENT (optionnel)
sent_df = None
if Path("sentiment_daily.csv").exists():
    s = pd.read_csv("sentiment_daily.csv", parse_dates=["date"])
    s = normalize_cols(s)
    if "ticker" in s.columns and "date" in s.columns:
        s["date"] = s["date"].dt.date
        s["ticker"] = s["ticker"].astype(str).str.upper()
        if "sentiment_mean" in s.columns:
            s_col = "sentiment_mean"
        elif "sentiment" in s.columns:
            s_col = "sentiment"
        else:
            s_col = None
        if s_col is not None:
            sent_df = s[["date","ticker",s_col]].rename(columns={s_col:"sent"}).copy()

# TICKERS communs prix/prédictions
tickers_prices = set(prices["ticker"].unique())
tickers_preds  = set(preds["ticker"].unique())
tickers_all = sorted(tickers_prices & tickers_preds)
if len(tickers_all) == 0:
    raise RuntimeError("Aucun ticker commun entre prices_long.csv et le fichier de prédictions.")

# Fenêtre test = fenêtre prédictions
dmin_te, dmax_te = preds["date"].min(), preds["date"].max()

# =============== 2) COVARIANCE TRAIN & PRIOR ===============
ret_all = (
    prices.sort_values(["ticker","date"])
          .assign(ret=lambda d: d.groupby("ticker", observed=True)[price_col].pct_change())
)
ret_train = ret_all[ret_all["date"] < dmin_te]

R = ret_train.pivot(index="date", columns="ticker", values="ret").fillna(0.0)
R = R[[c for c in tickers_all if c in R.columns]]
assets = list(R.columns); N = len(assets)
if N == 0:
    raise RuntimeError("Pas d'actifs pour Sigma (check prices_long.csv / prédictions).")

Sigma = R.cov().values
w_mkt = np.ones(N)/N
pi = LAM * Sigma.dot(w_mkt)

# =============== 3) POIDS JOURNALIERS (BL / EW / Sentiment) ===============
rows = []
preds_use = preds[["date","ticker",pred_col]].copy()

for d, g in preds_use.groupby("date", sort=True):
    tkrs = list(g["ticker"].unique())
    idx = [assets.index(t) for t in tkrs if t in assets]
    tkrs = [assets[i] for i in idx]
    if len(idx) == 0:
        continue

    S_d  = Sigma[np.ix_(idx, idx)]
    pi_d = pi[idx]

    g_unique = g.drop_duplicates("ticker").set_index("ticker")
    Q_d = g_unique[pred_col].reindex(tkrs).fillna(0.0).values.astype(float)

    Omega_inv_d = Omega_inv_from(S_d)
    mu_star = bl_posterior_mu(S_d, pi_d, Q_d, TAU, Omega_inv_d)

    w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)
    w_ew = np.ones(len(tkrs)) / len(tkrs)

    if sent_df is not None:
        s_today = (
            sent_df[sent_df["date"]==d]
            .drop_duplicates("ticker")
            .set_index("ticker")["sent"]
            .reindex(tkrs).fillna(0.0).values.astype(float)
        )
        s_pos = np.clip(s_today, 0, None)
    else:
        s_pos = np.clip(Q_d, 0, None)
    w_s = s_pos/s_pos.sum() if s_pos.sum()>0 else np.zeros_like(s_pos)

    for t, w1, w2, w3 in zip(tkrs, w_bl, w_ew, w_s):
        rows.append({"date": d, "ticker": t, "w_BL": float(w1), "w_EW": float(w2), "w_Sent": float(w3)})

weights = pd.DataFrame(rows).sort_values(["date","ticker"])
weights_out = f"weights_{MODEL}.csv"
weights.to_csv(weights_out, index=False)
print(f"✅ Saved {weights_out} | rows: {len(weights)}")

# =============== 4) MINI BACKTEST ===============
ret_test = (
    ret_all[(ret_all["date"]>=dmin_te) & (ret_all["date"]<=dmax_te)]
    .pivot(index="date", columns="ticker", values="ret").fillna(0.0)
)

def run_bt(col):
    W = weights.pivot(index="date", columns="ticker", values=col).reindex(ret_test.index).fillna(0.0)
    common = [c for c in W.columns if c in ret_test.columns]
    W = W[common]; R = ret_test[common]
    return (W*R).sum(axis=1)

bt = pd.DataFrame({
    "ret_BL":   run_bt("w_BL"),
    "ret_EW":   run_bt("w_EW"),
    "ret_Sent": run_bt("w_Sent"),
}).fillna(0.0)

for c in bt.columns:
    bt["cum_"+c.split("_")[1]] = (1.0 + bt[c]).cumprod()

bt_out = f"bt_{MODEL}.csv"
bt.to_csv(bt_out, index=True)
print(f"✅ Saved {bt_out} | rows: {len(bt)}")

# =============== 5) COURBE ===============
plt.figure(figsize=(10,6))
plt.plot(bt.index, bt["cum_BL"],   label=f"Black–Litterman ({MODEL})")
plt.plot(bt.index, bt["cum_EW"],   label="Equal Weight")
plt.plot(bt.index, bt["cum_Sent"], label="Sentiment-only")
plt.title(f"Évolution du portefeuille ({MODEL})")
plt.xlabel("Date"); plt.ylabel("Valeur cumulée")
plt.legend(); plt.grid(True); plt.tight_layout()
png_out = os.path.join(OUTDIR, f"nav_compare_{MODEL}.png")
plt.savefig(png_out, dpi=150)
print(f"✅ Courbe enregistrée → {png_out}")
