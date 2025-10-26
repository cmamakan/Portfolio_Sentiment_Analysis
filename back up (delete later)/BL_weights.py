# pip install pandas numpy
import pandas as pd, numpy as np
from pathlib import Path

# ====== FICHIERS ENTRÉE ======
PRICES_CSV = "prices_long.csv"        # déjà créé
PREDS_CSV  = "lstm_preds_test.csv"     # sorti par ton LSTM
SENT_TEST  = "daily_sentiment_2025_test.csv"  # optionnel : pour "Sentiment-only"

# ====== HYPERPARAMS BL (simples) ======
LAM = 1.0   # échelle du prior (aversion au risque)
TAU = 0.05  # incertitude sur Sigma
K   = 0.5   # incertitude des vues (plus grand => moins confiant dans Q)

def mv_weights_from_mu(Sigma, mu, long_only=True):
    Sig = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    w = np.linalg.solve(Sig, mu)
    if long_only: w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w)/len(w)) if s<=0 else (w/s)

def bl_posterior_mu(Sigma, pi, Q, tau, Omega_inv):
    A = np.linalg.inv(tau*Sigma) + Omega_inv
    b = np.linalg.inv(tau*Sigma)@pi + Omega_inv@Q
    return np.linalg.solve(A, b)

# ====== 1) LECTURE ======
prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
prices["date"] = prices["date"].dt.date
prices["ticker"] = prices["ticker"].str.upper()

preds = pd.read_csv(PREDS_CSV, parse_dates=["date"])
preds["date"] = preds["date"].dt.date
preds["ticker"] = preds["ticker"].str.upper()

sent_df = None
if Path(SENT_TEST).exists():
    sent_df = pd.read_csv(SENT_TEST, parse_dates=["date"])
    sent_df["date"] = sent_df["date"].dt.date
    sent_df["ticker"] = sent_df["ticker"].str.upper()

# Fenêtre test depuis les prédictions
dmin_te, dmax_te = preds["date"].min(), preds["date"].max()

# ====== 2) COVARIANCE TRAIN & PRIOR ======
ret_all = (prices.sort_values(["ticker","date"])
                 .assign(ret=lambda d: d.groupby("ticker", observed=True)["adj_close"].pct_change()))
ret_train = ret_all[ret_all["date"] < dmin_te]

# Matrice retours (dates x tickers), on exclut SPY des actifs
tickers_full = sorted(set(prices["ticker"]) - {"SPY"})
R = ret_train.pivot(index="date", columns="ticker", values="ret")
R = R[[c for c in tickers_full if c in R.columns]].fillna(0.0)

assets = list(R.columns); N = len(assets)
if N == 0: raise RuntimeError("Pas d'actifs pour Sigma. Vérifie prices_long.csv.")

Sigma = R.cov().values
w_mkt = np.ones(N)/N
pi = LAM * Sigma.dot(w_mkt)

def Omega_inv_from(S_sub):
    diag = K * np.diag(TAU * S_sub) + 1e-8
    return np.diag(1.0/diag)

# ====== 3) POIDS JOURNALIERS (BL, EW, Sentiment) ======
rows = []
for d, g in preds.groupby("date"):
    tkrs = list(g["ticker"].unique())
    idx = [assets.index(t) for t in tkrs if t in assets]
    tkrs = [assets[i] for i in idx]
    if len(idx) == 0: continue

    S_d  = Sigma[np.ix_(idx, idx)]
    pi_d = pi[idx]

    # Vues Q = retours prédits GRU (y_pred_return_t1)
    Q_d = (g.drop_duplicates("ticker").set_index("ticker")["y_pred_return_t1"]
             .reindex(tkrs).fillna(0.0).values)

    Omega_inv_d = Omega_inv_from(S_d)
    mu_star = bl_posterior_mu(S_d, pi_d, Q_d, TAU, Omega_inv_d)
    w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)

    w_ew = np.ones(len(tkrs))/len(tkrs)

    # Sentiment-only : poids ∝ sentiment positif si dispo, sinon ∝ Q positif
    if sent_df is not None:
        s_today = (sent_df[sent_df["date"]==d].drop_duplicates("ticker")
                      .set_index("ticker")["sentiment_mean"]
                      .reindex(tkrs).fillna(0.0).values)
        s_pos = np.clip(s_today, 0, None)
    else:
        s_pos = np.clip(Q_d, 0, None)
    w_s = s_pos/s_pos.sum() if s_pos.sum()>0 else np.zeros_like(s_pos)

    for t, w1, w2, w3 in zip(tkrs, w_bl, w_ew, w_s):
        rows.append({"date": d, "ticker": t, "w_BL": w1, "w_EW": w2, "w_Sent": w3})

weights = pd.DataFrame(rows).sort_values(["date","ticker"])
weights.to_csv("weights_test_BL_vs_baselines.csv", index=False)
print("✅ Saved weights_test_BL_vs_baselines.csv | rows:", len(weights))

# ====== 4) MINI BACKTEST (perf quotidienne & cumulées) ======
ret_test = ret_all[(ret_all["date"]>=dmin_te) & (ret_all["date"]<=dmax_te)] \
    .pivot(index="date", columns="ticker", values="ret").fillna(0.0)

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
    bt["cum_"+c.split("_")[1]] = (1+bt[c]).cumprod()
bt.to_csv("bt_BL_vs_baselines_test.csv")
print("✅ Saved bt_BL_vs_baselines_test.csv | rows:", len(bt))
