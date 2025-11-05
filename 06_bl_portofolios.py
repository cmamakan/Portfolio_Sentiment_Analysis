"""
06_bl_portfolios_LSTM.py
Black–Litterman (Q = LSTM-implied log-returns from price predictions)
vs Equal-Weight vs Sentiment-only vs Mean-Variance vs S&P 500 (SPY).

Inputs:
- prices_wide.csv
    → wide prices: [date, AAPL, AMZN, ..., SPY, ...]
- prediction_lstm.csv
    → [date, ticker, price_true_t1, price_pred_t1, price_prev_t,
       y_true_return_t1, y_pred_return_t1]
       (y_pred_return_t1 = log-returns from predicted prices)
- daily_sentiment.csv
    → [date, ticker, sentiment_mean]
- risk_free_IRX.csv (optional)
    → [date, irx_adj_close] annualized %, used to compute excess returns

Outputs:
- weights_LSTM.csv
    → daily weights by ticker:
      [date, ticker, w_BL_LSTM, w_EW, w_Sent, w_MV]
- bt_LSTM.csv
    → backtest series: daily returns and cumulative NAVs for:
      BL_LSTM, EW, Sent, MV, SPY
- outputs/corr_matrix_LSTM.png
    → correlation matrix (train window)
- outputs/nav_compare_LSTM.png
    → cumulative NAV comparison for the 5 portfolios
- outputs/metrics_bt_LSTM.csv
    → performance table (CAGR, Sharpe, Sortino, MaxDD, Volatility, Days)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIG ===============
MODEL = "LSTM"
PRICES_WIDE_CSV = "prices_wide.csv"
PREDS_FILE = "prediction_lstm.csv"
SENT_CSV = "daily_sentiment.csv"
RISKFREE_CSV = "risk_free_IRX.csv"   # optional

LAM = 1.0   # prior risk aversion (for BL prior Π)
TAU = 0.05  # prior uncertainty
K   = 0.5   # view uncertainty

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


# =============== UTILS ===============
def normalize_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def mv_weights_from_mu(Sigma, mu, long_only=True):
    """
    Mean-variance weights (Markowitz):
    w ∝ Σ^{-1} μ  (approximation; then normalized and clipped if long_only).
    """
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
    """
    Standard Black–Litterman posterior mean:
    mu* = ( (tau Σ)^{-1} + Ω^{-1} )^{-1} [ (tau Σ)^{-1} π + Ω^{-1} Q ].
    """
    inv_tauSigma = np.linalg.inv(tau * Sigma)
    A = inv_tauSigma + Omega_inv
    b = inv_tauSigma @ pi + Omega_inv @ Q
    return np.linalg.solve(A, b)


def Omega_inv_from(S_sub):
    """
    Ω^{-1} diagonal based on uncertainty parameter K and TAU.
    """
    diag = K * np.diag(TAU * S_sub) + 1e-8
    return np.diag(1.0 / diag)


def compute_metrics(ret_series: pd.Series, rf_daily_series: pd.Series | None = None):
    """
    ret_series: daily arithmetic returns.
    rf_daily_series: optional daily risk-free rate (aligned by date index).
    """
    r = ret_series.copy()
    if rf_daily_series is not None:
        rf_aligned = rf_daily_series.reindex(r.index).fillna(method="ffill").fillna(0.0)
        r = r - rf_aligned

    mean = r.mean()
    std = r.std()
    downside = r[r < 0].std()
    sharpe = mean / std if std > 0 else np.nan
    sortino = mean / downside if downside > 0 else np.nan

    cum = (1.0 + r).cumprod()
    # CAGR sur nb de jours de trading (≈ 252)
    if len(r) > 0:
        cagr = cum.iloc[-1] ** (252 / len(r)) - 1
    else:
        cagr = np.nan

    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    maxdd = dd.min() if len(dd) > 0 else np.nan

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": maxdd,
        "Volatility": std,
        "Days": len(r),
    }


# =============== 1) LOAD DATA ===============
prices_wide = pd.read_csv(PRICES_WIDE_CSV, parse_dates=["date"])
preds       = pd.read_csv(PREDS_FILE,       parse_dates=["date"])
sent        = pd.read_csv(SENT_CSV,         parse_dates=["date"])

prices_wide["date"] = prices_wide["date"].dt.date
preds["date"]       = preds["date"].dt.date
sent["date"]        = sent["date"].dt.date

# Uniformiser les noms de colonnes / tickers
preds["ticker"] = preds["ticker"].astype(str).str.upper()
sent["ticker"]  = sent["ticker"].astype(str).str.upper()

# Tickers dans prices_wide (format wide)
all_cols = [c for c in prices_wide.columns if c != "date"]
spy_ticker = "SPY" if "SPY" in [c.upper() for c in all_cols] else None

# On force les colonnes tickers en uppercase pour matcher les autres fichiers
col_map = {c: c.upper() for c in all_cols}
prices_wide = prices_wide.rename(columns=col_map)
all_tickers = [c for c in prices_wide.columns if c != "date"]

if spy_ticker and spy_ticker not in all_tickers:
    spy_ticker = None  # par sécurité si renommage foire

# Univers actif = tous les tickers SAUF SPY (SPY = benchmark à part)
active_tickers = [t for t in all_tickers if t != spy_ticker]

# Restreindre aux tickers présents dans les prédictions et le sentiment
pred_tickers = set(preds["ticker"].unique())
sent_tickers = set(sent["ticker"].unique())
active_tickers = sorted(set(active_tickers) & pred_tickers & sent_tickers)

if not active_tickers:
    raise SystemExit("No common active tickers between prices_wide, prediction_lstm and daily_sentiment.")


# =============== 2) RETURNS (log-returns) ===============
# Matrice de prix (wide) pour l'univers actif
P = prices_wide.set_index("date")[active_tickers].sort_index()

# Log-returns: r_t = log(P_t / P_{t-1})
R_log = np.log(P / P.shift(1))

# SPY (benchmark)
spy_log_ret = None
if spy_ticker and spy_ticker in prices_wide.columns:
    spy_price = prices_wide.set_index("date")[spy_ticker].sort_index()
    spy_log_ret = np.log(spy_price / spy_price.shift(1))

# Train / Test split basé sur les dates de prédiction LSTM
dmin_te, dmax_te = preds["date"].min(), preds["date"].max()

R_log_train = R_log.loc[R_log.index < dmin_te].dropna(how="all")
R_log_test  = R_log.loc[(R_log.index >= dmin_te) & (R_log.index <= dmax_te)].fillna(0.0)

if R_log_train.empty or R_log_test.empty:
    raise SystemExit("Train or test window is empty. Check dates in prices_wide.csv and prediction_lstm.csv.")


# =============== 3) COVARIANCE, PRIOR (BL) & MEAN-VAR ===============
Sigma = R_log_train.cov().values              # covariance matrix
n_assets = len(active_tickers)

# ---- Prior for Black–Litterman: Π = λ Σ w_mkt  (reverse optimization)
w_mkt = np.ones(n_assets) / n_assets
pi_bl = LAM * Sigma.dot(w_mkt)

# ---- Mean-Variance portfolio: μ = historical mean log-returns (Markowitz)
mu_hist = R_log_train.mean().values           # shape (n_assets,)
w_MV_full = mv_weights_from_mu(Sigma, mu_hist, long_only=True)


# =============== 4) CORRELATION HEATMAP (train) ===============
corr = R_log_train.corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr.values, vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(n_assets), active_tickers, rotation=90)
plt.yticks(range(n_assets), active_tickers)
plt.title("Correlation matrix between assets (train window)")
plt.tight_layout()
corr_png = os.path.join(OUTDIR, f"corr_matrix_{MODEL}.png")
plt.savefig(corr_png, dpi=150)
plt.close()
print(f"✅ Saved correlation heatmap → {corr_png}")


# =============== 5) ALIGN PREDICTIONS (Q) & SENTIMENT ===============
# On garde seulement les prédictions pour l'univers actif et la fenêtre test
preds = preds.copy()
preds = preds[(preds["ticker"].isin(active_tickers)) &
              (preds["date"] >= dmin_te) &
              (preds["date"] <= dmax_te)]

# On vérifie qu'on a bien y_pred_return_t1 (log-return prédit)
if "y_pred_return_t1" not in preds.columns:
    raise KeyError("prediction_lstm.csv must contain y_pred_return_t1 (log-return predictions).")

# Sentiment : on garde seulement l'univers actif
sent = sent.copy()
sent = sent[sent["ticker"].isin(active_tickers)]


# =============== 6) DAILY WEIGHTS (BL_LSTM / EW / Sent / MV) ===============
rows = []

for d in sorted(R_log_test.index):
    # d est un objet date
    tkrs = list(active_tickers)
    idx = [active_tickers.index(t) for t in tkrs]

    # Sous-matrice de covariance & prior BL
    S_d  = Sigma[np.ix_(idx, idx)]
    pi_d = pi_bl[idx]

    # Vues Q_d = log-returns prédits par LSTM pour la date d
    g_pred = preds[preds["date"] == d]
    if g_pred.empty:
        # pas de vues ce jour-là → BL se replie vers le prior
        Q_d = pi_d.copy()
    else:
        Q_d = (
            g_pred.set_index("ticker")["y_pred_return_t1"]
                  .reindex(tkrs)
                  .fillna(0.0)
                  .values
                  .astype(float)
        )

    # Black–Litterman posterior (BL_LSTM)
    Omega_inv_d = Omega_inv_from(S_d)
    mu_star = bl_posterior_mu(S_d, pi_d, Q_d, TAU, Omega_inv_d)
    w_BL = mv_weights_from_mu(S_d, mu_star, long_only=True)

    # Equal-weight (EW)
    w_EW = np.ones(len(tkrs)) / len(tkrs)

    # Sentiment-only: poids pro-rata du sentiment positif
    g_sent = (
        sent[sent["date"] == d]
        .drop_duplicates("ticker")
        .set_index("ticker")["sentiment_mean"]
        .reindex(tkrs)
        .fillna(0.0)
        .astype(float)
    )
    s_pos = np.clip(g_sent.values, 0, None)
    if s_pos.sum() > 0:
        w_Sent = s_pos / s_pos.sum()
    else:
        w_Sent = np.zeros_like(s_pos)

    # Mean-Variance baseline: sous-ensemble des w_MV_full (Markowitz)
    w_MV_sub = w_MV_full[idx]
    w_MV_sub = np.clip(w_MV_sub, 0, None)
    if w_MV_sub.sum() > 0:
        w_MV = w_MV_sub / w_MV_sub.sum()
    else:
        w_MV = np.ones(len(tkrs)) / len(tkrs)

    for t, w1, w2, w3, w4 in zip(tkrs, w_BL, w_EW, w_Sent, w_MV):
        rows.append({
            "date": d,
            "ticker": t,
            "w_BL_LSTM": w1,
            "w_EW":      w2,
            "w_Sent":    w3,
            "w_MV":      w4,
        })

weights = pd.DataFrame(rows).sort_values(["date", "ticker"])
weights_out = "weights_LSTM.csv"
weights.to_csv(weights_out, index=False)
print(f"✅ Saved {weights_out} ({len(weights)} rows)")


# =============== 7) BACKTEST (BL_LSTM / EW / Sent / MV / SPY) ===============
# R_log_test : log-returns des actifs (univers actif)
# On convertit en arithmetic returns pour le backtest.
R_test_simple = np.exp(R_log_test) - 1.0  # same index & columns

def run_bt(col_weights: str) -> pd.Series:
    """
    Retourne la série de retours quotidiens du portefeuille
    défini par la colonne de poids col_weights.
    """
    W = (
        weights.pivot(index="date", columns="ticker", values=col_weights)
               .reindex(R_test_simple.index)
               .fillna(0.0)
    )
    common = [c for c in W.columns if c in R_test_simple.columns]
    if not common:
        raise SystemExit(f"No common tickers between weights[{col_weights}] and test returns.")
    port_ret = (W[common] * R_test_simple[common]).sum(axis=1)
    return port_ret


ret_BL   = run_bt("w_BL_LSTM")
ret_EW   = run_bt("w_EW")
ret_Sent = run_bt("w_Sent")
ret_MV   = run_bt("w_MV")

# SPY: benchmark (si disponible)
ret_SPY = None
if spy_log_ret is not None:
    spy_log_ret_test = spy_log_ret.loc[R_test_simple.index]
    ret_SPY = np.exp(spy_log_ret_test) - 1.0

bt = pd.DataFrame({
    "ret_BL_LSTM": ret_BL,
    "ret_EW":      ret_EW,
    "ret_Sent":    ret_Sent,
    "ret_MV":      ret_MV,
})

if ret_SPY is not None:
    bt["ret_SPY"] = ret_SPY.reindex(bt.index).fillna(0.0)

# Cumulative NAVs
for c in bt.columns:
    suffix = c.split("_", 1)[1]  # BL_LSTM, EW, Sent, MV, SPY
    bt["cum_" + suffix] = (1.0 + bt[c]).cumprod()

bt_out = "bt_LSTM.csv"
bt.to_csv(bt_out, index=True)
print(f"✅ Saved {bt_out} ({len(bt)} rows)")


# =============== 8) PERFORMANCE METRICS (with optional risk-free) ===============
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

rf_daily_series = rf_series if rf_series is not None else None

metrics_rows = []
for col in ["ret_BL_LSTM", "ret_EW", "ret_Sent", "ret_MV"] + (["ret_SPY"] if "ret_SPY" in bt.columns else []):
    m = compute_metrics(bt[col], rf_daily_series=rf_daily_series)
    m["Portfolio"] = col.split("_", 1)[1]  # BL_LSTM, EW, Sent, MV, SPY
    metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("Portfolio")
metrics_out = os.path.join(OUTDIR, "metrics_bt_LSTM.csv")
metrics_df.to_csv(metrics_out)
print(f"✅ Saved metrics table → {metrics_out}")
print("\n=== PERFORMANCE METRICS (excess returns if rf available) ===")
print(metrics_df.round(4))


# =============== 9) PLOT NAVs (5 strategies) ===============
plt.figure(figsize=(10, 6))
plt.plot(bt.index, bt["cum_BL_LSTM"], label="Black–Litterman (LSTM)", linewidth=2)
plt.plot(bt.index, bt["cum_EW"],      label="Equal Weight",           linewidth=2)
plt.plot(bt.index, bt["cum_Sent"],    label="Sentiment-only",         linewidth=2)
plt.plot(bt.index, bt["cum_MV"],      label="Mean-Variance",          linewidth=2)
if "cum_SPY" in bt.columns:
    plt.plot(bt.index, bt["cum_SPY"], label="S&P 500 (SPY)",          linewidth=2)

plt.title("Portfolio Comparison")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
png_out = os.path.join(OUTDIR, "nav_compare_LSTM.png")
plt.savefig(png_out, dpi=150)
plt.close()
print(f"✅ Saved NAV plot → {png_out}")
