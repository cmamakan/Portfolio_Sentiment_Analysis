"""
Trains sequence models (RNN, LSTM, GRU) on market data + news sentiment
to predict next-day returns, then plugs predictions into a
Black–Litterman allocation and backtests performance.

Purpose:
To compare sequence models and evaluate their portfolio impact
with a clear, reproducible metric table.

Inputs:
- sentiment_prices.csv  → [date, ticker, adj_close, volume, sentiment_mean]

Outputs:
- prediction_rnn.csv, prediction_lstm.csv, prediction_gru.csv    → next-day return predictions (2025)
- weights_RNN.csv, weights_LSTM.csv, weights_GRU.csv             → BL portfolio weights (by date, ticker)
- bt_RNN.csv, bt_LSTM.csv, bt_GRU.csv                            → backtest returns & cumulative curves
- recap_metrics_models.csv                                       → MSE, RMSE, MAE, R², MAPE%, Sharpe, Sortino, CAGR, MaxDD, Days

Steps:
1) Build features per ticker: returns, log(volume), next-day target.
2) Split train (2023–2024) / test (2025).
3) Make rolling windows and train RNN/LSTM/GRU (PyTorch).
4) Use model predictions as BL “views”, compute weights and backtest.
5) Export predictions, weights, backtests, and a final comparison table.
"""

"""
Trains sequence models (RNN, LSTM, GRU) on market data + news sentiment
to predict next-day returns, then plugs predictions into a
Black–Litterman allocation and backtests performance.

Purpose:
To compare sequence models and evaluate their portfolio impact
with a clear, reproducible metric table.

Inputs:
- sentiment_prices.csv  → [date, ticker, adj_close, volume, sentiment_mean]

Outputs:
- prediction_rnn.csv, prediction_lstm.csv, prediction_gru.csv    → next-day return predictions (2025)
- weights_RNN.csv, weights_LSTM.csv, weights_GRU.csv             → BL portfolio weights (by date, ticker)
- bt_RNN.csv, bt_LSTM.csv, bt_GRU.csv                            → backtest returns & cumulative curves
- recap_metrics_models.csv                                       → MSE, RMSE, MAE, R², MAPE%, Sharpe, Sortino, CAGR, MaxDD, Days

Steps:
1) Build features per ticker: returns, log(volume), next-day target.
2) Split train (2023–2024) / test (2025).
3) Make rolling windows and train RNN/LSTM/GRU (PyTorch).
4) Use model predictions as BL “views”, compute weights and backtest.
5) Export predictions, weights, backtests, and a final comparison table.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sentiment_prices.csv", help="Path to sentiment_prices.csv")
    ap.add_argument("--window", type=int, default=10, help="Sequence length (days)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", help="Run only this subset of tickers (e.g., --only AAPL MSFT)")
    ap.add_argument("--quick", action="store_true", help="Quick smoke test: window=6, epochs=6")
    return ap.parse_args()


# =========================
# Utils
# =========================
def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def dedup_base(df: pd.DataFrame) -> pd.DataFrame:
    # 1 ligne par (date, ticker) — garder la dernière si doublon
    return (df.sort_values(["ticker", "date"])
              .drop_duplicates(subset=["date", "ticker"], keep="last")
              .reset_index(drop=True))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {k: np.nan for k in ["MSE", "RMSE", "MAE", "R2", "MAPE(%)"]}
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape}


def sharpe(r: pd.Series) -> float:
    mu, sd = r.mean(), r.std(ddof=1)
    return np.nan if sd == 0 else float((mu / sd) * np.sqrt(252))


def sortino(r: pd.Series) -> float:
    d = r[r < 0]
    sd = d.std(ddof=1)
    return np.nan if sd == 0 else float((r.mean() / sd) * np.sqrt(252))


def maxdd(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


# =========================
# Data → sequences
# =========================
def build_sequences_by_ticker(df: pd.DataFrame, feature_cols: List[str], target_col: str, window: int):
    X_list, y_list, dates, tickers = [], [], [], []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feature_cols].values.astype(np.float32)
        tgt = g[target_col].values.astype(np.float32)
        for i in range(len(g) - window):
            j = i + window
            y_i = tgt[j]
            if np.isnan(y_i):
                continue
            X_list.append(vals[i:j])
            y_list.append([y_i])
            dates.append(g.loc[j, "date"])
            tickers.append(tkr)
    return np.array(X_list), np.array(y_list), np.array(dates), np.array(tickers)


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


# =========================
# Models (PyTorch)
# =========================
class RNNHead(nn.Module):
    def __init__(self, in_features, cell="GRU", hidden=32, layers=1):
        super().__init__()
        if cell == "RNN":
            self.rnn = nn.RNN(in_features, hidden, layers, batch_first=True)
        elif cell == "LSTM":
            self.rnn = nn.LSTM(in_features, hidden, layers, batch_first=True)
        else:
            self.rnn = nn.GRU(in_features, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.cell = cell

    def forward(self, x):
        out = self.rnn(x)[0]  # (B,T,H)
        return self.head(out[:, -1, :])


def train_model(model, loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = crit(yhat, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[{model.cell}][Epoch {ep:02d}] train MSE: {np.mean(losses):.6f}")
    return model


def eval_model(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yhat = model(xb.to(device)).cpu().numpy()
            preds.append(yhat)
            trues.append(yb.numpy())
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()
    return compute_metrics(trues, preds), preds, trues


# =========================
# Robust linear algebra for BL
# =========================
def robust_inv(M, eps=1e-6):
    M = np.asarray(M, float)
    M = 0.5 * (M + M.T)
    for k in [eps, 1e-5, 1e-4, 1e-3]:
        try:
            return np.linalg.inv(M + k * np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(M)


def Omega_inv_from(S, TAU=0.05, K=0.5):
    d = np.diag(S).astype(float)
    d = np.where(d <= 1e-12, 1e-12, d)
    return np.diag(1.0 / (K * TAU * d + 1e-12))


def bl_mu(Sigma, pi, Q, tau, Omega_inv):
    inv_tauSigma = robust_inv(tau * Sigma, eps=1e-8)
    A = inv_tauSigma + Omega_inv
    b = inv_tauSigma @ pi + Omega_inv @ Q
    return robust_inv(A, eps=1e-8) @ b


def mv_weights_from_mu(Sigma, mu, long_only=True):
    S_reg = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(Sigma.shape[0])
    try:
        w = np.linalg.solve(S_reg, mu)
    except np.linalg.LinAlgError:
        w = robust_inv(S_reg) @ mu
    if long_only:
        w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w) / len(w)) if s <= 1e-12 else (w / s)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    if args.quick:
        args.window = 6
        args.epochs = 6

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Load & clean
    df = pd.read_csv(args.data, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df["ticker"] = df["ticker"].str.upper()
    df = dedup_base(df)

    # Optional subset of tickers
    if args.only:
        only = set([t.upper() for t in args.only])
        df = df[df["ticker"].isin(only)].copy()

    # Features/targets
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker", observed=True)["adj_close"].pct_change()
    df["target_next_return"] = df.groupby("ticker", observed=True)["return"].shift(-1)
    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)

    # Split by year
    train_df = df[(df["date"] >= pd.to_datetime("2023-01-01").date()) &
                  (df["date"] <= pd.to_datetime("2024-12-31").date())].copy()
    test_df = df[df["date"] >= pd.to_datetime("2025-01-01").date()].copy()

    # Scale numeric features
    scaler = StandardScaler()
    fit_mat = train_df[["adj_close", "log_volume", "return"]].fillna(0.0).values
    scaler.fit(fit_mat)

    def apply_scale(d):
        d = d.copy()
        d["return"] = d["return"].fillna(0.0)
        mat = d[["adj_close", "log_volume", "return"]].values
        d[["adj_close", "log_volume", "return"]] = scaler.transform(mat)
        return d

    train_df = apply_scale(train_df)
    test_df = apply_scale(test_df)

    feature_cols = ["sentiment_mean", "adj_close", "log_volume", "return"]
    target_col = "target_next_return"

    # Build sequences
    Xtr, ytr, _, _ = build_sequences_by_ticker(train_df, feature_cols, target_col, args.window)
    Xte, yte, dates_te, tkr_te = build_sequences_by_ticker(test_df, feature_cols, target_col, args.window)
    if len(Xtr) == 0 or len(Xte) == 0:
        raise SystemExit("Not enough data to build sequences. Check your CSV or lower --window.")

    in_features = Xtr.shape[-1]
    train_loader = DataLoader(SeqDS(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SeqDS(Xte, yte), batch_size=args.batch_size, shuffle=False)

    # Train & eval 3 models
    models = {"RNN": "RNN", "LSTM": "LSTM", "GRU": "GRU"}
    preds_tables: Dict[str, pd.DataFrame] = {}
    metrics_all: Dict[str, Dict[str, float]] = {}

    for name, cell in models.items():
        model = RNNHead(in_features=in_features, cell=cell, hidden=args.hidden, layers=args.layers)
        model = train_model(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)
        metrics, preds, trues = eval_model(model, test_loader, device=device)
        metrics_all[name] = metrics

        df_out = (pd.DataFrame({
            "date": dates_te,
            "ticker": tkr_te,
            "y_true_return_t1": trues,
            "y_pred_return_t1": preds
        }).sort_values(["ticker", "date"]))
        out_path = f"prediction_{cell.lower()}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[{cell}] Test MSE = {metrics['MSE']:.8f} | Saved {out_path}")
        preds_tables[name] = df_out

    # ---------- Black–Litterman
    # Compute returns (dedup already applied)
    ret_all = df.copy()
    ret_all["ret"] = ret_all.groupby("ticker", observed=True)["adj_close"].pct_change()

    cut = pd.to_datetime("2025-01-01").date()
    ret_train = (ret_all[ret_all["date"] < cut]
                 .groupby(["date", "ticker"], as_index=False)["ret"].last())
    ret_test = (ret_all[ret_all["date"] >= cut]
                .groupby(["date", "ticker"], as_index=False)["ret"].last())

    # TRAIN covariance
    R_train = (ret_train.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
               .sort_index().fillna(0.0))

    # Asset universe = intersection(train assets, all prediction assets)
    assets_train = set(R_train.columns)
    assets_pred = None
    for P in preds_tables.values():
        a = set(P["ticker"].unique())
        assets_pred = a if assets_pred is None else (assets_pred & a)
    assets = sorted(assets_train & (assets_pred or set()))
    if not assets:
        raise SystemExit("No common assets between TRAIN data and prediction files. Check tickers or coverage.")

    R = R_train[assets].copy()
    Sigma_raw = R.cov().values
    # Shrinkage + jitter
    alpha = 0.1
    Sigma = (1 - alpha) * Sigma_raw + alpha * np.diag(np.diag(Sigma_raw))
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(len(assets))
    w_mkt = np.ones(len(assets)) / len(assets)
    LAM = 1.0
    TAU = 0.05
    K = 0.5
    pi = LAM * Sigma.dot(w_mkt)

    # TEST returns on same universe
    RET_TEST = (ret_test.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
                .sort_index().fillna(0.0))
    RET_TEST = RET_TEST[[c for c in assets if c in RET_TEST.columns]].fillna(0.0)

    def run_bl_and_bt(preds_df: pd.DataFrame, label: str) -> pd.DataFrame:
        P = preds_df.copy()
        P["date"] = pd.to_datetime(P["date"]).dt.date
        P = (P.dropna(subset=["y_pred_return_t1"])
               .groupby(["date", "ticker"], as_index=False).last())
        P = P[P["ticker"].isin(assets)].copy()
        rows = []

        for d, g in P.groupby("date"):
            tkrs = [t for t in g["ticker"].unique() if t in assets]
            if len(tkrs) == 0:
                continue
            idx = [assets.index(t) for t in tkrs]

            if len(idx) == 1:
                rows.append({"date": d, "ticker": tkrs[0], f"w_BL_{label}": 1.0})
                continue

            S_d = Sigma[np.ix_(idx, idx)]
            S_d = 0.5 * (S_d + S_d.T) + 1e-8 * np.eye(len(idx))
            pi_d = pi[idx]
            Q_d = (g.drop_duplicates("ticker")
                     .set_index("ticker")["y_pred_return_t1"]
                     .reindex(tkrs).fillna(0.0).to_numpy(float))
            Omega_inv = Omega_inv_from(S_d, TAU=TAU, K=K)
            mu_star = bl_mu(S_d, pi_d, Q_d, TAU, Omega_inv)
            w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)
            for t, w in zip(tkrs, w_bl):
                rows.append({"date": d, "ticker": t, f"w_BL_{label}": w})

        W = (pd.DataFrame(rows)
               .groupby(["date", "ticker"], as_index=False).last()
               .sort_values(["date", "ticker"]))
        W.to_csv(f"weights_{label}.csv", index=False)

        wtab = (W.pivot_table(index="date", columns="ticker",
                              values=f"w_BL_{label}", aggfunc="last")
                  .reindex(RET_TEST.index)
                  .fillna(0.0))

        common = [c for c in wtab.columns if c in RET_TEST.columns]
        port = (wtab[common] * RET_TEST[common]).sum(axis=1)

        bt = pd.DataFrame({f"ret_BL_{label}": port})
        bt[f"cum_BL_{label}"] = (1 + bt[f"ret_BL_{label}"]).cumprod()
        bt.to_csv(f"bt_{label}.csv")
        return bt

    bt_rnn = run_bl_and_bt(preds_tables["RNN"], "RNN")
    bt_lstm = run_bl_and_bt(preds_tables["LSTM"], "LSTM")
    bt_gru = run_bl_and_bt(preds_tables["GRU"], "GRU")

    # ---------- Final recap
    bt_all = pd.concat([bt_rnn, bt_lstm, bt_gru], axis=1)

    summary = pd.DataFrame({
        "Model": ["RNN", "LSTM", "GRU"],
        "MSE": [metrics_all["RNN"]["MSE"], metrics_all["LSTM"]["MSE"], metrics_all["GRU"]["MSE"]],
        "RMSE": [metrics_all["RNN"]["RMSE"], metrics_all["LSTM"]["RMSE"], metrics_all["GRU"]["RMSE"]],
        "MAE": [metrics_all["RNN"]["MAE"], metrics_all["LSTM"]["MAE"], metrics_all["GRU"]["MAE"]],
        "R2": [metrics_all["RNN"]["R2"], metrics_all["LSTM"]["R2"], metrics_all["GRU"]["R2"]],
        "MAPE(%)": [metrics_all["RNN"]["MAPE(%)"], metrics_all["LSTM"]["MAPE(%)"], metrics_all["GRU"]["MAPE(%)"]],
        "Sharpe": [sharpe(bt_all["ret_BL_RNN"]), sharpe(bt_all["ret_BL_LSTM"]), sharpe(bt_all["ret_BL_GRU"])],
        "Sortino": [sortino(bt_all["ret_BL_RNN"]), sortino(bt_all["ret_BL_LSTM"]), sortino(bt_all["ret_BL_GRU"])],
        "CAGR": [bt_all["cum_BL_RNN"].iloc[-1] - 1,
                 bt_all["cum_BL_LSTM"].iloc[-1] - 1,
                 bt_all["cum_BL_GRU"].iloc[-1] - 1],
        "MaxDD": [maxdd(bt_all["cum_BL_RNN"]),
                  maxdd(bt_all["cum_BL_LSTM"]),
                  maxdd(bt_all["cum_BL_GRU"])],
        "Days": [len(bt_all)] * 3
    }).sort_values("Sharpe", ascending=False)

    summary.to_csv("prediction_metrics_models.csv", index=False)
    print("\n=== RECAP OF ALL MODELS (saved to prediction_metrics_models.csv) ===")
    print(summary.round(6))


if __name__ == "__main__":
    main()
