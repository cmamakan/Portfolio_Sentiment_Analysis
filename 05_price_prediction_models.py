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

import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ====== PARAMS ======
WINDOW = 10
BATCH_SIZE = 64
HIDDEN = 32
LAYERS = 1
EPOCHS = 20
LR = 1e-3
SEED = 42
DATA_CSV = "sentiment_prices.csv"
LAM = 1.0
TAU = 0.05
K   = 0.5

torch.manual_seed(SEED); np.random.seed(SEED)

# ====== DATA UTILS ======
def build_sequences_by_ticker(df, feature_cols, target_col, window=WINDOW):
    X_list, y_list, dates, tickers = [], [], [], []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feature_cols].values.astype(np.float32)
        tgt  = g[target_col].values.astype(np.float32)
        for i in range(len(g) - window):
            if i + window >= len(g): break
            y_i = tgt[i + window]
            if np.isnan(y_i): continue
            X_list.append(vals[i:i+window])
            y_list.append([y_i])
            dates.append(g.loc[i+window, "date"])
            tickers.append(tkr)
    return np.array(X_list), np.array(y_list), np.array(dates), np.array(tickers)

class SeqDS(Dataset):
    def __init__(self, X,y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i]

# ====== MODELS ======
class RNNHead(nn.Module):
    def __init__(self, in_features, cell="GRU", hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        if cell=="RNN":   self.rnn = nn.RNN(in_features, hidden, layers, batch_first=True)
        elif cell=="LSTM":self.rnn = nn.LSTM(in_features, hidden, layers, batch_first=True)
        else:             self.rnn = nn.GRU(in_features, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.cell = cell
    def forward(self, x):
        out = self.rnn(x)[0]
        return self.head(out[:,-1,:])

def train_model(model, loader, epochs=EPOCHS, lr=LR, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); yhat=model(xb); loss=crit(yhat,yb); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        print(f"[{model.cell}][Epoch {ep:02d}] train MSE: {np.mean(losses):.6f}")
    return model

def eval_model(model, loader, device="cpu"):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in loader:
            yhat = model(xb.to(device)).cpu().numpy()
            preds.append(yhat); trues.append(yb.numpy())
    preds = np.vstack(preds).ravel(); trues = np.vstack(trues).ravel()
    mse = float(mean_squared_error(trues, preds))
    mae = float(mean_absolute_error(trues, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(trues, preds))
    mape = float(np.mean(np.abs((trues - preds) / np.clip(np.abs(trues), 1e-8, None))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape}, preds, trues

# ====== BL FUNCTIONS ======
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

def Omega_inv_from(S):
    diag = K*np.diag(TAU*S) + 1e-8
    return np.diag(1.0/diag)

# ====== LOAD & PREP ======
df = pd.read_csv(DATA_CSV, parse_dates=["date"])
df["date"] = df["date"].dt.date
df["ticker"] = df["ticker"].str.upper()

df = df.sort_values(["ticker","date"])
df["return"] = df.groupby("ticker", observed=True)["adj_close"].pct_change()
df["target_next_return"] = df.groupby("ticker", observed=True)["return"].shift(-1)
df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)

train_df = df[(df["date"] >= pd.to_datetime("2023-01-01").date()) & (df["date"] <= pd.to_datetime("2024-12-31").date())]
test_df  = df[df["date"] >= pd.to_datetime("2025-01-01").date()]

scaler = StandardScaler()
fit_mat = train_df[["adj_close","log_volume","return"]].fillna(0.0).values
scaler.fit(fit_mat)
def apply_scale(df):
    df=df.copy(); df["return"]=df["return"].fillna(0.0)
    mat = df[["adj_close","log_volume","return"]].values
    df[["adj_close","log_volume","return"]] = scaler.transform(mat)
    return df
train_df = apply_scale(train_df); test_df = apply_scale(test_df)

feature_cols = ["sentiment_mean","adj_close","log_volume","return"]
target_col   = "target_next_return"

Xtr,ytr,_,_ = build_sequences_by_ticker(train_df, feature_cols, target_col, WINDOW)
Xte,yte,dates_te,tkr_te = build_sequences_by_ticker(test_df,  feature_cols, target_col, WINDOW)
train_loader = DataLoader(SeqDS(Xtr,ytr), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SeqDS(Xte,yte), batch_size=BATCH_SIZE, shuffle=False)
in_features = Xtr.shape[-1]

print(f"Train sequences: {Xtr.shape}, Test sequences: {Xte.shape}")

# ====== TRAIN/EVAL 3 MODELS ======
models = {"RNN":"RNN", "LSTM":"LSTM", "GRU":"GRU"}
preds_tables = {}
metrics_all = {}

for name, cell in models.items():
    model = RNNHead(in_features=in_features, cell=cell, hidden=HIDDEN, layers=LAYERS)
    model = train_model(model, train_loader)
    metrics, preds, trues = eval_model(model, test_loader)
    metrics_all[name] = metrics
    df_out = pd.DataFrame({
        "date": dates_te,
        "ticker": tkr_te,
        "y_true_return_t1": trues,
        "y_pred_return_t1": preds
    }).sort_values(["ticker","date"])
    out_path = f"prediction_{cell.lower()}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[{cell}] Test MSE = {metrics['MSE']:.8f} | Saved {out_path}")
    preds_tables[name] = df_out

# ====== BLACK–LITTERMAN ======
ret_all = df.sort_values(["ticker","date"]).assign(
    ret=lambda d: d.groupby("ticker", observed=True)["adj_close"].pct_change()
)
ret_train = (
    ret_all[ret_all["date"] < pd.to_datetime("2025-01-01").date()]
    .groupby(["date","ticker"], as_index=False)["ret"].last()
)
ret_test = (
    ret_all[ret_all["date"] >= pd.to_datetime("2025-01-01").date()]
    .groupby(["date","ticker"], as_index=False)["ret"].last()
)
R = ret_train.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last").fillna(0.0)
assets = list(R.columns); N=len(assets)
Sigma = R.cov().values
w_mkt = np.ones(N)/N
pi = LAM * Sigma.dot(w_mkt)

def run_bl_and_bt(preds_df, label):
    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"]).dt.date
    rows = []
    for d, g in preds_df.groupby("date"):
        tkrs = list(g["ticker"].unique())
        idx = [assets.index(t) for t in tkrs if t in assets]
        tkrs = [assets[i] for i in idx]
        if len(idx) == 0: continue
        S_d  = Sigma[np.ix_(idx, idx)]
        pi_d = pi[idx]
        Q_d = (g.drop_duplicates("ticker")
                 .set_index("ticker")["y_pred_return_t1"]
                 .reindex(tkrs).fillna(0.0).values)
        Omega_inv = Omega_inv_from(S_d)
        mu_star = bl_mu(S_d, pi_d, Q_d, TAU, Omega_inv)
        w_bl = mv_weights_from_mu(S_d, mu_star, long_only=True)
        for t, w in zip(tkrs, w_bl):
            rows.append({"date": d, "ticker": t, f"w_BL_{label}": w})

    W = pd.DataFrame(rows).groupby(["date","ticker"], as_index=False).last()
    W.to_csv(f"weights_{label}.csv", index=False)

    ret_test_pivot = (
        ret_test.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
        .fillna(0.0)
    )
    wtab = (
        W.pivot_table(index="date", columns="ticker", values=f"w_BL_{label}", aggfunc="last")
        .reindex(ret_test_pivot.index)
        .fillna(0.0)
    )

    common = [c for c in wtab.columns if c in ret_test_pivot.columns]
    port = (wtab[common] * ret_test_pivot[common]).sum(axis=1)
    bt = pd.DataFrame({f"ret_BL_{label}": port})
    bt[f"cum_BL_{label}"] = (1 + bt[f"ret_BL_{label}"]).cumprod()
    bt.to_csv(f"bt_{label}.csv")
    return bt

bt_rnn  = run_bl_and_bt(preds_tables["RNN"],  "RNN")
bt_lstm = run_bl_and_bt(preds_tables["LSTM"], "LSTM")
bt_gru  = run_bl_and_bt(preds_tables["GRU"],  "GRU")

# ====== METRICS & SUMMARY ======
def sharpe(r):  mu, sd = r.mean(), r.std(ddof=1); return np.nan if sd==0 else (mu/sd)*np.sqrt(252)
def sortino(r): d=r[r<0]; sd=d.std(ddof=1); return np.nan if sd==0 else (r.mean()/sd)*np.sqrt(252)
def maxdd(cum): peak=cum.cummax(); dd=cum/peak-1.0; return dd.min()

bt = pd.concat([bt_rnn, bt_lstm, bt_gru], axis=1)
summary = pd.DataFrame({
    "Model": ["RNN","LSTM","GRU"],
    "MSE": [metrics_all["RNN"]["MSE"], metrics_all["LSTM"]["MSE"], metrics_all["GRU"]["MSE"]],
    "RMSE": [metrics_all["RNN"]["RMSE"], metrics_all["LSTM"]["RMSE"], metrics_all["GRU"]["RMSE"]],
    "MAE": [metrics_all["RNN"]["MAE"], metrics_all["LSTM"]["MAE"], metrics_all["GRU"]["MAE"]],
    "R2": [metrics_all["RNN"]["R2"], metrics_all["LSTM"]["R2"], metrics_all["GRU"]["R2"]],
    "MAPE(%)": [metrics_all["RNN"]["MAPE(%)"], metrics_all["LSTM"]["MAPE(%)"], metrics_all["GRU"]["MAPE(%)"]],
    "Sharpe": [sharpe(bt["ret_BL_RNN"]), sharpe(bt["ret_BL_LSTM"]), sharpe(bt["ret_BL_GRU"])],
    "Sortino": [sortino(bt["ret_BL_RNN"]), sortino(bt["ret_BL_LSTM"]), sortino(bt["ret_BL_GRU"])],
    "CAGR": [bt["cum_BL_RNN"].iloc[-1]-1, bt["cum_BL_LSTM"].iloc[-1]-1, bt["cum_BL_GRU"].iloc[-1]-1],
    "MaxDD": [maxdd(bt["cum_BL_RNN"]), maxdd(bt["cum_BL_LSTM"]), maxdd(bt["cum_BL_GRU"])],
    "Days": [len(bt)]*3
}).sort_values("Sharpe", ascending=False)

summary.to_csv("recap_metrics_models.csv", index=False)
print("\n=== RECAP OF ALL MODELS (saved to recap_metrics_models.csv) ===")
print(summary.round(6))
