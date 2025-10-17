# lstm_price_pred.py
# pip install torch pandas numpy scikit-learn tqdm

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 0) PARAMÈTRES
# -------------------------------
WINDOW = 10            # longueur de la séquence (jours)
BATCH_SIZE = 64
HIDDEN = 32
LAYERS = 1
EPOCHS = 20
LR = 1e-3
SEED = 42

# Chemins (adapte si besoin)
PRICES_CSV = "prices_long.csv"
DAILY_TRAIN_CSV = "daily_sentiment_2023_2024_train.csv"
DAILY_TEST_CSV  = "daily_sentiment_2025_test.csv"

# -------------------------------
# 1) UTILS
# -------------------------------
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_features(prices_df, daily_df):
    """
    Fusionne prix & sentiment par (date, ticker), crée return et features standard.
    Colonnes clés en sortie:
      date, ticker, adj_close, volume, sentiment_mean, return, log_volume, target_next_return
    """
    df = prices_df.merge(daily_df, on=["date","ticker"], how="left")
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # Rendement simple
    df["return"] = df.groupby("ticker")["adj_close"].pct_change()

    # Sentiment manquant -> 0
    if "sentiment_mean" not in df.columns:
        df["sentiment_mean"] = np.nan
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)

    # Volume en log
    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))

    # Cible = return_{t+1}
    df["target_next_return"] = df.groupby("ticker")["return"].shift(-1)

    # Jours de bourse
    df = df.dropna(subset=["adj_close"])
    return df

def build_sequences_by_ticker(df, feature_cols, target_col, window=WINDOW):
    """
    Construit des séquences glissantes par ticker.
    Retour: X (N, window, F), y (N,1), dates (N,), tickers (N,)
    """
    X_list, y_list, dates, tickers = [], [], [], []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feature_cols].values.astype(np.float32)
        tgt  = g[target_col].values.astype(np.float32)

        for i in range(len(g) - window):
            target_idx = i + window
            if target_idx >= len(g): break
            y_i = tgt[target_idx]
            if np.isnan(y_i): 
                continue
            X_list.append(vals[i:target_idx])
            y_list.append([y_i])
            dates.append(g.loc[target_idx, "date"])
            tickers.append(tkr)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, np.array(dates), np.array(tickers)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)      # (N, T, F)
        self.y = torch.from_numpy(y)      # (N, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, in_features, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features, hidden_size=hidden,
            num_layers=layers, batch_first=True
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)      # out: (B, T, H)
        last = out[:, -1, :]       # (B, H)
        return self.head(last)     # (B, 1)

def train_model(model, loader, optimizer, criterion, epochs=EPOCHS, device="cpu"):
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {ep:02d}] train MSE: {np.mean(losses):.6f}")
    return model

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yhat = model(xb).cpu().numpy()
            preds.append(yhat)
            trues.append(yb.numpy())
    preds = np.vstack(preds).ravel()
    trues = np.vstack(trues).ravel()
    mse = float(np.mean((preds - trues)**2))
    return mse, preds, trues

# -------------------------------
# 2) CHARGEMENT & PRÉPA DATA
# -------------------------------
def main():
    set_seed(SEED)

    # -- Prix
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    prices["date"] = prices["date"].dt.date
    prices["ticker"] = prices["ticker"].str.upper()

    # -- Sentiments train/test
    daily_train = pd.read_csv(DAILY_TRAIN_CSV, parse_dates=["date"])
    daily_test  = pd.read_csv(DAILY_TEST_CSV,  parse_dates=["date"])
    for d in (daily_train, daily_test):
        d["date"] = d["date"].dt.date
        d["ticker"] = d["ticker"].str.upper()

    # Fenêtres
    dmin_tr, dmax_tr = daily_train["date"].min(), daily_train["date"].max()
    dmin_te, dmax_te = daily_test["date"].min(),  daily_test["date"].max()
    print(f"Train window: {dmin_tr} → {dmax_tr}")
    print(f"Test  window: {dmin_te} → {dmax_te}")

    # -- Fusion prix + sentiment
    train_df = make_features(
        prices[(prices["date"]>=dmin_tr) & (prices["date"]<=dmax_tr)],
        daily_train
    )
    test_df  = make_features(
        prices[(prices["date"]>=dmin_te) & (prices["date"]<=dmax_te)],
        daily_test
    )

    # Features / cible
    feature_cols = ["sentiment_mean", "adj_close", "log_volume", "return"]
    target_col   = "target_next_return"

    # -- Standardisation (sauf sentiment déjà borné)
    scaler = StandardScaler()
    fit_mat = train_df[["adj_close","log_volume","return"]].fillna(0.0).values
    scaler.fit(fit_mat)

    def apply_scale(df):
        df = df.copy()
        df["return"] = df["return"].fillna(0.0)
        mat = df[["adj_close","log_volume","return"]].values
        df[["adj_close","log_volume","return"]] = scaler.transform(mat)
        return df

    train_df = apply_scale(train_df)
    test_df  = apply_scale(test_df)

    # -- Séquences
    Xtr, ytr, _, _ = build_sequences_by_ticker(train_df, feature_cols, target_col, WINDOW)
    Xte, yte, dates_te, tkr_te = build_sequences_by_ticker(test_df,  feature_cols, target_col, WINDOW)

    print(f"Train sequences: {Xtr.shape}, Test sequences: {Xte.shape}")

    # -- DataLoaders
    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SeqDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------------
    # 3) MODÈLE & TRAIN
    # -------------------------------
    in_features = Xtr.shape[-1]
    model = LSTMModel(in_features=in_features, hidden=HIDDEN, layers=LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model = train_model(model, train_loader, optimizer, criterion, epochs=EPOCHS, device="cpu")

    # -------------------------------
    # 4) ÉVAL TEST & EXPORT
    # -------------------------------
    mse, preds, trues = evaluate(model, test_loader, device="cpu")
    print(f"\n[Test] MSE (return_{{t+1}}) = {mse:.8f}")

    # ✅ NE PAS caster la date en string (on garde des dates natives)
    out = pd.DataFrame({
        "date": dates_te,
        "ticker": tkr_te,
        "y_true_return_t1": trues,
        "y_pred_return_t1": preds,
    }).sort_values(["ticker","date"]).reset_index(drop=True)

    out.to_csv("lstm_preds_test.csv", index=False)
    print("Saved: lstm_preds_test.csv")

if __name__ == "__main__":
    main()
