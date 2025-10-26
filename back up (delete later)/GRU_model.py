# pip install torch pandas numpy scikit-learn tqdm

import os
import math
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
    Retourne un DataFrame trié par (ticker, date) avec colonnes :
    date, ticker, adj_close, volume, sentiment_mean, return, (feat_cols...)
    """
    df = prices_df.merge(daily_df, on=["date","ticker"], how="left")
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)

    # Retour log ou simple ; on prend simple ici
    df["return"] = df.groupby("ticker")["adj_close"].pct_change()

    # Sentiment manquant -> 0 (aucune news)
    if "sentiment_mean" not in df.columns:
        df["sentiment_mean"] = np.nan
    df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)

    # Volume : log(1+vol) pour stabiliser
    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))

    # On gardera comme features : sentiment, prix (adj_close), log_volume, return (t)
    # La cible sera return_{t+1} (shift -1)
    df["target_next_return"] = df.groupby("ticker")["return"].shift(-1)

    # On garde que les lignes où on a adj_close et return disponibles
    df = df.dropna(subset=["adj_close"])  # trading days
    # On laissera la cible NaN tomber plus tard lors de la création des séquences

    return df

def build_sequences_by_ticker(df, feature_cols, target_col, window=WINDOW):
    """
    Construit des séquences glissantes par ticker.
    Retourne X (N, window, F), y (N,), et des meta (date, ticker) alignées aux y.
    """
    X_list, y_list, dates, tickers = [], [], [], []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feature_cols].values
        tgt  = g[target_col].values

        for i in range(len(g) - window):
            y_i = tgt[i + window - 1 + 1 - 0]  # cible = return_{t+1} à la fin de la fenêtre
            # (i -> i+window-1) = fenêtre ; target au lendemain (i+window)
            if i + window >= len(g):
                break
            y_i = tgt[i + window]  # plus explicite
            if np.isnan(y_i):
                continue
            X_list.append(vals[i:i+window])
            y_list.append(y_i)
            dates.append(g.loc[i+window, "date"])
            tickers.append(tkr)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y, np.array(dates), np.array(tickers)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)      # (N, T, F)
        self.y = torch.from_numpy(y)      # (N, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class GRUModel(nn.Module):
    def __init__(self, in_features, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden,
                          num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.gru(x)          # (B, T, H)
        last = out[:, -1, :]          # (B, H)
        yhat = self.head(last)        # (B, 1)
        return yhat

def train_model(model, loader, optimizer, criterion, epochs=EPOCHS, val_loader=None, device="cpu"):
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
        msg = f"[Epoch {ep:02d}] train MSE: {np.mean(losses):.6f}"
        if val_loader is not None:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vlosses.append(criterion(model(xb), yb).item())
            msg += f" | val MSE: {np.mean(vlosses):.6f}"
        print(msg)
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

    # -- Lire les prix
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    prices["date"] = prices["date"].dt.date
    prices["ticker"] = prices["ticker"].str.upper()

    # -- Lire sentiments train/test (déjà agrégés daily)
    daily_train = pd.read_csv(DAILY_TRAIN_CSV, parse_dates=["date"])
    daily_test  = pd.read_csv(DAILY_TEST_CSV,  parse_dates=["date"])
    for d in (daily_train, daily_test):
        d["date"] = d["date"].dt.date
        d["ticker"] = d["ticker"].str.upper()

    # Fenêtres temporelles effectives
    dmin_tr, dmax_tr = daily_train["date"].min(), daily_train["date"].max()
    dmin_te, dmax_te = daily_test["date"].min(),  daily_test["date"].max()
    print(f"Train window: {dmin_tr} → {dmax_tr}")
    print(f"Test  window: {dmin_te} → {dmax_te}")

    # -- Fusion prix + sentiment (séparément train/test)
    train_df = make_features(
        prices[(prices["date"]>=dmin_tr) & (prices["date"]<=dmax_tr)],
        daily_train
    )
    test_df  = make_features(
        prices[(prices["date"]>=dmin_te) & (prices["date"]<=dmax_te)],
        daily_test
    )

    # -- Sélection des features
    feature_cols = ["sentiment_mean", "adj_close", "log_volume", "return"]
    target_col   = "target_next_return"

    # -- Normalisation par TICKER (plus propre)
    Xscalers, yscaler = {}, None  # on normalise X par feature globalement (sauf sentiment déjà borné)
    # Concat train pour fitter les scalers (sauf sentiment)
    fit_df = train_df.copy()
    fit_df["adj_close"]  = fit_df["adj_close"].astype(float)
    fit_df["log_volume"] = fit_df["log_volume"].astype(float)
    fit_df["return"]     = fit_df["return"].fillna(0.0).astype(float)
    # standardiser adj_close, log_volume, return (pas sentiment)
    X_scaler = StandardScaler()
    fit_mat = fit_df[["adj_close","log_volume","return"]].values
    X_scaler.fit(fit_mat)

    def apply_scale(df):
        df = df.copy()
        df["return"] = df["return"].fillna(0.0)
        mat = df[["adj_close","log_volume","return"]].values
        mat_scaled = X_scaler.transform(mat)
        df[["adj_close","log_volume","return"]] = mat_scaled
        return df

    train_df = apply_scale(train_df)
    test_df  = apply_scale(test_df)

    # -- Construire séquences
    Xtr, ytr, dates_tr, tkr_tr = build_sequences_by_ticker(train_df, feature_cols, target_col, WINDOW)
    Xte, yte, dates_te, tkr_te = build_sequences_by_ticker(test_df,  feature_cols, target_col, WINDOW)

    print(f"Train sequences: {Xtr.shape}, Test sequences: {Xte.shape}")

    # -- DataLoaders
    train_ds = SeqDataset(Xtr, ytr)
    test_ds  = SeqDataset(Xte, yte)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------------
    # 3) MODÈLE & TRAIN
    # -------------------------------
    in_features = Xtr.shape[-1]
    model = GRUModel(in_features=in_features, hidden=HIDDEN, layers=LAYERS)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model = train_model(model, train_loader, optim, criterion, epochs=EPOCHS, val_loader=None, device="cpu")

    # -------------------------------
    # 4) ÉVAL TEST & EXPORT
    # -------------------------------
    mse, preds, trues = evaluate(model, test_loader, device="cpu")
    print(f"\n[Test] MSE (return_{'{'}t+1{'}'}) = {mse:.8f}")

    # Sauvegarder prédictions alignées (date, ticker, y_true, y_pred)
    out = pd.DataFrame({
        "date": dates_te.astype(str),
        "ticker": tkr_te,
        "y_true_return_t1": trues,
        "y_pred_return_t1": preds,
    })
    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    out.to_csv("gru_preds_test.csv", index=False)
    print("Saved: gru_preds_test.csv")

if __name__ == "__main__":
    main()
