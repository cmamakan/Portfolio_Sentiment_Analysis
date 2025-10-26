import pandas as pd

# Charger tes deux fichiers actuels
train_old = pd.read_csv("daily_sentiment_train.csv", parse_dates=["date"])
test_old  = pd.read_csv("daily_sentiment_test.csv", parse_dates=["date"])

# Fusionner les deux sans doublons
df_full = pd.concat([train_old, test_old], ignore_index=True)
df_full = df_full.drop_duplicates(subset=["date","ticker"]).sort_values(["date","ticker"])

print("Full data:", df_full["date"].min().date(), "→", df_full["date"].max().date(), "| rows:", len(df_full))

# Découper à nouveau proprement
train = df_full[df_full["date"] < "2025-01-01"]
test  = df_full[(df_full["date"] >= "2025-01-01") & (df_full["date"] <= "2025-10-05")]

print("Train:", train["date"].min().date(), "→", train["date"].max().date(), "| rows:", len(train))
print("Test :", test["date"].min().date(), "→", test["date"].max().date(), "| rows:", len(test))

# Sauvegarder avec nouveaux noms
train.to_csv("daily_sentiment_2023_2024_train.csv", index=False)
test.to_csv("daily_sentiment_2025_test.csv", index=False)
print("✅ Fichiers créés sans écraser les anciens.")

