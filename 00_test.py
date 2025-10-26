# 04_merge_prices_only.py
import pandas as pd

print(">> Merging sentiment+prices files...")

# Lis les deux fichiers
train_prices = pd.read_csv("train_sentiment_prices.csv")
test_prices  = pd.read_csv("test_sentiment_prices.csv")

# Fusionne sans nettoyage
sentiment_prices = pd.concat([train_prices, test_prices], ignore_index=True)

# Sauvegarde le fichier final
sentiment_prices.to_csv("sentiment_prices.csv", index=False)

print(f"[OK] Saved sentiment_prices.csv ({len(sentiment_prices)} rows)")