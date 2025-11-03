import pandas as pd

# === Load your file ===
df = pd.read_csv("prices_long.csv")

# === Remove rows where ticker == "SPY" ===
before = len(df)
df = df[df["ticker"].str.upper() != "SPY"]
after = len(df)

# === Save cleaned file ===
df.to_csv("prices_long.csv", index=False)

print(f"✅ Removed {before - after} rows with ticker 'SPY'")
print(f"✅ Cleaned file saved → prices_long.csv ({after} rows remaining)")
