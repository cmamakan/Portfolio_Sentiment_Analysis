# save as: merge_historical.py  (or run in a notebook/cell)
import pandas as pd
from glob import glob

OUTPUT_FILE = "articles_zpip install yfinance pandashistorical_010123_310825.csv"

# 1) Collect all monthly GDELT slices
files = sorted(glob("tmp/gdelt_*_to_*.csv"))
if not files:
    raise SystemExit("No files found in tmp/. Did the GDELT loop run?")

# 2) Concatenate
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# 3) Deduplicate
if "url" in df.columns:
    df = df.drop_duplicates(subset=["url"])
else:
    df = df.drop_duplicates(subset=["date","ticker","title"], keep="first")

# 4) Sort & save
df = df.sort_values(["date","ticker"]).reset_index(drop=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE}: {len(df)} rows")

# 5) Quick checks
print("\nCounts by ticker:")
print(df["ticker"].value_counts().sort_index())

print("\nDate range:", df["date"].min(), "→", df["date"].max())
