# 04_merge_prices_only.py
import pandas as pd

"""

(essayer de tourner FINBERT sur vos parties respectives svp)
Splits 'articles_merge_of_APIs.csv' into 4 equal parts:
- data_benoit.csv
- data_aymane.csv
- data_raphael.csv
- data_tom.csv

Purpose:
To distribute the articles equally among team members for manual review or labeling.

Input:
- articles_merge_of_APIs.csv

Outputs:
- data_benoit.csv
- data_aymane.csv
- data_raphael.csv
- data_tom.csv
"""

import pandas as pd

# === Load the dataset ===
df = pd.read_csv("articles_merge_of_APIs.csv")

# Shuffle if you want random distribution (optional)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Split into 4 equal parts ===
n = len(df)
part_size = n // 4

parts = [
    ("data_benoit.csv", df.iloc[:part_size]),
    ("data_aymane.csv", df.iloc[part_size:2*part_size]),
    ("data_raphael.csv", df.iloc[2*part_size:3*part_size]),
    ("data_tom.csv", df.iloc[3*part_size:]),
]

# === Save each part ===
for name, part in parts:
    part.to_csv(name, index=False)
    print(f"✅ Saved {name} ({len(part)} rows)")

# Optional sanity check
print("\nTotal rows:", sum(len(p[1]) for p in parts))
