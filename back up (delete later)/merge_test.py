import pandas as pd

files = [
    "articles_AAPL_MSFT_NVDA.csv",
    "articles_AMZN_META_TSLA_0609_2009.csv",
    "articles_AMZN_META_TSLA_2109_0410.csv",
    "articles_JPM_V_JNJ_GOOGL_0609_2009.csv",
    "articles_JPM_V_JNJ_GOOGL_2109_0410.csv",
    "articles_V_JPM_gdelt_0609_0410.csv",   # complément GDELT-only
]

dfs = [pd.read_csv(f) for f in files]
test = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["url"])
test = test.sort_values(["date","ticker"]).reset_index(drop=True)
test.to_csv("articles_zfull.csv", index=False)

df = pd.read_csv("articles_zfull.csv")
print(df['ticker'].value_counts().sort_index())
print(df.groupby(['ticker','provider']).size())
print(df['date'].min(), "→", df['date'].max())
