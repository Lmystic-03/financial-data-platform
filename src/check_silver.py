import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_parquet("data/silver/AAPL.parquet")

print(df.tail(5))
print()
print(df[["log_return", "ma_20", "volatility_20"]].isna().sum())
print()
print(df["log_return"].describe())