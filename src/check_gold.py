import pandas as pd

prices = pd.read_parquet("data/gold/prices_all.parquet")

print(prices.groupby("ticker")["Date"].agg(["min", "max", "count"]))
print()

returns = prices.pivot(index="Date", columns="ticker", values="log_return").dropna()
print(returns.shape)
print()
print(returns.tail(5).round(4))