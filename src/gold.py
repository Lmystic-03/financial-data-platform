from pathlib import Path

import numpy as np
import pandas as pd

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM"]


def build_gold() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stack all silver tables into one
    frames = [pd.read_parquet(SILVER_DIR / f"{t}.parquet") for t in TICKERS]
    prices = pd.concat(frames, ignore_index=True)
    prices.to_parquet(GOLD_DIR / "prices_all.parquet", index=False)

    # 2. Returns matrix: one row per date, one column per ticker
    returns = prices.pivot(index="Date", columns="ticker", values="log_return")
    returns = returns.dropna()

    # 3. Correlation matrix
    corr = returns.corr()
    corr.to_parquet(GOLD_DIR / "correlation.parquet")

    # 4. Summary per stock
    summary = returns.agg(["mean", "std"]).T
    summary["annual_return"] = summary["mean"] * 252
    summary["annual_vol"] = summary["std"] * np.sqrt(252)
    summary["return_to_vol"] = summary["annual_return"] / summary["annual_vol"]
    summary = summary[["annual_return", "annual_vol", "return_to_vol"]].round(4)
    summary.to_parquet(GOLD_DIR / "summary.parquet")

    print("Correlation matrix:")
    print(corr.round(3))
    print()
    print("Summary:")
    print(summary)


if __name__ == "__main__":
    build_gold()