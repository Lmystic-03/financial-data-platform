from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
SILVER_DIR = Path("data/silver")


def build_silver(ticker: str) -> Path:
    """Read raw prices, clean them, add features, save to silver."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RAW_DIR / f"{ticker}.parquet")

    # Cleaning
    df = df.sort_values("Date").drop_duplicates(subset="Date").dropna()

    # Features
    df["ticker"] = ticker
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

    output_path = SILVER_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"{ticker}: {len(df)} rows saved to {output_path}")
    return output_path


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM"]
    for ticker in TICKERS:
        build_silver(ticker)