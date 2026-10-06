
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")

for file_path in sorted(RAW_DIR.glob("*.parquet")):
    df = pd.read_parquet(file_path)

    prices = df[["Open", "High", "Low", "Close"]]

    invalid_dates = pd.to_datetime(
        df["Date"], errors="coerce"
    ).isna().sum()

    invalid_prices = (prices <= 0).any(axis=1).sum()

    ohlc_violations = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    ).sum()

    print(f"\n--- {file_path.stem} ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Duplicate dates: {df['Date'].duplicated().sum()}")
    print(f"Invalid dates: {invalid_dates}")
    print(f"Rows with non-positive prices: {invalid_prices}")
    print(f"Negative volumes: {(df['Volume'] < 0).sum()}")
    print(f"OHLC violations: {ohlc_violations}")