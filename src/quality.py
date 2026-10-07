from pathlib import Path

import pandas as pd

from config import ALL_TICKERS

SILVER_DIR = Path("data/silver")
MAX_ABS_RETURN = 0.25  # a daily move above 25% is suspicious


def check_ticker(df: pd.DataFrame) -> dict:
    """Run basic quality checks on one silver table."""
    return {
        "rows": len(df),
        "first_date": df["Date"].min().date(),
        "last_date": df["Date"].max().date(),
        "duplicate_dates": int(df["Date"].duplicated().sum()),
        "non_positive_prices": int((df["Close"] <= 0).sum()),
        "extreme_returns": int((df["log_return"].abs() > MAX_ABS_RETURN).sum()),
        "max_gap_days": int(df["Date"].diff().dt.days.max()),
    }


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    results = {}
    for ticker in ALL_TICKERS:
        df = pd.read_parquet(SILVER_DIR / f"{ticker}.parquet")
        results[ticker] = check_ticker(df)

    print(pd.DataFrame(results).T)