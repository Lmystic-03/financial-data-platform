from pathlib import Path

import yfinance as yf

from config import ALL_TICKERS, END_DATE, START_DATE

RAW_DIR = Path("data/raw")


def ingest_prices(ticker: str, start: str = START_DATE, end: str = END_DATE) -> Path:
    """Download daily prices for one ticker and save them as Parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)  # flatten columns
    df = df.reset_index()  # Date becomes a normal column

    output_path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"{ticker}: {len(df)} rows saved to {output_path}")
    return output_path


if __name__ == "__main__":
    for ticker in ALL_TICKERS:
        ingest_prices(ticker)