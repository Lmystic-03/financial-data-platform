from pathlib import Path

import yfinance as yf

RAW_DIR = Path("data/raw")


def ingest_prices(ticker: str, period: str = "1y") -> Path:
    """Download daily prices for one ticker and save them as Parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)  # flatten columns
    df = df.reset_index()  # Date becomes a normal column

    output_path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"{ticker}: {len(df)} rows saved to {output_path}")
    return output_path


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM"]
    for ticker in TICKERS:
        ingest_prices(ticker)