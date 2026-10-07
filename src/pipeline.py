from config import ALL_TICKERS
from gold import build_gold
from ingest import ingest_prices
from transform import build_silver


def run_pipeline() -> None:
    print("=== Bronze: ingestion ===")
    for ticker in ALL_TICKERS:
        ingest_prices(ticker)

    print("=== Silver: cleaning and features ===")
    for ticker in ALL_TICKERS:
        build_silver(ticker)

    print("=== Gold: correlation and summary ===")
    build_gold()

    print("Pipeline finished.")


if __name__ == "__main__":
    run_pipeline()