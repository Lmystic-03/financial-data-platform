from datetime import date, timedelta

# 8 large US stocks across sectors
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ", "PG"]

# Market benchmark (S&P 500 ETF), needed later for beta / CAPM
BENCHMARK = "SPY"

ALL_TICKERS = STOCKS + [BENCHMARK]

# Rolling window: always the last 10 years up to today
HISTORY_YEARS = 10
TODAY = date.today()
START_DATE = (TODAY - timedelta(days=365 * HISTORY_YEARS + 2)).isoformat()

# yfinance excludes the end date, so we add 1 day to include today
END_DATE = (TODAY + timedelta(days=1)).isoformat()