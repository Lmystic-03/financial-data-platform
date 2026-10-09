import pandas as pd

from risk import load_returns


def rolling_beta(stock: pd.Series, market: pd.Series, window: int = 250) -> pd.Series:
    """Rolling beta = Cov(stock, market) / Var(market) over the last `window` days."""
    cov = stock.rolling(window).cov(market)
    var = market.rolling(window).var()
    return cov / var


if __name__ == "__main__":
    returns = load_returns()
    market = returns["SPY"]
    stocks = [c for c in returns.columns if c != "SPY"]

    full_beta = {s: returns[s].cov(market) / market.var() for s in stocks}
    print("Full-sample beta vs SPY:")
    print(pd.Series(full_beta).round(2))

    betas = pd.DataFrame({s: rolling_beta(returns[s], market) for s in stocks}).dropna()
    print("\nRolling beta (250d): min / mean / max")
    print(betas.agg(["min", "mean", "max"]).T.round(2))