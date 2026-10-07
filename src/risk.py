from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

GOLD_DIR = Path("data/gold")


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR: loss exceeded on only (1 - level) of days. Positive number."""
    return float(-np.quantile(returns.dropna(), 1 - level))


def parametric_var(returns: pd.Series, level: float = 0.95) -> float:
    """Normal VaR: -(mean + z * std), where z is the normal quantile."""
    r = returns.dropna()
    z = NormalDist().inv_cdf(1 - level)  # negative number, e.g. -1.645
    return float(-(r.mean() + z * r.std()))


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss on the days when the loss exceeds the historical VaR."""
    r = returns.dropna()
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def load_returns() -> pd.DataFrame:
    """One column of daily log returns per ticker, from the gold prices table."""
    prices = pd.read_parquet(GOLD_DIR / "prices_all.parquet")
    returns = prices.pivot(index="Date", columns="ticker", values="log_return")
    return returns.dropna()


def build_risk_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for ticker in returns.columns:
        r = returns[ticker]
        rows[ticker] = {
            "hist_var_95": historical_var(r, 0.95),
            "param_var_95": parametric_var(r, 0.95),
            "es_95": expected_shortfall(r, 0.95),
            "hist_var_99": historical_var(r, 0.99),
            "param_var_99": parametric_var(r, 0.99),
            "es_99": expected_shortfall(r, 0.99),
        }
    return pd.DataFrame(rows).T.round(4)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    table = build_risk_table(load_returns())
    table.to_parquet(GOLD_DIR / "risk.parquet")
    print(table)