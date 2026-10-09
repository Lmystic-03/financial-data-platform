import numpy as np
import pandas as pd

from risk import load_returns, historical_var, expected_shortfall


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio return = weighted sum of asset returns (daily rebalanced)."""
    return returns[weights.index] @ weights


def portfolio_vol(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Annualized volatility via the covariance matrix: sqrt(w' Σ w) * sqrt(252)."""
    cov = returns[weights.index].cov()
    var = float(weights.values @ cov.values @ weights.values)
    return float(np.sqrt(var) * np.sqrt(252))


if __name__ == "__main__":
    returns = load_returns()
    stocks = [c for c in returns.columns if c != "SPY"]
    weights = pd.Series(1 / len(stocks), index=stocks)   # equal weights

    indiv_vol = returns[stocks].std() * np.sqrt(252)
    no_div = float((weights * indiv_vol).sum())          # weighted avg of vols
    port = portfolio_vol(returns, weights)

    print("Individual annualized vols:")
    print((indiv_vol * 100).round(1))
    print(f"\nWeighted avg of vols (no diversification): {no_div:.2%}")
    print(f"Portfolio vol (with diversification):       {port:.2%}")
    print(f"Diversification benefit:                    {1 - port / no_div:.1%}")

    pr = portfolio_returns(returns, weights)
    print("\nRisk (daily, positive = loss):")
    for name, r in (("Portfolio", pr), ("SPY", returns["SPY"])):
        print(f"{name:10s} VaR95 {historical_var(r, 0.95):.4f}  "
        f"VaR99 {historical_var(r, 0.99):.4f}  "
            f"ES99 {expected_shortfall(r, 0.99):.4f}")