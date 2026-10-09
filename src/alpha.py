import pandas as pd
import statsmodels.api as sm

from risk import load_returns


def capm_regression(stock: pd.Series, market: pd.Series) -> dict:
    """OLS of stock returns on market returns: r_s = alpha + beta * r_m + e.
    Returns daily alpha, annualized alpha, beta, their p-values and R²."""
    X = sm.add_constant(market)            # adds the intercept column
    res = sm.OLS(stock, X).fit()
    alpha = res.params["const"]
    return {
        "alpha_daily_%": alpha * 100,
        "alpha_annual_%": alpha * 252 * 100,
        "alpha_pvalue": res.pvalues["const"],
        "beta": res.params[market.name],
        "beta_pvalue": res.pvalues[market.name],
        "r_squared": res.rsquared,
    }


if __name__ == "__main__":
    returns = load_returns()
    market = returns["SPY"]
    stocks = [c for c in returns.columns if c != "SPY"]

    table = pd.DataFrame({s: capm_regression(returns[s], market) for s in stocks}).T
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(table.round(4))