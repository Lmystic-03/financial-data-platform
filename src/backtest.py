from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2, t 

from risk import historical_var, parametric_var, load_returns


def count_exceptions(returns: pd.Series, var_func, level: float, window: int = 250):
    """Slide a window over the series. For each day t, compute VaR using only
    the `window` days BEFORE t, then check if the return on day t is a loss
    bigger than that VaR (an exception)."""
    exceptions = []
    for t in range(window, len(returns)):
        past = returns.iloc[t - window : t]   # days t-window ... t-1 (excludes t)
        var = var_func(past, level)
        exceptions.append(returns.iloc[t] < -var)
    return pd.Series(exceptions, index=returns.index[window:])


def kupiec_test(n_exceptions: int, n_obs: int, level: float):
    """Kupiec POF test. H0: the true exception rate equals 1 - level.
    Returns (LR statistic, p-value)."""
    p = 1 - level                 # expected exception rate
    x, n = n_exceptions, n_obs
    phat = x / n                  # observed exception rate
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ll_h0 = (n - x) * np.log(1 - p) + x * np.log(p)
        ll_h1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
        lr = -2 * (ll_h0 - ll_h1)
    p_value = 1 - chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def garch_var(past: pd.Series, level: float = 0.95, dist: str = "t") -> float:
    """VaR from a zero-mean GARCH(1,1) with Student-t shocks."""
    r = past.dropna() * 100
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    f = res.forecast(horizon=1)
    sigma = float(np.sqrt(f.variance.iloc[-1, 0]))
    nu = float(res.params["nu"])                       # degrees of freedom
    q = t.ppf(1 - level, nu) * np.sqrt((nu - 2) / nu)  # standardized t quantile
    return float(-(q * sigma) / 100)        # back to decimals


if __name__ == "__main__":
    spy = load_returns()["SPY"]
    rows = []
    for level in (0.95, 0.99):
        for name, func in (("historical", historical_var), ("parametric", parametric_var), ("garch_t", garch_var)):
            exc = count_exceptions(spy, func, level, window=1000)
            lr, pval = kupiec_test(int(exc.sum()), len(exc), level)
            rows.append({
                "level": level,
                "method": name,
                "n_obs": len(exc),
                "exceptions": int(exc.sum()),
                "expected": round((1 - level) * len(exc), 1),
                "rate": round(exc.mean(), 4),
                "kupiec_lr": round(lr, 2),
                "p_value": round(pval, 4),
            })
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows))