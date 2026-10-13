from pathlib import Path

import pandas as pd
from arch import arch_model

from risk import load_returns

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "dashboard_data" / "garch_volatility.parquet"


def fit_garch_volatility(spy_returns: pd.Series) -> tuple[pd.DataFrame, object]:
    """Fit GARCH(1,1) and return annualized conditional and 20-day rolling volatility."""
    spy = spy_returns.dropna() * 100  # percent, for a stable optimizer
    if spy.empty:
        raise ValueError("SPY returns must contain at least one non-null observation")

    model = arch_model(spy, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    result = model.fit(disp="off")

    garch_vol = result.conditional_volatility * (252 ** 0.5)
    roll_vol = spy.rolling(20).std() * (252 ** 0.5)
    comparison = pd.DataFrame({"garch": garch_vol, "rolling_20": roll_vol}).dropna()
    comparison.index.name = "Date"
    return comparison, result


def save_garch_volatility(comparison: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Save annualized volatility series for the dashboard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(path)
    return path


if __name__ == "__main__":
    comparison, result = fit_garch_volatility(load_returns()["SPY"])
    print(result.summary())
    params = result.params
    print("\nPersistence (alpha + beta):", round(params["alpha[1]"] + params["beta[1]"], 4))
    print("\nCorrelation:", round(comparison.corr().iloc[0, 1], 3))
    print(comparison.describe().round(2))
    print("Peak dates:\n", comparison.idxmax())
    print("\nSaved dashboard volatility data to:", save_garch_volatility(comparison))