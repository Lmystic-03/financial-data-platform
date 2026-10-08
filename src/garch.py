import pandas as pd
from arch import arch_model

from risk import load_returns

if __name__ == "__main__":
    spy = load_returns()["SPY"] * 100   # in percent, for a stable optimizer

    model = arch_model(spy, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    result = model.fit(disp="off")

    print(result.summary())
    params = result.params
    print("\nPersistence (alpha + beta):", round(params["alpha[1]"] + params["beta[1]"], 4))
    garch_vol = result.conditional_volatility * (252 ** 0.5)   # annualized, in %
    roll_vol = spy.rolling(20).std() * (252 ** 0.5)
    comparison = pd.DataFrame({"garch": garch_vol, "rolling_20": roll_vol}).dropna()
    print("\nCorrelation:", round(comparison.corr().iloc[0, 1], 3))
    print(comparison.describe().round(2))
    print("Peak dates:\n", comparison.idxmax())