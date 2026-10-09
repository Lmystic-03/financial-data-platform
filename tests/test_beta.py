import numpy as np
import pandas as pd
import pytest

from beta import rolling_beta


def test_rolling_beta_of_double_market_is_two():
    market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    stock = 2 * market
    beta = rolling_beta(stock, market, window=250).dropna()
    assert beta.iloc[-1] == pytest.approx(2.0)