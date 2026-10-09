import numpy as np
import pandas as pd
import pytest

from alpha import capm_regression


def test_capm_recovers_known_alpha_and_beta():
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, 1000), name="SPY")
    stock = 0.0002 + 1.5 * market
    out = capm_regression(stock, market)
    assert out["beta"] == pytest.approx(1.5)
    assert out["alpha_daily_%"] == pytest.approx(0.02)