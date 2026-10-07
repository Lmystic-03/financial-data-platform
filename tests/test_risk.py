import numpy as np
import pandas as pd

from risk import expected_shortfall, historical_var, parametric_var


def fake_returns(n: int = 100_000) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=0.0, scale=0.01, size=n))


def test_var_99_is_larger_than_var_95():
    r = fake_returns()
    assert historical_var(r, 0.99) > historical_var(r, 0.95)


def test_expected_shortfall_is_larger_than_var():
    r = fake_returns()
    assert expected_shortfall(r, 0.95) > historical_var(r, 0.95)


def test_parametric_matches_historical_on_normal_data():
    r = fake_returns()
    assert abs(parametric_var(r, 0.95) - historical_var(r, 0.95)) < 0.001


def test_parametric_var_value():
    # for N(0, 0.01), 95% VaR is about 1.645 * 0.01
    r = fake_returns()
    assert abs(parametric_var(r, 0.95) - 0.01645) < 0.001