import numpy as np
import pandas as pd
import pytest

from portfolio import portfolio_returns, portfolio_vol


def test_perfectly_correlated_assets_give_no_diversification():
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    returns = pd.DataFrame({"A": r, "B": 2 * r})
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    expected = (0.5 * returns["A"].std() + 0.5 * returns["B"].std()) * np.sqrt(252)
    assert portfolio_vol(returns, w) == pytest.approx(expected)


def test_portfolio_returns_is_weighted_sum():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.00]})
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    pr = portfolio_returns(returns, w)
    assert pr.tolist() == pytest.approx([0.02, 0.01])