import numpy as np
import pandas as pd
import pytest

from portfolio import min_variance_weights


def test_min_variance_weights_are_valid_and_favor_low_vol():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({
        "LOW": rng.normal(0, 0.005, 1000),
        "HIGH": rng.normal(0, 0.02, 1000),
    })
    w = min_variance_weights(returns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert w["LOW"] > w["HIGH"]