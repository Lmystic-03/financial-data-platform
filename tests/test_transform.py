import numpy as np
import pandas as pd

from transform import add_features


def make_raw(n: int = 30) -> pd.DataFrame:
    """Fake price table: 30 business days, price rising from 100 to 130."""
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    close = np.linspace(100, 130, n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Open": close,
            "Volume": 1000,
        }
    )


def test_duplicate_dates_are_removed():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[5]]])  # add a duplicated day
    out = add_features(raw, "TEST")
    assert out["Date"].is_unique


def test_log_return_value():
    out = add_features(make_raw(), "TEST")
    expected = np.log(out["Close"].iloc[1] / out["Close"].iloc[0])
    assert np.isclose(out["log_return"].iloc[1], expected)


def test_first_return_is_missing():
    out = add_features(make_raw(), "TEST")
    assert np.isnan(out["log_return"].iloc[0])


def test_moving_average_missing_count():
    out = add_features(make_raw(), "TEST")
    assert out["ma_20"].isna().sum() == 19


def test_volatility_missing_count():
    out = add_features(make_raw(), "TEST")
    assert out["volatility_20"].isna().sum() == 20