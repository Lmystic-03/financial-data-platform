import numpy as np
import pandas as pd
import pytest

from backtest import count_exceptions, kupiec_test


def test_kupiec_perfect_rate_gives_zero_lr():
    # 5 exceptions sur 100 jours avec un niveau de 95% : taux observé = taux attendu
    lr, pval = kupiec_test(5, 100, 0.95)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert pval == pytest.approx(1.0, abs=1e-6)


def test_kupiec_bad_rate_is_rejected():
    # 20 exceptions sur 100 jours alors qu'on en attend 5 : rejet net
    lr, pval = kupiec_test(20, 100, 0.95)
    assert lr > 3.84
    assert pval < 0.05


def test_kupiec_zero_exceptions_does_not_crash():
    lr, pval = kupiec_test(0, 250, 0.99)
    assert lr >= 0
    assert 0 <= pval <= 1


def test_count_exceptions_no_lookahead():
    # Série calme (rendements ~ 0), puis un choc de -10% au jour 300.
    # La VaR calculée sur les 250 jours précédents est minuscule,
    # donc le choc doit être une exception, et seulement lui.
    returns = pd.Series(np.zeros(400))
    returns.iloc[300] = -0.10
    returns += np.random.default_rng(0).normal(0, 0.001, 400)

    def var_func(r, level):
        return float(-np.quantile(r, 1 - level))

    exc = count_exceptions(returns, var_func, 0.99, window=250)
    assert exc.iloc[300 - 250]            # le choc est bien détecté
    assert exc.sum() <= 5                 # et presque rien d'autre