import numpy as np
import pytest

from volsurf.black_scholes import bsm_price
from volsurf.heston import HestonParams, heston_price_cos_spot
from volsurf.monte_carlo import (
    lsm_american_price,
    mc_european_gbm,
    mc_european_heston,
    simulate_gbm_paths,
    simulate_heston_paths,
)

PARAMS = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)


def test_gbm_price_within_standard_errors():
    result = mc_european_gbm(100, 105, 1.0, 0.04, 0.01, 0.25, True, 100_000, seed=1)
    assert result.contains(float(bsm_price(100, 105, 1.0, 0.04, 0.01, 0.25, True)))


def test_control_variate_reduces_variance():
    plain = mc_european_gbm(100, 100, 1.0, 0.04, 0.0, 0.2, True, 60_000, seed=2, control_variate=False)
    controlled = mc_european_gbm(100, 100, 1.0, 0.04, 0.0, 0.2, True, 60_000, seed=2, control_variate=True)
    assert controlled.std_error < 0.5 * plain.std_error


def test_gbm_paths_are_reproducible_and_antithetic():
    a = simulate_gbm_paths(100, 1.0, 0.03, 0.0, 0.2, 12, 1000, seed=5)
    b = simulate_gbm_paths(100, 1.0, 0.03, 0.0, 0.2, 12, 1000, seed=5)
    np.testing.assert_array_equal(a, b)
    log_increments = np.diff(np.log(a), axis=1)
    np.testing.assert_allclose(log_increments[:500] + log_increments[500:], 2 * (0.03 - 0.02) / 12, atol=1e-12)


def test_heston_qe_is_martingale():
    paths = simulate_heston_paths(100, 1.0, 0.03, 0.01, PARAMS, 25, 60_000, seed=9)
    discounted = np.exp(-0.03) * paths.spot[:, -1]
    target = 100 * np.exp(-0.01)
    assert abs(discounted.mean() - target) < 4 * discounted.std() / np.sqrt(discounted.size)
    assert (paths.variance >= 0).all()


def test_heston_qe_price_matches_cos():
    exact = float(heston_price_cos_spot(100, [100.0], 1.0, 0.0, 0.0, PARAMS, True)[0])
    result = mc_european_heston(100, 100, 1.0, 0.0, 0.0, PARAMS, True, n_steps=50, n_paths=60_000, seed=4)
    assert result.contains(exact, n_std=4.0, abs_tol=0.02)


def test_longstaff_schwartz_benchmark():
    result = lsm_american_price(36, 40, 1.0, 0.06, 0.0, 0.2, False, 50, 60_000, seed=11)
    assert result.price == pytest.approx(4.472, abs=0.04)
    assert result.price > bsm_price(36, 40, 1.0, 0.06, 0.0, 0.2, False)
