"""Calibration, identifiability diagnostics, and simulation across the Feller boundary."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.stochvol.calibration import Expiry, calibrate, identifiability, model_vols, profile
from volsurf.stochvol.characteristic import Heston
from volsurf.stochvol.pricing import otm_price
from volsurf.stochvol.simulation import exact_variance, exact_variance_cdf, otm_estimate, qe_variance, simulate

TRUE = Heston(0.025, 2.5, 0.05, 0.9, -0.65)


@pytest.fixture(scope="module")
def expiries():
    out = []
    for t in (0.1, 0.25, 0.5, 1.0, 2.0):
        k = np.linspace(-0.35, 0.15, 9) * np.sqrt(t / 0.25)
        vols = model_vols(TRUE, [Expiry(t, k, np.zeros_like(k), np.ones_like(k))])[0]
        out.append(Expiry(t, k, vols, np.full(k.size, 0.002)))
    return out


def test_calibration_recovers_known_parameters(expiries):
    fit = calibrate(expiries, Heston(0.04, 1.0, 0.04, 0.5, -0.3))
    np.testing.assert_allclose(fit.heston.as_array(), TRUE.as_array(), rtol=1e-3)
    assert fit.rmse < 1e-8


def test_fixed_parameters_stay_fixed_and_the_profile_is_minimal_at_the_truth(expiries):
    fits = profile(expiries, TRUE, np.array([1.5, 2.5, 4.0]))
    assert [f.heston.kappa for f in fits] == [1.5, 2.5, 4.0]
    weighted = [f.weighted for f in fits]
    assert weighted[1] < weighted[0] and weighted[1] < weighted[2]


def test_identifiability_at_the_truth(expiries):
    ident = identifiability(expiries, TRUE)
    assert ident.singular_values.size == 5 and ident.condition > 1
    assert np.linalg.norm(ident.weakest) == pytest.approx(1.0)


def test_qe_matches_the_first_two_moments_of_the_exact_transition():
    h, v, dt = Heston(0.04, 1.0, 0.04, 1.0, -0.7), 0.01, 0.25
    rng = np.random.default_rng(3)
    n = 400_000
    qe, *_ = qe_variance(np.full(n, v), dt, h, rng.standard_normal(n), rng.random(n))
    exact = exact_variance(np.full(n, v), dt, h, rng)
    assert qe.min() >= 0
    assert qe.mean() == pytest.approx(exact.mean(), rel=5e-3)
    assert qe.var() == pytest.approx(exact.var(), rel=2e-2)
    x = np.linspace(0.005, 0.1, 20)
    np.testing.assert_allclose(np.searchsorted(np.sort(exact), x) / n, exact_variance_cdf(x, v, dt, h), atol=5e-3)


def test_qe_prices_and_martingale_where_euler_fails():
    h = Heston(0.04, 1.0, 0.04, 1.0, -0.7)  # Feller ratio 0.08
    k = np.array([-0.3, 0.0, 0.2])
    exact = otm_price(k, 1.0, h)
    qe = otm_estimate(simulate(h, 1.0, 8, 200_000, "qe", seed=5), k)
    euler = otm_estimate(simulate(h, 1.0, 8, 200_000, "euler", seed=5), k)
    assert np.all(np.abs(qe.value - exact) < 4 * qe.stderr)
    assert np.max(np.abs(euler.value - exact) / euler.stderr) > 10
    paths = simulate(h, 1.0, 4, 200_000, "qe", seed=6)
    s = np.exp(paths.log_moneyness)
    assert abs(s.mean() - 1.0) < 4 * s.std() / np.sqrt(s.size)
    with pytest.raises(ValueError):
        simulate(h, 1.0, 2, 10, "milstein")
