"""SABR: Hagan's expansion against QuantLib, the exact conditional reference, densities and calibration."""

from __future__ import annotations

import numpy as np
import pytest
import QuantLib as ql

from volsurf.localvol.sabr import SABR, calibrate, density_from_vols, hagan_vol, reference_vols
from volsurf.market.implied import implied_vol


def test_hagan_matches_quantlib():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = SABR(
            rng.uniform(0.05, 0.5), float(rng.choice([0.0, 0.3, 0.5, 0.7, 1.0])), rng.uniform(-0.9, 0.9), rng.uniform(0.05, 2.0)
        )
        f, t = rng.uniform(0.01, 2.0), rng.uniform(0.05, 5)
        k = f * np.exp(np.linspace(-0.5, 0.5, 11))
        theirs = [ql.sabrVolatility(float(kk), float(f), float(t), p.alpha, p.beta, p.nu, p.rho) for kk in k]
        np.testing.assert_allclose(hagan_vol(f, k, t, p), theirs, rtol=1e-12)


def test_obloj_agrees_with_hagan_at_beta_one_and_at_the_money():
    p = SABR(0.2, 1.0, -0.6, 0.8)
    k = np.exp(np.linspace(-0.5, 0.3, 9))
    np.testing.assert_allclose(hagan_vol(1.0, k, 1.0, p, obloj=True), hagan_vol(1.0, k, 1.0, p), rtol=1e-12)
    q = SABR(0.035, 0.5, -0.3, 0.6)
    assert hagan_vol(0.03, np.array([0.03]), 1.0, q, obloj=True)[0] == pytest.approx(
        hagan_vol(0.03, np.array([0.03]), 1.0, q)[0], rel=1e-6
    )
    assert abs(hagan_vol(0.03, np.array([0.005]), 1.0, q, obloj=True)[0] - hagan_vol(0.03, np.array([0.005]), 1.0, q)[0]) > 1e-4


def test_the_reference_agrees_with_quantlibs_sabr_pde():
    p = SABR(0.2, 1.0, -0.7, 1.0)
    days = 183
    t = days / 365
    k = np.exp(np.array([-0.3, 0.0, 0.15]))
    ref = reference_vols(1.0, k, t, p, paths=200_000, steps=200, seed=4)
    today = ql.Date(13, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    engine = ql.FdSabrVanillaEngine(1.0, p.alpha, 0.999, p.nu, p.rho, flat, 200, 400, 100, 1, 0.8)  # beta < 1 only
    prices = []
    for kk in k:
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call if kk >= 1 else ql.Option.Put, float(kk)), ql.EuropeanExercise(today + days)
        )
        option.setPricingEngine(engine)
        prices.append(option.NPV())
    fd = implied_vol(np.array(prices), 1.0, k, t, 1.0, k >= 1)
    assert np.all(np.abs(fd - ref.vols) < 4 * ref.stderr + 5e-4)
    with pytest.raises(ValueError):
        reference_vols(1.0, k, t, SABR(0.2, 0.5, -0.7, 1.0))


def test_hagan_is_accurate_when_nu_squared_t_is_small_and_not_when_it_is_large():
    k = np.exp(np.linspace(-0.2, 0.1, 5))
    small = SABR(0.2, 1.0, -0.5, 0.3)
    ref = reference_vols(1.0, k, 0.1, small, paths=200_000, steps=100, seed=2)
    assert np.all(np.abs(hagan_vol(1.0, k, 0.1, small) - ref.vols) < 3 * ref.stderr + 2e-4)
    large = SABR(0.2, 1.0, -0.7, 1.0)
    ref = reference_vols(1.0, np.array([np.exp(-0.4)]), 1.0, large, paths=100_000, steps=200, seed=2)
    # 1.5 volatility points off, thirteen standard errors of the reference
    assert hagan_vol(1.0, np.array([np.exp(-0.4)]), 1.0, large)[0] - ref.vols[0] > 8 * ref.stderr[0]


def test_hagan_densities_turn_negative_at_low_strikes_for_long_maturities():
    q = SABR(0.035, 0.5, -0.3, 0.6)
    k = np.linspace(0.0005, 0.08, 1600)
    assert density_from_vols(0.03, k, 1.0, hagan_vol(0.03, k, 1.0, q)).min() > 0
    assert density_from_vols(0.03, k, 10.0, hagan_vol(0.03, k, 10.0, q)).min() < 0


def test_calibration_recovers_parameters():
    true = SABR(0.18, 1.0, -0.65, 1.3)
    k = np.exp(np.linspace(-0.4, 0.15, 21))
    fit = calibrate(1.0, k, 0.5, hagan_vol(1.0, k, 0.5, true, obloj=True))
    assert fit.rmse < 1e-8
    np.testing.assert_allclose([fit.params.alpha, fit.params.rho, fit.params.nu], [true.alpha, true.rho, true.nu], rtol=1e-5)
