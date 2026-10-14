"""The hedging engine, the gamma formula, the Greek explain, and the Heston and local-volatility pricers behind them."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.analytic.generalized import greeks
from volsurf.risk.explain import explain
from volsurf.risk.hedging import black_scholes_study, gbm_paths, hedge, wrong_volatility
from volsurf.risk.models import heston_study
from volsurf.risk.pricers import HestonGrid, LocalVolGrid, cos_calls, minimum_variance_delta
from volsurf.smile.interpolation import Surface
from volsurf.smile.svi import RawSVI
from volsurf.stochvol import pricing
from volsurf.stochvol.characteristic import Heston

H = Heston(0.04, 2.0, 0.04, 0.6, -0.7)


def test_an_unhedged_book_earns_the_premium_less_the_discounted_payoff():
    spot, times = gbm_paths(0.5, 0.05, 0.02, 0.2, 10, 1000, seed=1)
    payoff = np.maximum(spot[:, -1] - 1.0, 0.0)
    pnl = hedge(spot, times, 0.05, 0.02, 0.07, payoff, np.zeros((1000, 10)))
    np.testing.assert_allclose(pnl, 0.07 - np.exp(-0.05 * 0.5) * payoff, atol=1e-14)


def test_holding_one_share_returns_the_dividends_and_nothing_else():
    # short nothing (zero payoff, zero premium), long one share financed at r: P&L = discounted (S_T e^{q-ish} - S_0 e^{rT})
    spot, times = gbm_paths(1.0, 0.0, 0.0, 0.2, 252, 2000, seed=2)
    pnl = hedge(spot, times, 0.0, 0.0, 0.0, np.zeros(2000), np.ones((2000, 252)))
    np.testing.assert_allclose(pnl, spot[:, -1] - 1.0, atol=1e-12)


def test_delta_hedging_follows_derman_kamal_and_a_gamma_hedge_beats_it():
    study = black_scholes_study(paths=8000, steps=252, rebalances=(12, 63, 252), show=63)
    t = study.table
    np.testing.assert_allclose(t["delta"], t["derman_kamal"], rtol=0.08)
    assert np.all(t["delta_gamma"] < 0.3 * t["delta"])
    assert set(study.pnl) == {"delta", "delta and gamma"}


def test_the_gamma_formula_explains_hedging_at_the_wrong_volatility():
    w = wrong_volatility(paths=6000, steps=126)
    assert np.corrcoef(w.pnl, w.predicted)[0, 1] > 0.99
    assert w.pnl.mean() < 0  # short gamma, hedged too low
    assert abs(w.pnl.mean() - w.predicted.mean()) < 0.05 * abs(w.pnl.mean())
    up = wrong_volatility(sigma_true=0.2, sigma_hedge=0.25, paths=6000, steps=126)
    assert up.pnl.mean() > 0


def test_theta_and_the_explain_are_consistent_with_revaluation():
    g = greeks(1.0, 1.0, 0.25, 0.04, 0.01, 0.2)
    dt = 1 / 252
    assert float(g.theta) * dt == pytest.approx(float(greeks(1.0, 1.0, 0.25 - dt, 0.04, 0.01, 0.2).price - g.price), rel=0.01)
    ds, dv = np.array([0.005, -0.01, 0.02]), np.array([-0.002, 0.004, -0.01])
    full = greeks(1.0 + ds, 1.0, 0.25 - dt, 0.04, 0.01, 0.2 + dv).price - g.price
    e = explain(g, ds, dv, dt, full)
    assert np.all(np.abs(e.unexplained) < 2e-4)
    assert np.all(np.abs(e.unexplained) < 0.05 * np.abs(full))
    # the residual grows with the move: a 15% fall is not explained to second order
    big = greeks(0.85, 1.0, 0.25 - dt, 0.04, 0.01, 0.3).price - g.price
    assert abs(explain(g, -0.15, 0.1, dt, big).unexplained) > 10 * np.max(np.abs(e.unexplained))


def test_cos_calls_agree_with_the_contour_integral():
    k = np.linspace(-0.8, 0.5, 41)
    for v0, tau in ((1e-7, 0.02), (0.01, 0.1), (0.04, 0.25), (0.3, 1.0)):
        h = Heston(v0, H.kappa, H.theta, H.sigma, H.rho)
        calls = pricing.price(1.0, np.exp(k), tau, h, 1.0, True)
        np.testing.assert_allclose(cos_calls(k, tau, h), calls, atol=2e-6)


def test_heston_grid_prices_and_greeks():
    times = np.array([0.0, 0.1])
    grid = HestonGrid.build(H, 1.0, 0.25, 0.04, 0.03, times, v_points=28)
    spots = np.array([0.85, 1.0, 1.1])
    for i, t in enumerate(times):
        tau = 0.25 - t
        for v0 in (0.02, 0.06):
            val = grid.at(i, spots, np.full(3, v0))
            hv = Heston(v0, H.kappa, H.theta, H.sigma, H.rho)

            def direct(s, h=hv, tau=tau):
                return pricing.price(s * np.exp(0.03 * tau), np.ones(3), tau, h, np.exp(-0.04 * tau), True)

            np.testing.assert_allclose(val.value, direct(spots), atol=2e-6)
            fd = (direct(spots * (1 + 1e-4)) - direct(spots * (1 - 1e-4))) / (2e-4 * spots)
            np.testing.assert_allclose(val.delta, fd, atol=5e-5)
    mv = minimum_variance_delta(H, spots, val.delta, val.vega_v)
    np.testing.assert_allclose(mv, val.delta + H.rho * H.sigma * val.vega_v / spots)


def test_local_volatility_on_a_flat_surface_is_black_scholes():
    # one slice beyond expiry: before it the self-similar short end is exactly flat
    surface = Surface(np.array([2.0]), [RawSVI(0.04 * 2.0, 0.0, 0.0, 0.0, 0.1)])
    grid = LocalVolGrid.build(surface, 1.0, 0.5, 0.04, 0.03)
    spots = np.array([0.7, 0.9, 1.0, 1.1, 1.3])
    for t in (0.0, 0.2, 0.45):
        val = grid.at(t, spots)
        bs = greeks(spots, 1.0, 0.5 - t, 0.04, 0.01, 0.2)
        np.testing.assert_allclose(val.value, bs.price, atol=5e-5)
        np.testing.assert_allclose(val.delta, bs.delta, atol=5e-4)


def test_in_hestons_world_delta_plateaus_and_a_vega_hedge_does_not():
    study = heston_study(H, steps=16, paths=4000, rebalances=(4, 16), hedge_expiry=0.5)
    t = study.table.pivot(index="rebalances", columns="strategy", values="std")
    vega = t["delta and vega, with a longer call"]
    mv = t["minimum-variance delta"]
    assert vega[16] < 0.6 * vega[4]  # still falling
    assert mv[16] > 0.6 * mv[4]  # stalled
    assert mv[16] < t["Heston delta"][16]
    assert abs(study.table["mean"]).max() < 3 * study.table["std"].max() / np.sqrt(4000)
