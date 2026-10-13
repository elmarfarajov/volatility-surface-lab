"""Monte Carlo for exotics: the Brownian bridge, the BGK correction, the Asian control variate, and three models of a surface."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volsurf.exotics import closed_form as cf
from volsurf.exotics.model_risk import contracts, price_book
from volsurf.exotics.monte_carlo import (
    BETA,
    arithmetic_asian,
    barrier_mc,
    bgk_barrier,
    black_scholes_paths,
    heston_paths,
    local_vol_paths,
    survival_bridge,
)
from volsurf.heston import HestonParams
from volsurf.market.black import black_price
from volsurf.market.chains import Chain
from volsurf.market.cleaning import prepare
from volsurf.market_data import synthetic_chain
from volsurf.smile.build import build_surface
from volsurf.smile.interpolation import Surface
from volsurf.stochvol import pricing
from volsurf.stochvol.characteristic import Heston

S, T, R, B, SIGMA = 100.0, 1.0, 0.05, 0.03, 0.25
HESTON = Heston(0.03, 2.0, 0.04, 0.6, -0.7)


@pytest.fixture(scope="module")
def heston_surface() -> Surface:
    params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(30, 72, 170, 352, 540))
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
    chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
    return build_surface(prepare(chain)).surface


def test_beta_is_minus_zeta_of_a_half_over_root_two_pi():
    assert BETA == pytest.approx(0.5825971579, abs=1e-9)


@pytest.mark.parametrize(("kind", "h"), [("down-out", 85.0), ("up-out", 120.0), ("down-in", 85.0), ("up-in", 120.0)])
def test_the_bridge_recovers_the_continuous_price_from_a_coarse_grid(kind, h):
    exact = cf.barrier(kind, S, 100.0, h, T, R, B, SIGMA, True)
    for m in (12, 52):
        estimate = barrier_mc(kind, S, 100.0, h, T, R, B, SIGMA, True, m, 100_000, bridge=True, seed=21)
        assert abs(estimate.value - exact) < 4 * estimate.stderr


@pytest.mark.parametrize(("kind", "h"), [("down-out", 85.0), ("up-out", 120.0)])
def test_bgk_matches_daily_monitoring_and_the_discrete_price_exceeds_the_continuous(kind, h):
    estimate = barrier_mc(kind, S, 100.0, h, T, R, B, SIGMA, True, 252, 200_000, seed=22)
    corrected = bgk_barrier(kind, S, 100.0, h, T, R, B, SIGMA, True, 252)
    assert abs(estimate.value - corrected) < 4 * estimate.stderr
    assert corrected > cf.barrier(kind, S, 100.0, h, T, R, B, SIGMA, True) + 0.1


def test_bgk_fails_for_coarse_monitoring_near_the_barrier():
    # documented in ADR 0022: monthly monitoring of an up-and-out 20% away overprices by about 9%
    estimate = barrier_mc("up-out", S, 100.0, 120.0, T, R, B, SIGMA, True, 12, 200_000, seed=23)
    assert bgk_barrier("up-out", S, 100.0, 120.0, T, R, B, SIGMA, True, 12) > estimate.value + 10 * estimate.stderr


def test_survival_is_certain_far_from_the_barrier_and_zero_across_it():
    far = np.full((3, 4), 200.0)
    np.testing.assert_allclose(survival_bridge(far, 200.0, 50.0, 0.2, 0.25, down=True), 1.0, atol=1e-12)
    across = np.array([[110.0, 90.0, 120.0]])
    assert survival_bridge(across, 100.0, 95.0, 0.2, 1 / 3, down=True)[0] == 0.0


def test_paths_are_martingales_after_carry_under_every_model(heston_surface):
    disc_carry = np.exp(-B * np.arange(1, 13) / 12)
    for paths in (
        black_scholes_paths(1.0, T, R, B, SIGMA, 12, 100_000, seed=24),
        heston_paths(1.0, T, R, B, HESTON, 12, 100_000, seed=24),
        local_vol_paths(1.0, T, R, B, heston_surface, 12, 100_000, seed=24),
    ):
        mean = paths.mean(axis=0) * disc_carry
        stderr = paths.std(axis=0) * disc_carry / np.sqrt(paths.shape[0])
        assert np.all(np.abs(mean - 1.0) < 4 * stderr + 1e-4)


def test_heston_paths_reprice_heston_vanillas():
    paths = heston_paths(1.0, T, R, B, HESTON, 52, 200_000, seed=25)
    strikes = np.array([0.8, 1.0, 1.2])
    fwd = np.exp(B * T)
    exact = pricing.price(fwd, strikes, T, HESTON, np.exp(-R * T), is_call=strikes >= 1.0)
    for k, target in zip(strikes, exact, strict=True):
        payoff = np.maximum(paths[:, -1] - k, 0.0) if k >= 1.0 else np.maximum(k - paths[:, -1], 0.0)
        sample = np.exp(-R * T) * payoff
        assert abs(sample.mean() - target) < 4 * sample.std() / np.sqrt(sample.size) + 2e-4  # plus QE's weekly bias


def test_local_volatility_paths_reprice_the_surface(heston_surface):
    t = 352 / 365
    paths = local_vol_paths(1.0, t, 0.0, 0.0, heston_surface, 100, 100_000, seed=26)
    for strike in (0.85, 1.0, 1.1):
        k = np.log(strike)
        target = float(black_price(1.0, strike, t, heston_surface.implied_vol(np.array([k]), t)[0], 1.0, strike >= 1.0))
        payoff = np.maximum(paths[:, -1] - strike, 0.0) if strike >= 1.0 else np.maximum(strike - paths[:, -1], 0.0)
        assert abs(payoff.mean() - target) < 4 * payoff.std() / np.sqrt(payoff.size) + 1e-3  # plus Euler's bias


def test_the_geometric_control_removes_most_variance_and_agrees_with_the_plain_estimate():
    estimate = arithmetic_asian(S, 100.0, T, R, B, SIGMA, 52, True, 100_000, seed=27)
    assert estimate.variance_ratio > 300
    assert abs(estimate.plain.value - estimate.controlled.value) < 4 * estimate.plain.stderr
    # QuantLib's MCDiscreteArithmeticAPEngine with its own control variate gives 6.35795 +- 0.00066 for this option (364 days)
    reference = arithmetic_asian(S, 100.0, 364 / 365, R, B, SIGMA, 52, True, 400_000, seed=28).controlled
    assert abs(reference.value - 6.35795) < 4 * np.hypot(reference.stderr, 0.00066)


def test_arithmetic_asian_lies_between_the_geometric_and_the_vanilla():
    estimate = arithmetic_asian(S, 100.0, T, R, B, SIGMA, 52, True, 100_000, seed=29).controlled
    assert cf.geometric_asian(S, 100.0, T, R, B, SIGMA, True, 52) < estimate.value < cf.black_scholes(S, 100.0, T, R, B, SIGMA)


def test_model_risk_book_on_a_heston_surface(heston_surface):
    book = price_book(1.0, 0.04, 0.027, 0.2, HESTON, heston_surface, paths=40_000, steps=126, seed=30)
    assert book.models == ("Black-Scholes", "Heston", "local volatility")
    assert book.contracts == tuple(c.name for c in contracts())
    atm = "call at the money"
    # both smile models price the vanilla the surface was built from, within noise and discretisation
    assert abs(book.prices[("Heston", atm)].value - book.prices[("local volatility", atm)].value) < 2e-3
    # the knock-out is worth less than the vanilla under every model, the one-touch is a probability
    for model in book.models:
        assert book.prices[(model, "up-and-out call, barrier 120%")].value < book.prices[(model, atm)].value
        assert 0.0 < book.prices[(model, "one-touch at 85%")].value < 1.0
    # the negative skew makes the 90% put dearer than at one flat volatility
    assert book.prices[("local volatility", "put at 90%")].value > book.prices[("Black-Scholes", "put at 90%")].value
