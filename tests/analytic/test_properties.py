"""What must hold for any contract, property-tested: the PDE, parity, homogeneity, the bounds, the limits."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volsurf.analytic import bound_violations, greeks, merton_bounds, price, strike_violations

spots = st.floats(5.0, 500.0)
moneyness = st.floats(-1.5, 1.5)  # ln(K/S)
maturities = st.floats(0.01, 10.0)
rates = st.floats(-0.03, 0.15)
yields = st.floats(-0.02, 0.10)
vols = st.floats(0.02, 2.0)


@settings(max_examples=400, deadline=None)
@given(spots, moneyness, maturities, rates, yields, vols, st.booleans())
def test_every_price_solves_the_black_scholes_pde(S, m, T, r, q, sigma, call):
    """dV/dt + (r - q) S dV/dS + sigma^2 S^2/2 d2V/dS2 - r V = 0, with theta as time passing."""
    g = greeks(S, S * np.exp(m), T, r, q, sigma, call)
    residual = g.theta + (r - q) * S * g.delta + 0.5 * sigma**2 * S**2 * g.gamma - r * g.price
    scale = max(abs(float(g.theta)), abs(float(r * g.price)), 0.5 * sigma**2 * S**2 * abs(float(g.gamma)), 1e-8)
    assert abs(float(residual)) <= 1e-10 * scale


@settings(max_examples=400, deadline=None)
@given(spots, moneyness, maturities, rates, yields, vols)
def test_calls_and_puts_obey_parity_in_price_and_in_every_greek(S, m, T, r, q, sigma):
    K = S * np.exp(m)
    call, put = greeks(S, K, T, r, q, sigma, True), greeks(S, K, T, r, q, sigma, False)
    forward_leg, strike_leg = S * np.exp(-q * T), K * np.exp(-r * T)
    # C - P = S e^{-qT} - K e^{-rT}, so the difference of each Greek is that of the forward contract
    expected = {
        "price": forward_leg - strike_leg,
        "delta": np.exp(-q * T),
        "theta": q * forward_leg - r * strike_leg,
        "rho": T * strike_leg,
        "psi": -T * forward_leg,
        "dual_delta": -np.exp(-r * T),
        "charm": q * np.exp(-q * T),  # the forward's delta, e^{-qT}, decays as time passes
    }
    for name, value in call.as_dict().items():
        difference = float(value - getattr(put, name))
        target = expected.get(name, 0.0)  # every curvature Greek is the same for the call and the put
        assert difference == pytest.approx(target, abs=1e-9 * max(1.0, S)), name


@settings(max_examples=400, deadline=None)
@given(spots, moneyness, maturities, rates, yields, vols, st.booleans())
def test_the_price_is_homogeneous_of_degree_one_in_spot_and_strike(S, m, T, r, q, sigma, call):
    """Euler: V = S dV/dS + K dV/dK, and doubling both doubles the price."""
    K = S * np.exp(m)
    g = greeks(S, K, T, r, q, sigma, call)
    assert float(g.price) == pytest.approx(float(S * g.delta + K * g.dual_delta), abs=1e-10 * max(S, K))
    assert float(price(2 * S, 2 * K, T, r, q, sigma, call)) == pytest.approx(2 * float(g.price), rel=1e-12, abs=1e-14)


@settings(max_examples=400, deadline=None)
@given(spots, moneyness, maturities, rates, yields, vols, st.booleans())
def test_model_prices_respect_mertons_model_free_bounds(S, m, T, r, q, sigma, call):
    K = S * np.exp(m)
    value = price(S, K, T, r, q, sigma, call)
    assert not bound_violations(value, S, K, T, r, q, call, tolerance=1e-10 * S)


@settings(max_examples=100, deadline=None)
@given(spots, maturities, rates, yields, vols)
def test_model_call_prices_are_decreasing_and_convex_in_strike(S, T, r, q, sigma):
    strikes = S * np.exp(np.linspace(-1.0, 1.0, 41))
    calls = price(S, strikes, T, r, q, sigma, True)
    assert not strike_violations(strikes, calls, np.exp(-r * T), tolerance=1e-10)


def test_a_butterfly_with_negative_value_is_reported():
    strikes = np.array([90.0, 100.0, 110.0])
    assert [v.kind for v in strike_violations(strikes, [12.0, 8.0, 3.0], 0.99)] == ["not convex"]
    assert [v.kind for v in strike_violations(strikes, [10.0, 11.0, 3.0], 0.99)][0] == "increasing in strike"


def test_bounds_are_the_forward_and_the_discounted_strike():
    lower, upper = merton_bounds(100.0, 90.0, 1.0, 0.05, 0.0, True)
    assert float(lower) == pytest.approx(100 - 90 * np.exp(-0.05)) and float(upper) == 100.0


# ---------------------------------------------------------------------- limits at expiry and at zero volatility
def test_at_expiry_an_option_is_its_intrinsic_value_and_delta_is_a_step():
    g = greeks(100.0, [90.0, 110.0, 100.0], 0.0, 0.03, 0.0, 0.2, True)
    np.testing.assert_allclose(g.price, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(g.delta, [1.0, 0.0, 0.5])
    assert g.gamma[0] == 0.0 and g.gamma[1] == 0.0 and np.isnan(g.gamma[2])  # at the money the limit does not exist


def test_zero_volatility_gives_discounted_intrinsic_on_the_forward_without_warnings():
    with np.errstate(all="raise"):
        g = greeks(100.0, [90.0, 110.0], 1.0, 0.03, 0.01, 0.0, False)
    forward_leg, strike_leg = 100 * np.exp(-0.01), 110 * np.exp(-0.03)
    np.testing.assert_allclose(g.price, [0.0, strike_leg - forward_leg])
    np.testing.assert_allclose(g.delta, [0.0, -np.exp(-0.01)])
    assert np.signbit(g.price).sum() == 0  # no negative zeros


@settings(max_examples=200, deadline=None)
@given(spots, moneyness, rates, yields, vols, st.booleans())
def test_greeks_approach_their_limits_as_expiry_nears(S, m, r, q, sigma, call):
    """A day before expiry, away from the money, the option is almost its intrinsic value."""
    K = S * np.exp(m)
    if abs(m) < 0.25:
        return  # near the money a day is still a long time
    near, at = greeks(S, K, 1e-6, r, q, sigma, call), greeks(S, K, 0.0, r, q, sigma, call)
    assert float(near.price) == pytest.approx(float(at.price), abs=1e-6 * S)
    assert float(near.delta) == pytest.approx(float(at.delta), abs=1e-6)


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="positive"):
        greeks(-1.0, 100.0, 1.0, 0.0, 0.0, 0.2)
    with pytest.raises(ValueError, match="negative"):
        greeks(100.0, 100.0, -1.0, 0.0, 0.0, 0.2)
