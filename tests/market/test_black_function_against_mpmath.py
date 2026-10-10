"""The normalised Black function against 60-digit arithmetic, QuantLib, and its own identities."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
import QuantLib as ql
from scipy.special import ndtr

from volsurf.black_scholes import black76_price
from volsurf.market.black import (
    black_price,
    intrinsic,
    log_otm_complement,
    log_otm_value,
    mills_ratio,
    mills_ratio_slope,
    normalised_black,
)


def _mp_call(x: float, s: float) -> mp.mpf:
    with mp.workdps(60):
        x, s = mp.mpf(x), mp.mpf(s)
        return mp.ncdf(x / s + s / 2) * mp.exp(x / 2) - mp.ncdf(x / s - s / 2) * mp.exp(-x / 2)


def test_mills_ratio_slope_is_exact_where_the_sum_cancels():
    z = -np.geomspace(1e-3, 1e6, 300)
    ours = mills_ratio_slope(z)
    with mp.workdps(60):
        exact = np.array([float(1 + mp.mpf(v) * mp.ncdf(v) / mp.npdf(v)) for v in z])
    np.testing.assert_allclose(ours, exact, rtol=16 * np.finfo(float).eps)
    near = z >= -1e4  # beyond, mpmath's own normal tail (exponents of -5e11) is the less accurate side
    with mp.workdps(60):
        ratio = [float(mp.ncdf(v) / mp.npdf(v)) for v in z[near]]
    np.testing.assert_allclose(mills_ratio(z[near]), ratio, rtol=16 * np.finfo(float).eps)


def test_out_of_the_money_prices_match_60_digits_over_300_orders_of_magnitude():
    rng = np.random.default_rng(5)
    x = -np.concatenate([np.zeros(20), 10 ** rng.uniform(-9, 1.2, 1480)])
    s = 10 ** rng.uniform(-3.5, 1.3, x.size)
    exact = np.array([_mp_call(a, b) for a, b in zip(x, s, strict=True)])
    live = np.array([v > mp.mpf("1e-300") for v in exact])
    log_exact = np.array([float(mp.log(v)) if ok else -np.inf for v, ok in zip(exact, live, strict=True)])
    error = np.abs(log_otm_value(x, s)[live] - log_exact[live])
    # ln b is accurate to a few roundings of its own size (h^2/2 carries the rounding of x/s)
    assert np.all(error <= 8 * np.finfo(float).eps * np.maximum(1.0, np.abs(log_exact[live])))
    price = normalised_black(x, s)[live]
    assert np.all(np.abs(price - np.exp(log_exact[live])) <= 1e-12 * np.exp(log_exact[live]))


def test_the_textbook_formula_loses_digits_that_the_mills_form_keeps():
    x, s = -0.6, 0.02
    exact = float(_mp_call(x, s))
    d1 = x / s + s / 2
    literal = np.exp(x / 2) * ndtr(d1) - np.exp(-x / 2) * ndtr(d1 - s)
    assert abs(literal / exact - 1) > 1e-12
    # what is left is the rounding of (x/s)^2 / 2 = 450 in the exponent: a few ulps of 450
    floor = 8 * np.finfo(float).eps * 0.5 * (x / s) ** 2
    assert abs(float(normalised_black(x, s)) / exact - 1) < floor
    # black76_price now goes through the Mills form too
    assert abs(float(black76_price(np.exp(x / 2), np.exp(-x / 2), 1.0, 1.0, s, True)) / exact - 1) < floor


def test_puts_in_the_money_and_the_upper_complement():
    x = np.array([-2.0, -0.3, 0.0, 0.4, 1.7])
    s = np.array([0.1, 0.5, 1.0, 0.25, 3.0])
    for is_call in (True, False):
        exact = np.array([float(_mp_call(a if is_call else -a, b)) for a, b in zip(x, s, strict=True)])
        # tolerance: a few ulps of ln b, which is -209 for the first quote
        np.testing.assert_allclose(normalised_black(x, s, is_call), exact, rtol=1e-13)
    gap = np.exp(-0.5 * np.abs(x)) - np.array([float(_mp_call(-abs(a), b)) for a, b in zip(x, s, strict=True)])
    np.testing.assert_allclose(np.exp(log_otm_complement(x, s)), gap, rtol=1e-13)
    np.testing.assert_allclose(intrinsic([1.0, -1.0], [True, False]), [2 * np.sinh(0.5)] * 2)


def test_limits_at_zero_volatility():
    assert normalised_black(-0.5, 0.0) == 0.0
    assert normalised_black(0.5, 0.0) == pytest.approx(2 * np.sinh(0.25))
    assert np.isneginf(log_otm_value(0.0, 0.0))
    assert black_price(100.0, 90.0, 0.0, 0.3, 0.95, True) == pytest.approx(0.95 * 10.0)


def test_discounted_prices_agree_with_quantlib():
    rng = np.random.default_rng(9)
    f, k = rng.uniform(50, 150, 400), rng.uniform(50, 150, 400)
    t, v, d = rng.uniform(0.01, 5, 400), rng.uniform(0.05, 1.0, 400), rng.uniform(0.8, 1.0, 400)
    call = rng.random(400) < 0.5
    ours = black_price(f, k, t, v, d, call)
    theirs = [
        ql.blackFormula(ql.Option.Call if c else ql.Option.Put, kk, ff, vv * np.sqrt(tt), dd)
        for ff, kk, tt, vv, dd, c in zip(f, k, t, v, d, call, strict=True)
    ]
    np.testing.assert_allclose(ours, theirs, rtol=1e-11, atol=1e-12)
