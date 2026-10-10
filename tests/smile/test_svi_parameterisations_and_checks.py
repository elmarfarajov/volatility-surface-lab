"""SVI's three parameterisations, its derivatives, and the arbitrage checks on the whole real line."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from volsurf.smile.svi import RawSVI, butterfly_check, calendar_check

VOGT = RawSVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)


@settings(max_examples=200, deadline=None)
@given(
    a=st.floats(0.0, 0.1),
    b=st.floats(0.01, 1.0),
    rho=st.floats(-0.95, 0.95),
    m=st.floats(-0.5, 0.5),
    sigma=st.floats(0.02, 1.0),
)
def test_natural_and_jump_wings_round_trips(a, b, rho, m, sigma):
    assume(abs(m) > 1e-3 or abs(rho) > 1e-3)  # with the minimum at the money, jump-wings cannot see sigma
    s = RawSVI(a, b, rho, m, sigma)
    k = np.linspace(-2, 2, 41)
    np.testing.assert_allclose(s.natural().raw().total_variance(k), s.total_variance(k), rtol=1e-10, atol=1e-12)
    # the jump-wings inverse is ill-conditioned when the skew parameter is near its bounds: relative tolerance
    np.testing.assert_allclose(s.jump_wings(0.7).raw().total_variance(k), s.total_variance(k), rtol=1e-7, atol=1e-10)


def test_analytic_derivatives_and_g():
    s = RawSVI(0.02, 0.3, -0.6, 0.05, 0.15)
    k, h = np.linspace(-1, 1, 21), 1e-5
    np.testing.assert_allclose(s.dw(k), (s.total_variance(k + h) - s.total_variance(k - h)) / (2 * h), rtol=1e-8)
    np.testing.assert_allclose(s.d2w(k), (s.dw(k + h) - s.dw(k - h)) / (2 * h), rtol=1e-6)
    w, w1, w2 = s.total_variance(k), s.dw(k), s.d2w(k)
    expected = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
    np.testing.assert_allclose(s.g(k), expected)


def test_vogts_slice_is_caught_where_gatheral_and_jacquier_say():
    check = butterfly_check(VOGT)
    assert not check.passed and check.reason == "density negative"
    assert check.worst == pytest.approx(-0.0329, abs=5e-4)
    assert 0.6 < check.at < 1.2


def test_the_wings_are_settled_analytically():
    steep = RawSVI(0.01, 1.5, 0.5, 0.0, 0.1)  # right wing slope 2.25 > 2
    check = butterfly_check(steep)
    assert not check.passed and "Lee" in check.reason and np.isposinf(check.at)
    fine = butterfly_check(RawSVI(0.04, 0.1, -0.5, 0.0, 0.2))
    assert fine.passed and fine.worst == pytest.approx(min(0.25 - 0.15**2 / 16, 0.25 - 0.05**2 / 16), rel=1e-3)


def test_arbitrage_outside_any_fitting_grid_is_still_found():
    # Vogt's slice moved three units of log-moneyness out: a grid on [-1, 1] sees nothing wrong
    far = RawSVI(VOGT.a, VOGT.b, VOGT.rho, VOGT.m + 3.0, VOGT.sigma)
    assert far.g(np.linspace(-1, 1, 81)).min() > 0
    check = butterfly_check(far)
    # g depends on k itself, not only on k - m, so the dip does not simply move by three: it is found far out
    assert not check.passed and check.at > 3.0


def test_calendar_crossings_in_the_wings_and_in_between():
    base = RawSVI(0.04, 0.1, -0.5, 0.0, 0.2)
    assert calendar_check(base, RawSVI(0.05, 0.12, -0.5, 0.0, 0.2)).passed
    wing = calendar_check(base, RawSVI(0.05, 0.08, -0.5, 0.0, 0.2))
    assert not wing.passed and wing.reason == "left wing crosses"
    # steeper wings but a lower ATM variance (0.054 against 0.06): crosses near the money only
    dip = calendar_check(base, RawSVI(0.03, 0.12, -0.5, 0.0, 0.2))
    assert not dip.passed and dip.reason == "slices cross" and abs(dip.at) < 0.5


def test_jump_wings_refuse_a_symmetric_slice_at_the_money():
    with pytest.raises(ValueError, match="sigma"):
        RawSVI(0.02, 0.2, 0.0, 0.0, 0.3).jump_wings(1.0).raw()


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        RawSVI(0.0, -0.1, 0.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        RawSVI(0.0, 0.1, 1.0, 0.0, 0.1)
