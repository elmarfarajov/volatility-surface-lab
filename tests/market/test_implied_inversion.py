"""The implied-volatility inversion: held to the problem's own conditioning against 50-digit prices,
and compared with Jaeckel's reference implementation and QuantLib."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import py_lets_be_rational as lbr
import pytest
import QuantLib as ql
from hypothesis import given, settings
from hypothesis import strategies as st

from volsurf.implied_vol import implied_vol_black76
from volsurf.market import black_price, implied_vol, invert_normalised
from volsurf.market.black import log_otm_value, log_vega

EPS = np.finfo(float).eps


def _mp_beta(x: float, s: float, is_call: bool) -> float:
    with mp.workdps(50):
        xs, sm = mp.mpf(x if is_call else -x), mp.mpf(s)
        return float(mp.ncdf(xs / sm + sm / 2) * mp.exp(xs / 2) - mp.ncdf(xs / sm - sm / 2) * mp.exp(-xs / 2))


@pytest.fixture(scope="module")
def quotes():
    rng = np.random.default_rng(3)
    n = 3000
    x = np.concatenate([rng.uniform(-8, 8, n - 300), np.zeros(150), rng.uniform(-1e-6, 1e-6, 150)])
    s = 10 ** rng.uniform(-3.5, 1.2, n)
    call = rng.random(n) < 0.5
    beta = np.array([_mp_beta(a, b, c) for a, b, c in zip(x, s, call, strict=True)])
    with np.errstate(divide="ignore", over="ignore"):
        kappa = EPS * np.maximum(beta, 1e-300) / (s * np.exp(log_vega(-np.abs(x), s))) + EPS
    return x, s, call, beta, kappa


def test_error_is_within_a_few_condition_numbers_everywhere(quotes):
    x, s, call, beta, kappa = quotes
    well_posed = (beta > 0) & (kappa < 1e-3)
    assert well_posed.sum() > 1200
    result = invert_normalised(beta, x, call)
    error = np.abs(result.total_vol - s) / s
    assert not np.isnan(result.total_vol[well_posed]).any()
    assert np.max(error[well_posed] / kappa[well_posed]) < 8.0
    assert result.iterations[well_posed].max() <= 10


def test_agrees_with_jaeckels_lets_be_rational(quotes):
    x, s, call, beta, kappa = quotes
    well_posed = np.flatnonzero((beta > 0) & (kappa < 1e-6))[:800]
    ours = invert_normalised(beta[well_posed], x[well_posed], call[well_posed]).total_vol
    theirs = np.array(
        [
            lbr.normalised_implied_volatility_from_a_transformed_rational_guess(b, xi, 1.0 if c else -1.0)
            for b, xi, c in zip(beta[well_posed], x[well_posed], call[well_posed], strict=True)
        ]
    )
    np.testing.assert_allclose(ours, theirs, rtol=8 * kappa[well_posed].max() + 1e-14)


def test_the_v1_failure_region_is_now_exact():
    # prices between 1e-300 and 1e-10, where the v1.0 Newton iteration stopped early
    x = -np.linspace(0.05, 4.0, 60)
    s = np.geomspace(0.003, 0.2, 60)
    xx, ss = np.meshgrid(x, s)
    beta = np.exp(log_otm_value(xx, ss))
    tiny = (beta > 1e-300) & (beta < 1e-10)
    assert tiny.sum() > 500
    recovered = invert_normalised(beta[tiny], xx[tiny]).total_vol
    np.testing.assert_allclose(recovered, ss[tiny], rtol=1e-13)


def test_bounds_intrinsic_and_shapes():
    f, k, t, d = 100.0, 90.0, 0.5, 0.98
    out = implied_vol([d * (f - k) - 0.01, d * f + 1.0, -1.0, d * (f - k)], f, k, t, d, True)
    assert np.isnan(out[:3]).all() and out[3] == 0.0
    grid = black_price(100.0, np.full((3, 4), 105.0), 0.5, 0.2, 0.97, True)
    assert implied_vol(grid, 100.0, 105.0, 0.5, 0.97, True).shape == (3, 4)
    assert np.shape(implied_vol_black76(float(grid[0, 0]), 100.0, 105.0, 0.5, 0.97)) == ()
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, 1.0))


def test_quantlib_agrees_where_its_solver_is_accurate():
    rng = np.random.default_rng(11)
    f, k = 100.0, 100 * np.exp(rng.uniform(-0.5, 0.5, 200))
    t, v = rng.uniform(0.1, 3, 200), rng.uniform(0.1, 0.8, 200)
    price = black_price(f, k, t, v, 0.96, k >= f)
    ours = implied_vol(price, f, k, t, 0.96, k >= f)
    theirs = [
        ql.blackFormulaImpliedStdDev(side, kk, f, p, 0.96, 0.0, ql.nullDouble(), 1e-14, 200) / np.sqrt(tt)
        for kk, p, tt in zip(k, price, t, strict=True)
        for side in [ql.Option.Call if kk >= f else ql.Option.Put]
    ]
    np.testing.assert_allclose(ours, v, rtol=1e-13)
    np.testing.assert_allclose(theirs, v, rtol=5e-8)  # QuantLib's Newton stops earlier


@settings(max_examples=300, deadline=None)
@given(
    x=st.floats(-6.0, 6.0),
    s=st.floats(1e-3, 5.0),
    is_call=st.booleans(),
)
def test_round_trip_property(x, s, is_call):
    beta = float(np.exp(log_otm_value(x, s))) + (max(2 * np.sinh(x / 2), 0) if is_call else max(-2 * np.sinh(x / 2), 0))
    vega = float(np.exp(log_vega(-abs(x), s)))
    if not beta > 1e-300 or vega < 1e-280:
        return
    kappa = EPS * beta / (s * vega) + EPS
    if kappa > 1e-6:
        return
    recovered = float(invert_normalised(beta, x, is_call).total_vol)
    assert abs(recovered - s) / s <= 16 * kappa + 1e-15
