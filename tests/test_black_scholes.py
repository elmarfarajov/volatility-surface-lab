import numpy as np
import pytest

from volsurf.black_scholes import black76_price, black76_vega, bsm_greeks, bsm_price


def test_textbook_values():
    assert bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, True) == pytest.approx(10.450584, abs=1e-6)
    assert bsm_price(100, 100, 1.0, 0.05, 0.0, 0.2, False) == pytest.approx(5.573526, abs=1e-6)


def test_put_call_parity_on_random_contracts():
    rng = np.random.default_rng(0)
    n = 2000
    s, k = rng.uniform(20, 200, n), rng.uniform(20, 200, n)
    t, r, q, v = rng.uniform(0.01, 5, n), rng.uniform(-0.02, 0.1, n), rng.uniform(0, 0.06, n), rng.uniform(0.02, 1.5, n)
    lhs = bsm_price(s, k, t, r, q, v, True) - bsm_price(s, k, t, r, q, v, False)
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    np.testing.assert_allclose(lhs, rhs, atol=1e-9)


def test_black76_matches_spot_formulation():
    s, k, t, r, q, v = 250.0, 260.0, 0.8, 0.04, 0.015, 0.31
    forward = s * np.exp((r - q) * t)
    for is_call in (True, False):
        assert black76_price(forward, k, t, np.exp(-r * t), v, is_call) == pytest.approx(
            bsm_price(s, k, t, r, q, v, is_call), rel=1e-13
        )


def test_zero_volatility_gives_discounted_intrinsic():
    assert black76_price(110.0, 100.0, 1.0, 0.95, 0.0, True) == pytest.approx(9.5)
    assert black76_price(110.0, 100.0, 1.0, 0.95, 0.0, False) == pytest.approx(0.0)


def test_deep_otm_put_is_not_lost_to_cancellation():
    price = black76_price(100.0, 40.0, 0.25, 1.0, 0.2, False)
    assert 0.0 < price < 1e-15


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    s, k, t, r, q, v = 105.0, 100.0, 0.7, 0.03, 0.01, 0.27
    g = bsm_greeks(s, k, t, r, q, v, is_call)

    def p(**kw):
        args = dict(spot=s, strike=k, maturity=t, rate=r, dividend_yield=q, vol=v)
        args.update(kw)
        return bsm_price(
            args["spot"], args["strike"], args["maturity"], args["rate"], args["dividend_yield"], args["vol"], is_call
        )

    h = 1e-4
    assert g.delta == pytest.approx((p(spot=s + h) - p(spot=s - h)) / (2 * h), abs=1e-7)
    assert g.gamma == pytest.approx((p(spot=s + h) - 2 * p() + p(spot=s - h)) / h**2, abs=1e-4)
    assert g.vega == pytest.approx((p(vol=v + h) - p(vol=v - h)) / (2 * h), abs=1e-6)
    assert g.rho == pytest.approx((p(rate=r + h) - p(rate=r - h)) / (2 * h), abs=1e-6)
    assert g.theta == pytest.approx(-(p(maturity=t + h) - p(maturity=t - h)) / (2 * h), abs=1e-5)

    delta_up = bsm_greeks(s, k, t, r, q, v + h, is_call).delta
    delta_dn = bsm_greeks(s, k, t, r, q, v - h, is_call).delta
    assert g.vanna == pytest.approx((delta_up - delta_dn) / (2 * h), abs=1e-6)
    vega_up = bsm_greeks(s, k, t, r, q, v + h, is_call).vega
    vega_dn = bsm_greeks(s, k, t, r, q, v - h, is_call).vega
    assert g.volga == pytest.approx((vega_up - vega_dn) / (2 * h), abs=1e-4)


def test_black76_vega_consistent_with_bsm_vega():
    s, k, t, r, q, v = 100.0, 95.0, 0.5, 0.02, 0.0, 0.25
    forward = s * np.exp((r - q) * t)
    assert black76_vega(forward, k, t, np.exp(-r * t), v) == pytest.approx(bsm_greeks(s, k, t, r, q, v).vega, rel=1e-12)
