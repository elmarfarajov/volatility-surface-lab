import numpy as np
import pytest

from volsurf.black_scholes import bsm_price
from volsurf.heston import (
    HestonParams,
    heston_cf,
    heston_cos_sensitivities,
    heston_price_cos,
    heston_price_cos_spot,
    heston_price_integration,
)

FANG_OOSTERLEE = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)


def test_characteristic_function_is_normalised_and_martingale():
    assert heston_cf(np.array([0.0]), 1.0, FANG_OOSTERLEE)[0] == pytest.approx(1.0)
    assert heston_cf(np.array([-1j]), 1.0, FANG_OOSTERLEE)[0] == pytest.approx(1.0, abs=1e-12)


def test_cos_matches_published_reference():
    assert heston_price_cos(100.0, [100.0], 1.0, 1.0, FANG_OOSTERLEE)[0] == pytest.approx(5.785155450, abs=1e-6)


def test_integration_matches_published_reference():
    assert heston_price_integration(100.0, 100.0, 1.0, 1.0, FANG_OOSTERLEE) == pytest.approx(5.785155450, abs=1e-6)


def test_cos_agrees_with_quadrature_on_random_models():
    rng = np.random.default_rng(1)
    for _ in range(12):
        p = HestonParams(
            rng.uniform(0.005, 0.3), rng.uniform(0.3, 8), rng.uniform(0.01, 0.3), rng.uniform(0.1, 1.5), rng.uniform(-0.95, 0.3)
        )
        maturity = float(rng.choice([0.02, 0.25, 1.0, 3.0]))
        strikes = 100 * np.exp(np.linspace(-0.6, 0.4, 7) * max(np.sqrt(maturity), 0.3))
        cos = heston_price_cos(100.0, strikes, maturity, 0.95, p, False)
        quad = [heston_price_integration(100.0, k, maturity, 0.95, p, False) for k in strikes]
        np.testing.assert_allclose(cos, quad, atol=1e-5)


def test_put_call_parity_holds():
    strikes = np.linspace(70, 140, 15)
    call = heston_price_cos(100.0, strikes, 0.5, 0.98, FANG_OOSTERLEE, True)
    put = heston_price_cos(100.0, strikes, 0.5, 0.98, FANG_OOSTERLEE, False)
    np.testing.assert_allclose(call - put, 0.98 * (100.0 - strikes), atol=1e-10)


def test_vanishing_vol_of_vol_recovers_black_scholes():
    vol = 0.22
    p = HestonParams(v0=vol**2, kappa=1.0, theta=vol**2, sigma=1e-4, rho=0.0)
    strikes = np.array([80.0, 100.0, 125.0])
    heston = heston_price_cos_spot(100.0, strikes, 1.0, 0.03, 0.01, p, True)
    np.testing.assert_allclose(heston, bsm_price(100.0, strikes, 1.0, 0.03, 0.01, vol, True), atol=1e-5)


def test_negative_correlation_creates_downside_skew():
    from volsurf.implied_vol import implied_vol_black76

    strikes = np.array([80.0, 120.0])
    prices = heston_price_cos(100.0, strikes, 0.5, 1.0, FANG_OOSTERLEE, strikes >= 100.0)
    iv = implied_vol_black76(prices, 100.0, strikes, 0.5, 1.0, strikes >= 100.0)
    assert iv[0] > iv[1]


def test_path_sensitivities_match_finite_differences():
    s, v, k, t, r, q = 100.0, 0.04, 105.0, 0.5, 0.03, 0.01
    p = HestonParams(v, 2.0, 0.05, 0.6, -0.7)
    sens = heston_cos_sensitivities(np.array([s, 90.0]), np.array([v, 0.09]), k, t, r, q, p.kappa, p.theta, p.sigma, p.rho, True)

    def price(spot, var):
        return heston_price_integration(
            spot * np.exp((r - q) * t), k, t, np.exp(-r * t), HestonParams(var, p.kappa, p.theta, p.sigma, p.rho)
        )

    assert sens.price[0] == pytest.approx(price(s, v), abs=1e-5)
    assert sens.price[1] == pytest.approx(price(90.0, 0.09), abs=1e-5)
    assert sens.delta[0] == pytest.approx((price(s + 1e-3, v) - price(s - 1e-3, v)) / 2e-3, abs=1e-5)
    assert sens.dprice_dv[0] == pytest.approx((price(s, v + 1e-6) - price(s, v - 1e-6)) / 2e-6, rel=1e-4)


def test_parameter_validation():
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-1.0)
    with pytest.raises(ValueError):
        HestonParams(v0=-0.01, kappa=1.0, theta=0.04, sigma=0.5, rho=0.0)
    p = HestonParams(0.04, 2.0, 0.05, 0.5, -0.5)
    assert p.feller_ratio == pytest.approx(0.8)
    assert HestonParams.from_array(p.as_array()) == p
