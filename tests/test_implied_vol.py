import numpy as np
import pytest

from volsurf.black_scholes import black76_price, bsm_price
from volsurf.implied_vol import implied_vol_black76, implied_vol_bsm


@pytest.mark.parametrize("maturity", [1 / 365, 7 / 365, 0.25, 1.0, 5.0])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_across_the_surface(maturity, vol, is_call):
    forward, discount = 100.0, 0.97
    strikes = forward * np.exp(np.linspace(-1.5, 1.5, 61))
    prices = black76_price(forward, strikes, maturity, discount, vol, is_call)
    otm_value = black76_price(forward, strikes, maturity, discount, vol, strikes >= forward) / (discount * forward)
    recoverable = otm_value > 1e-10
    iv = implied_vol_black76(prices, forward, strikes, maturity, discount, is_call)
    assert not np.isnan(iv[recoverable]).any()
    np.testing.assert_allclose(iv[recoverable], vol, atol=2e-7)


def test_scalar_inputs_return_scalar_shape():
    iv = implied_vol_bsm(bsm_price(100, 100, 1, 0.03, 0.01, 0.23), 100, 100, 1, 0.03, 0.01, True)
    assert np.shape(iv) == ()
    assert float(iv) == pytest.approx(0.23, abs=1e-10)


def test_arbitrage_violating_quotes_return_nan():
    forward, strike, t, d = 100.0, 90.0, 0.5, 0.98
    below_intrinsic = d * (forward - strike) - 0.01
    above_upper = d * forward + 1.0
    iv = implied_vol_black76([below_intrinsic, above_upper, -1.0], forward, strike, t, d, True)
    assert np.isnan(iv).all()


def test_in_the_money_quotes_use_parity():
    forward, t, d, vol = 100.0, 0.75, 0.96, 0.33
    strikes = np.array([60.0, 80.0, 120.0, 150.0])
    itm_is_call = strikes < forward
    prices = black76_price(forward, strikes, t, d, vol, itm_is_call)
    np.testing.assert_allclose(implied_vol_black76(prices, forward, strikes, t, d, itm_is_call), vol, atol=1e-9)


def test_large_chain_is_fast():
    import time

    rng = np.random.default_rng(1)
    strikes = 100 * np.exp(rng.uniform(-0.4, 0.4, 50_000))
    prices = black76_price(100.0, strikes, 0.5, 1.0, 0.3, True)
    start = time.perf_counter()
    iv = implied_vol_black76(prices, 100.0, strikes, 0.5, 1.0, True)
    assert time.perf_counter() - start < 2.0
    np.testing.assert_allclose(iv, 0.3, atol=1e-8)
