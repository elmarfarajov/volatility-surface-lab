import pytest

from volsurf.black_scholes import bsm_price
from volsurf.lattice import binomial_price

S, K, T, R, Q, VOL = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25


@pytest.mark.parametrize("is_call", [True, False])
def test_leisen_reimer_converges_to_black_scholes(is_call):
    exact = bsm_price(S, K, T, R, Q, VOL, is_call)
    assert binomial_price(S, K, T, R, Q, VOL, is_call, steps=201) == pytest.approx(exact, abs=5e-5)


def test_leisen_reimer_beats_crr_with_ten_times_fewer_steps():
    exact = bsm_price(S, K, T, R, Q, VOL, True)
    lr_error = abs(binomial_price(S, K, T, R, Q, VOL, True, steps=101) - exact)
    crr_error = abs(binomial_price(S, K, T, R, Q, VOL, True, steps=1010, method="crr") - exact)
    assert lr_error < crr_error / 10


def test_even_step_count_is_made_odd_for_leisen_reimer():
    assert binomial_price(S, K, T, R, Q, VOL, True, steps=200) == binomial_price(S, K, T, R, Q, VOL, True, steps=201)


def test_american_call_without_dividends_equals_european():
    european = bsm_price(S, 110.0, T, R, 0.0, VOL, True)
    assert binomial_price(S, 110.0, T, R, 0.0, VOL, True, american=True, steps=501) == pytest.approx(european, abs=1e-4)


def test_american_put_benchmark():
    value = binomial_price(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False, american=True, steps=2001)
    assert value == pytest.approx(4.478, abs=0.01)
    assert value > bsm_price(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False)


def test_expired_option_returns_intrinsic():
    assert binomial_price(105.0, 100.0, 0.0, R, Q, VOL, True) == 5.0
    assert binomial_price(105.0, 100.0, 0.0, R, Q, VOL, False) == 0.0


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, Q, VOL, method="trinomial")
