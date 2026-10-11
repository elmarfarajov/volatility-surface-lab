"""The Heston characteristic function, its moment strip, and the contour engine against independent references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from volsurf.gallery.heston import CASES, _quantlib, reference
from volsurf.stochvol.characteristic import Heston, cf, cumulants, explosion_time, moment_strip
from volsurf.stochvol.pricing import carr_madan_fft, cos_otm_price, optimal_alpha, otm_price, price

FO = Heston(0.0175, 1.5768, 0.0398, 0.5751, -0.5711)


def test_characteristic_function_basics():
    h = FO
    assert cf(np.array([0.0]), 1.0, h)[0] == pytest.approx(1.0)
    # E[S_T / F_T] = phi(-i) = 1: the forward is the mean
    assert cf(np.array([-1j]), 1.0, h)[0] == pytest.approx(1.0, abs=1e-14)
    # continuity for long maturities (the little trap): no jumps in the phase along the real line
    u = np.linspace(0, 60, 20001)
    phase = np.unwrap(np.angle(cf(u, 30.0, h)))
    assert np.max(np.abs(np.diff(phase))) < 0.05


def test_cumulants_match_the_characteristic_function():
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = Heston(
            rng.uniform(0.005, 0.2), rng.uniform(0.1, 6), rng.uniform(0.01, 0.2), rng.uniform(0.1, 1.5), rng.uniform(-0.95, 0.5)
        )
        t = float(10 ** rng.uniform(-2, 1))
        c1, c2 = cumulants(t, h)
        e = 1e-4
        log_phi = np.log(cf(np.array([-e, e]), t, h))
        assert c1 == pytest.approx(((log_phi[1] - log_phi[0]) / (2 * e) / 1j).real, rel=1e-6)
        assert c2 > 0


def test_explosion_time_inverts_the_strip_and_tends_to_infinity_at_the_boundary():
    h = Heston(0.04, 1.0, 0.04, 1.0, -0.7)
    for t in (0.2, 1.0, 5.0):
        lo, hi = moment_strip(t, h)
        assert lo < 0 < 1 < hi
        assert explosion_time(hi, h) == pytest.approx(t, rel=1e-6)
        assert explosion_time(lo, h) == pytest.approx(t, rel=1e-6)
    # the strip narrows with maturity
    assert moment_strip(5.0, h)[1] < moment_strip(1.0, h)[1]
    # as D rises to zero with k > 0 the explosion time grows without bound (the form recalled first did not)
    g = Heston(0.04, 3.0, 0.04, 0.3, -0.7)
    p_star = 35.0128  # D = 0 here, k > 0
    assert explosion_time(p_star + 0.05, g) > explosion_time(p_star + 0.5, g) > explosion_time(p_star + 5, g)
    assert explosion_time(p_star + 1e-4, g) > 100


@pytest.mark.parametrize("case", range(3))
def test_contour_engine_matches_40_digits_in_both_wings(case):
    _, h, t = CASES[case]
    spread = 4.0 * np.sqrt(max(h.theta * t, h.v0 * t))
    k = np.array([-1.5, -0.4, 0.0, 0.4, 1.2]) * spread
    exact = np.array([reference(float(x), t, h) for x in k])
    np.testing.assert_allclose(otm_price(k, t, h), exact, rtol=1e-10)
    np.testing.assert_allclose(otm_price(k, t, h, per_strike=True), exact, rtol=1e-10)


def test_the_damping_does_not_matter_inside_the_strip():
    k, t, h = -0.08, 1.0, FO
    exact = reference(k, t, h)
    import volsurf.stochvol.pricing as pricing

    original = pricing.optimal_alpha
    try:
        for alpha in (-1.05, -1.5, -3.0, -4.5):
            pricing.optimal_alpha = lambda *args, a=alpha: a  # type: ignore[assignment]
            assert pricing.otm_price(np.array([k]), t, h)[0] == pytest.approx(exact, rel=1e-13)
    finally:
        pricing.optimal_alpha = original
    assert optimal_alpha(0.1, t, h) > 0 and optimal_alpha(-0.1, t, h) < -1


def test_put_call_parity_and_discounting():
    forward, discount, t = 100.0, 0.95, 0.75
    strikes = np.array([70.0, 95.0, 100.0, 110.0, 140.0])
    calls = price(forward, strikes, t, FO, discount, True)
    puts = price(forward, strikes, t, FO, discount, False)
    np.testing.assert_allclose(calls - puts, discount * (forward - strikes), rtol=1e-12, atol=1e-12)


def test_agreement_with_quantlib_and_scipy_quadrature():
    k = np.linspace(-0.3, 0.2, 6)
    np.testing.assert_allclose(otm_price(k, 1.0, FO), _quantlib(k, 1.0, FO), rtol=1e-11)
    # an independent quadrature of the same contour integral at one damping
    kk, alpha = 0.1, 1.5

    def integrand(u: float) -> float:
        z = u - 1j * (alpha + 1.0)
        return float(
            (np.exp(-1j * u * kk) * cf(np.array([z]), 1.0, FO)[0] / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)).real
        )

    value = np.exp(-alpha * kk) / np.pi * integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-15, epsrel=1e-13)[0]
    assert otm_price(np.array([kk]), 1.0, FO)[0] == pytest.approx(value, rel=1e-11)


def test_the_comparison_engines_are_what_they_claim():
    # textbook COS (256 terms, L = 12) is good to about 1e-5 near the money and loses out-of-the-money calls
    k = np.array([0.0, 0.05])
    np.testing.assert_allclose(cos_otm_price(k, 1.0, FO), otm_price(k, 1.0, FO), rtol=1e-4)
    deep = np.array([0.9])
    assert abs(cos_otm_price(deep, 1.0, FO)[0] / otm_price(deep, 1.0, FO)[0] - 1) > 1e-3
    # the FFT is exact on its own grid, to its Simpson rule, near the money
    grid, calls = carr_madan_fft(1.0, FO)
    i = int(np.flatnonzero(grid >= 0)[0])
    assert calls[i] == pytest.approx(otm_price(np.array([grid[i]]), 1.0, FO)[0], rel=1e-6)
