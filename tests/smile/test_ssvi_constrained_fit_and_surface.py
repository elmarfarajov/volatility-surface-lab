"""SSVI and eSSVI, the constrained SVI fit, densities, and the price-interpolated surface."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volsurf.heston import HestonParams
from volsurf.market.black import normalised_black
from volsurf.market.chains import Chain
from volsurf.market.cleaning import prepare
from volsurf.market_data import synthetic_chain
from volsurf.smile.build import build_surface
from volsurf.smile.density import call_price, density, moments
from volsurf.smile.fit import fit_slice
from volsurf.smile.interpolation import Surface
from volsurf.smile.ssvi import SSVI, Quotes, SSVISlice, calendar_margin, fit_essvi, fit_ssvi, psi_bounds
from volsurf.smile.svi import RawSVI, butterfly_check, calendar_check


@pytest.fixture(scope="module")
def heston_surface():
    params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(23, 72, 170, 352, 716))
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
    chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
    return build_surface(prepare(chain))


def test_an_ssvi_slice_is_the_raw_svi_it_claims():
    s = SSVISlice(0.04, -0.6, 0.3)
    k = np.linspace(-2, 2, 101)
    np.testing.assert_allclose(s.raw().total_variance(k), s.total_variance(k), rtol=1e-13)
    assert s.total_variance(0.0) == pytest.approx(0.04)


def test_the_ssvi_theorem_holds_on_the_whole_line():
    rng = np.random.default_rng(2)
    for _ in range(25):
        rho = rng.uniform(-0.95, 0.95)
        surface = SSVI(
            rho,
            2 * rng.uniform(0.05, 1) / (1 + abs(rho)),
            rng.uniform(0.01, 0.5),
            np.arange(1.0, 5.0),
            np.cumsum(rng.uniform(1e-4, 0.05, 4)),
        )
        raw = [s.raw() for s in surface.slices()]
        assert all(butterfly_check(r).passed for r in raw)
        assert all(calendar_check(a, b).passed for a, b in zip(raw[:-1], raw[1:], strict=True))


def test_ssvi_and_essvi_recover_a_surface_they_can_represent():
    true = SSVI(-0.6, 1.2, 0.4, np.array([0.1, 0.3, 0.7, 1.5]), np.array([0.004, 0.012, 0.028, 0.06]))
    quotes = []
    for i, t in enumerate(true.maturities):
        k = np.linspace(-0.5, 0.3, 25)
        quotes.append(Quotes(float(t), k, np.sqrt(true.slice(i).total_variance(k) / t), np.ones(k.size)))
    fitted = fit_ssvi(quotes)
    assert fitted.rho == pytest.approx(-0.6, abs=1e-4) and fitted.gamma == pytest.approx(0.4, abs=1e-3)
    slices = fit_essvi(quotes)
    for s, q in zip(slices, quotes, strict=True):
        assert np.max(np.abs(np.sqrt(s.total_variance(q.k) / q.maturity) - q.iv)) < 1e-4
        assert s.butterfly_margin() >= 0
    assert all(calendar_margin(a, b) >= -1e-12 for a, b in zip(slices[:-1], slices[1:], strict=True))


def test_psi_bounds_hold_the_conditions():
    previous = SSVISlice(0.01, -0.5, 0.15)
    low, high = psi_bounds(-0.4, 0.02, previous)
    assert low <= high
    for psi in (low, high):
        s = SSVISlice(0.02, -0.4, psi)
        assert s.butterfly_margin() >= -1e-12 and calendar_margin(previous, s) >= -1e-12


def test_the_constrained_fit_repairs_vogts_slice():
    vogt = RawSVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    k = np.linspace(-1.5, 1.5, 31)
    quotes = Quotes(1.0, k, vogt.implied_vol(k, 1.0), np.ones(k.size))
    fit = fit_slice(quotes, fit_essvi([quotes])[0].raw())
    assert fit.butterfly_margin >= 0 and butterfly_check(fit.slice).passed
    assert fit.rmse < 0.01


def test_a_heston_chain_gives_an_arbitrage_free_surface(heston_surface):
    fit = heston_surface
    assert len(fit.svi) == 5
    assert all(f.butterfly_margin >= 0 for f in fit.svi)
    assert all(f.calendar_margin >= 0 for f in fit.svi[1:])
    assert max(f.rmse for f in fit.svi) < 0.01
    for f in fit.svi:
        m = moments(f.slice)
        assert abs(m.mass - 1) < 1e-10 and abs(m.forward - 1) < 1e-10 and m.breeden_litzenberger < 1e-4


def test_price_interpolation_is_free_of_arbitrage_and_exact_at_the_nodes(heston_surface):
    surface = heston_surface.surface
    k = np.linspace(-0.8, 0.5, 651)
    h = k[1] - k[0]
    previous = None
    for t in np.linspace(0.01, surface.maturities[-1] * 2, 80):
        c = surface.call(k, t)
        dens = np.exp(-k[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2 * h))
        assert dens.min() > -1e-8
        if previous is not None:
            assert np.min(c - previous) > -1e-12
        previous = c
    for t, s in zip(surface.maturities, heston_surface.svi, strict=True):
        np.testing.assert_allclose(surface.implied_vol(k, t), s.slice.implied_vol(k, t), rtol=1e-9)
        np.testing.assert_allclose(surface.density(k, t), density(s.slice, k), rtol=1e-12, atol=1e-300)
    t_mid = 0.5 * (surface.maturities[1] + surface.maturities[2])
    c = surface.call(k, t_mid)
    numeric = np.exp(-k[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2 * h))
    exact = surface.density(k[1:-1], t_mid)
    assert np.max(np.abs(numeric - exact)) < 1e-3 * np.max(exact)  # central differences, h = 0.002: O(h^2)


def test_extrapolation_is_a_lognormal_convolution():
    last = RawSVI(0.03, 0.1, -0.5, 0.05, 0.2)
    surface = Surface(np.array([1.0]), [last])
    k = np.linspace(-0.5, 0.5, 11)
    extra = 0.02  # theta grows by 2% of total variance from t = 1 to t = 1 + 0.02 / theta_1
    t = 1.0 + extra / float(last.total_variance(0.0))
    # brute force: integrate e^z c_1(k - z) against the normal density of z on a fine grid
    z = np.linspace(-0.5 * extra - 10 * np.sqrt(extra), -0.5 * extra + 10 * np.sqrt(extra), 20001)
    pdf = np.exp(-0.5 * (z + 0.5 * extra) ** 2 / extra) / np.sqrt(2 * np.pi * extra)
    brute = [np.trapezoid(pdf * np.exp(z) * call_price(last.total_variance(kk - z), kk - z), z) for kk in k]
    np.testing.assert_allclose(surface.call(k, t), brute, rtol=1e-9)


def test_a_falling_atm_variance_is_refused():
    with pytest.raises(ValueError):
        Surface(np.array([0.5, 1.0]), [RawSVI(0.03, 0.1, -0.5, 0.0, 0.2), RawSVI(0.02, 0.1, -0.5, 0.0, 0.2)])


def test_call_price_matches_the_black_function():
    w, k = np.array([0.04, 0.2]), np.array([-0.1, 0.3])
    np.testing.assert_allclose(call_price(w, k), np.exp(k / 2) * normalised_black(-k, np.sqrt(w)))


def test_a_bad_slsqp_exit_never_replaces_a_good_start():
    # nine days out, SLSQP once ended on "constraints incompatible" at a point that passed the arbitrage
    # checks but missed the quotes by 23 volatility points; it was accepted, and broke the next slice
    params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(9, 23))
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
    chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
    fit = build_surface(prepare(chain))
    first, start = fit.svi[0], fit.essvi[0]
    start_rmse = float(
        np.sqrt(np.mean((np.sqrt(start.total_variance(fit.quotes[0].k) / first.maturity) - fit.quotes[0].iv) ** 2))
    )
    assert first.rmse <= start_rmse * 1.5 and first.rmse < 0.01
    assert fit.svi[1].calendar_margin >= 0
