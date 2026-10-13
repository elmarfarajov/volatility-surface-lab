"""Dupire local volatility from the Day 4 surface, the forward equation that checks it, and smile dynamics."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volsurf.heston import HestonParams
from volsurf.localvol.dupire import implied_from_calls, local_variance, short_end_variance, solve_forward
from volsurf.localvol.dynamics import black_delta, sabr_delta, smile_moves
from volsurf.localvol.sabr import SABR, calibrate
from volsurf.market.chains import Chain
from volsurf.market.cleaning import prepare
from volsurf.market_data import synthetic_chain
from volsurf.smile.build import build_surface
from volsurf.smile.interpolation import Surface
from volsurf.smile.svi import RawSVI


@pytest.fixture(scope="module")
def heston_surface() -> Surface:
    params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(30, 72, 170, 352))
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
    chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
    return build_surface(prepare(chain)).surface


def test_a_flat_surface_and_the_smile_price_interpolation_puts_between_its_expiries():
    vol = 0.2
    slices = [RawSVI(vol**2 * t, 0.0, 0.0, 0.0, 0.1) for t in (0.25, 0.5, 1.0)]
    surface = Surface(np.array([0.25, 0.5, 1.0]), slices)
    k = np.linspace(-0.4, 0.3, 15)
    # before the first expiry (self-similar) and after the last (convolution) local volatility is the flat 20%
    for t in (0.1, 0.25, 2.0):
        np.testing.assert_allclose(np.sqrt(local_variance(surface, k, t)), vol, rtol=1e-6)
    # between expiries the price interpolation mixes two log-normal laws, which is not log-normal: at the money the
    # volatility stays 20%, in the wings a smile appears - 2.8 points at ln(K/F) = -0.4, a fifth of the way along
    middle = surface.implied_vol(k, 0.3)
    assert middle[np.argmin(np.abs(k))] == pytest.approx(vol, abs=2e-4)
    assert 0.02 < middle[0] - vol < 0.04


def test_local_variance_is_positive_and_its_short_end_continuous(heston_surface):
    k = np.linspace(-0.6, 0.3, 91)
    for t in np.geomspace(0.01, 1.5, 25):
        v = local_variance(heston_surface, k, float(t))
        assert np.all(np.isfinite(v)) and np.all(v > 0)
    t1 = float(heston_surface.maturities[0])
    # beyond the last expiry: the convolution's constant rate
    np.testing.assert_allclose(local_variance(heston_surface, k, 2.0), heston_surface.thetas[-1] / heston_surface.maturities[-1])
    # the interpolation's own short end is singular at the forward as t -> 0; the self-similar one is not
    atom = local_variance(heston_surface, np.array([1e-3]), 0.01 * t1, short_end="interpolation")[0]
    smooth = short_end_variance(heston_surface, np.array([1e-3]), 0.01 * t1)[0]
    assert atom > 50 * smooth


def test_the_forward_equation_returns_the_surface(heston_surface):
    targets = heston_surface.maturities[1:]
    sol = solve_forward(lambda k, t: local_variance(heston_surface, k, t), targets, space_steps=2400, steps_per_year=1000)
    for i, t in enumerate(targets):
        theta = float(heston_surface.thetas[i + 1])
        sel = (sol.k > -3 * np.sqrt(theta)) & (sol.k < 1.2 * np.sqrt(theta))
        error = implied_from_calls(sol.k[sel], sol.calls[i][sel], float(t)) - heston_surface.implied_vol(sol.k[sel], float(t))
        assert np.nanmax(np.abs(error)) < 5e-4  # 5 basis points of volatility


def test_local_volatility_moves_the_smile_against_the_index(heston_surface):
    t = float(heston_surface.maturities[1])
    k = np.log(np.linspace(0.85, 1.1, 26))
    sabr = calibrate(1.0, np.exp(k), t, heston_surface.implied_vol(k, t)).params
    shifts = np.log(np.array([0.97, 1.0, 1.03]))
    strikes = np.exp(np.linspace(-0.08, 0.06, 29))
    move = smile_moves(heston_surface, sabr, t, shifts, strikes, space_steps=1200)
    atm_lv = [np.interp(np.exp(s), strikes, row) for s, row in zip(shifts, move.local_vol, strict=True)]
    atm_sabr = [np.interp(np.exp(s), strikes, row) for s, row in zip(shifts, move.sabr, strict=True)]
    skew = float((heston_surface.implied_vol(np.array([0.01]), t) - heston_surface.implied_vol(np.array([-0.01]), t))[0] / 0.02)
    lv_slope = np.polyfit(shifts, atm_lv, 1)[0]
    assert skew < 0 and 1.5 < lv_slope / skew < 2.8  # about twice the implied skew (Hagan et al., 2002)
    assert abs(np.polyfit(shifts, atm_sabr, 1)[0]) < 0.3 * abs(skew)
    # deltas: Bartlett's below Hagan's when rho < 0, and below Black's
    p = SABR(0.2, 1.0, -0.6, 1.0)
    assert sabr_delta(1.0, 1.0, t, p) < black_delta(1.0, 1.0, t, 0.2) < sabr_delta(1.0, 1.0, t, p, bartlett=False)
