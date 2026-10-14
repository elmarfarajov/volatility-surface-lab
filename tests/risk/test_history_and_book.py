"""The historical backtest and spot-volatility regression on synthetic history, and the book's risk on a Heston surface."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volsurf.heston import HestonParams
from volsurf.market.chains import Chain
from volsurf.market.cleaning import prepare
from volsurf.market_data import synthetic_chain
from volsurf.risk import book as bk
from volsurf.risk import history
from volsurf.smile.build import SurfaceFit, build_surface


def _synthetic_history(beta: float = -1.0, days: int = 2000, sigma: float = 0.2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=days)
    ret = sigma * np.sqrt(1 / 252) * rng.standard_normal(days)
    noise = 0.004 * rng.standard_normal(days)
    vix = np.empty(days)
    level = 0.2
    for i in range(days):  # mean-reverting to 20%, each day's change moving with that day's return
        level = max(level + beta * ret[i] + noise[i] - 0.02 * (level - 0.2), 0.05)
        vix[i] = level
    spot = 1000 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"spot": spot, "vix": vix, "vix3m": 0.2 + 0.6 * (vix - 0.2), "rate": 0.02}, index=dates)


@pytest.fixture(scope="module")
def fit() -> SurfaceFit:
    params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(30, 91, 182, 365, 540))
    quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
    chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
    return build_surface(prepare(chain))


def test_the_backtest_matches_the_gamma_formula_on_paths():
    h = _synthetic_history()
    trades = history.backtest(h, every=5).table
    assert len(trades) > 300
    assert np.corrcoef(trades["pnl"], trades["predicted"])[0, 1] > 0.95
    # sold at the volatility the path has, on average it neither makes nor loses much
    assert abs(trades["pnl"].mean()) < 0.15 * trades["premium"].mean()


def test_spot_vol_regression_recovers_the_slope():
    h = _synthetic_history(beta=-1.2, days=4000)
    sv = history.spot_vol(h, window=252)
    assert sv.beta == pytest.approx(-1.2, abs=0.08)
    assert sv.term_ratio == pytest.approx(0.6, abs=1e-6)
    assert sv.rolling.notna().all()


def test_the_minimum_variance_delta_reduces_hedging_errors_when_vol_moves_with_spot():
    h = _synthetic_history(beta=-1.2, days=3000)
    errors = history.hedge_errors(h, warmup=252)
    assert errors["error_mv"].var() < 0.8 * errors["error_bs"].var()


def test_the_book_is_delta_hedged_and_the_dynamics_order_its_losses(fit):
    book = bk.build(fit)
    assert np.sum(book.contracts * book.greeks().delta) + book.hedge == pytest.approx(0.0, abs=1e-10)
    ladder = bk.stress_ladder(book, np.array([-0.1, 0.0, 0.05]), {"moneyness": 0.0, "strike": 1.0, "local": 2.0})
    # at no move the dynamics agree; the book is short skew, so a fall hurts more as the smile is held to strikes
    assert ladder.loc[0.0].nunique() == 1
    assert ladder.loc[-0.1, "moneyness"] > ladder.loc[-0.1, "strike"] > ladder.loc[-0.1, "local"]


def test_vol_shocks_by_expiry():
    shocks = bk.vol_shocks(np.array([0.05]), np.array([0.03]), np.array([10 / 365, 30 / 365, 91 / 365, 364 / 365]), alpha=0.5)
    np.testing.assert_allclose(shocks[0], [0.05, 0.05, 0.03, 0.03 * np.sqrt(91 / 364)])


def test_historical_risk_on_synthetic_history(fit):
    book = bk.build(fit)
    h = _synthetic_history(days=600)
    risk = bk.historical(book, h, horizon=1)
    assert risk.full.size == 599
    assert risk.var() > 0 and risk.es() >= risk.var(level=0.975)
    # quiet days: the Greeks explain the revaluation closely
    assert np.median(np.abs(risk.full - risk.greeks)) < 0.1 * np.median(np.abs(risk.full))
