"""Forwards, discount factors and cleaning on synthetic chains whose truth is known."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volsurf.market import black_price
from volsurf.market.chains import Chain
from volsurf.market.cleaning import RULES, executable_violations, mid_violations, prepare, strike_arbitrage
from volsurf.market.parity import fit_curve, fit_forward

SPOT = 5000.0


def _smile(k: np.ndarray, t: float) -> np.ndarray:
    return 0.18 - 0.12 * k / np.sqrt(t) * 0.25 + 0.05 * (k / np.sqrt(t)) ** 2 * 0.25


def _expiry(t: float, rate: float, carry: float, stale: tuple[float, ...] = (), seed: int = 0) -> pd.DataFrame:
    """Calls and puts on one expiry, quoted on a 0.05 tick around Black prices with a smile."""
    rng = np.random.default_rng(seed)
    forward, discount = SPOT * np.exp((rate - carry) * t), np.exp(-rate * t)
    strikes = np.arange(np.round(forward * 0.7, -1), forward * 1.3, 10.0)
    k = np.log(strikes / forward)
    rows = []
    for is_call in (True, False):
        price = black_price(forward, strikes, t, _smile(k, t), discount, is_call)
        half = np.maximum(0.05, 0.01 * price) * (1.0 + 0.3 * rng.random(strikes.size))
        bid = np.floor((price - half) / 0.05) * 0.05
        ask = np.ceil((price + half) / 0.05) * 0.05
        for strike, b, a in zip(strikes, bid, ask, strict=True):
            if strike in stale and (strike < forward) != is_call:  # a stale out-of-the-money quote, far off its neighbours
                b, a = 1.6 * b + 5.0, 1.6 * a + 5.0
            rows.append({"strike": strike, "is_call": is_call, "bid": max(b, 0.0), "ask": a})
    frame = pd.DataFrame(rows)
    frame["expiry"] = f"T{t:.3f}"
    frame["root"] = "SYN"
    frame["maturity"] = t
    frame["last"] = 0.5 * (frame["bid"] + frame["ask"])
    frame["last_trade"] = pd.Timestamp("2026-10-09", tz="UTC")
    frame["volume"] = 10.0
    frame["open_interest"] = 100.0
    frame["vendor_iv"] = np.nan
    return frame


def _fit(frame: pd.DataFrame, **kwargs):
    t = float(frame["maturity"].iloc[0])
    calls = frame[frame["is_call"]].set_index("strike")
    puts = frame[~frame["is_call"]].set_index("strike")
    k = calls.index.to_numpy(float)
    return fit_forward(
        k, calls["bid"].to_numpy(), calls["ask"].to_numpy(), puts["bid"].to_numpy(), puts["ask"].to_numpy(), SPOT, t, **kwargs
    )


def test_parity_recovers_forward_and_discount_within_their_errors():
    rate, carry, t = 0.045, 0.012, 0.5
    fit = _fit(_expiry(t, rate, carry))
    true_forward, true_discount = SPOT * np.exp((rate - carry) * t), np.exp(-rate * t)
    assert abs(fit.forward - true_forward) < max(4 * fit.forward_se, 0.05)
    assert abs(fit.discount - true_discount) < max(4 * fit.discount_se, 2e-5)
    assert fit.inside > 0.95
    assert fit.rate == pytest.approx(rate, abs=5e-4)
    assert fit.carry == pytest.approx(carry, abs=5e-4)


def test_parity_is_not_pulled_by_stale_quotes_far_from_the_money():
    frame = _expiry(0.5, 0.045, 0.012)
    # a block of 44 stale puts deep out of the money, 80 points (about seven half-widths of their pairs) too high;
    # at 40 points - under four half-widths - the block is admitted and moves the forward by two points (see the notes)
    deep = frame["strike"] < 4000
    frame.loc[deep & ~frame["is_call"], ["bid", "ask"]] += 80.0
    fit = _fit(frame)
    assert abs(fit.forward - SPOT * np.exp(0.033 * 0.5)) < 0.5
    assert not fit.used[np.flatnonzero(frame[frame["is_call"]]["strike"].to_numpy() < 4000)].any()


def test_a_known_discount_factor_is_held_fixed():
    fit = _fit(_expiry(0.5, 0.045, 0.012), discount=np.exp(-0.045 * 0.5))
    assert fit.discount == pytest.approx(np.exp(-0.045 * 0.5)) and fit.discount_se == 0.0
    assert fit.forward == pytest.approx(SPOT * np.exp(0.033 * 0.5), abs=0.1)


def test_too_few_pairs_is_an_error():
    frame = _expiry(0.5, 0.045, 0.012)
    frame = frame[frame["strike"].isin(sorted(frame["strike"].unique())[:3])]
    with pytest.raises(ValueError, match="pairs"):
        _fit(frame)


def test_the_curve_through_many_expiries_recovers_the_rate():
    fits = [_fit(_expiry(t, 0.04 + 0.01 * t, 0.01, seed=i)) for i, t in enumerate((0.25, 0.5, 1.0, 1.5, 2.0))]
    curve = fit_curve(fits, min_pairs=5)
    for t in (0.5, 1.0, 2.0):
        assert float(curve.rate(t)) == pytest.approx(0.04 + 0.01 * t, abs=1e-3)


def test_executable_violations_find_a_tradeable_butterfly_and_ignore_mid_noise():
    k = np.array([100.0, 105.0, 110.0])
    # mids 10, 5.2, 0.1: a convexity wiggle inside the spread is not arbitrage
    assert executable_violations(k, np.array([9.9, 5.0, 0.0]), np.array([10.1, 5.4, 0.2]), 1.0) == []
    found = executable_violations(k, np.array([9.9, 6.0, 0.0]), np.array([10.1, 6.2, 0.2]), 1.0)
    # the butterfly, and the 105-110 call spread sold at 6.0 for a payoff of at most 5
    expected = {(0, 1, 2): (6.0 * 10 - 10.1 * 5 - 0.2 * 5) / 10, (1, 2): 0.8}
    assert dict(found) == pytest.approx(expected)


def test_one_stale_quote_does_not_take_its_neighbours_with_it():
    # the regression behind the removal rule: blame by count removed honest wings one after another
    k = np.arange(100.0, 200.0, 5.0)
    mid = 0.002 * (200.0 - k) ** 2
    mid[10] -= 6.0  # one stale quote, far below the curve
    removed = strike_arbitrage(k, mid - 0.05, mid + 0.05, 1.0)
    assert list(removed) == [10]


def test_prepare_cleans_a_whole_chain_and_accounts_for_every_quote():
    stale = (4600.0, 5300.0)
    frames = [_expiry(t, 0.045, 0.012, stale=stale, seed=i) for i, t in enumerate((0.1, 0.25, 0.5, 1.0))]
    chain = Chain("SYN", SPOT, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), pd.concat(frames, ignore_index=True))
    market = prepare(chain)
    assert len(market.slices) == 4 and not market.skipped
    for s in market.slices:
        assert sum(s.funnel[rule] for rule in RULES) + s.funnel["kept"] == s.funnel["raw"]
        assert not np.isin(stale, s.quotes["strike"]).any()  # the stale quotes are gone
        assert s.funnel["strike arbitrage"] <= 2 * len(stale)  # and little else
        true = _smile(s.quotes["k"].to_numpy(), s.maturity)
        assert np.median(np.abs(s.quotes["iv_mid"] - true)) < 0.003
        assert mid_violations(s.quotes, s.forward, s.discount) >= 0
    ts = market.term_structure()
    assert np.allclose(ts["curve_rate"], 0.045, atol=1e-3)
    assert market.funnel()["kept"] == sum(len(s.quotes) for s in market.slices)
