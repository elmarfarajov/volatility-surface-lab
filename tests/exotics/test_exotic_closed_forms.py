"""Closed forms for barriers, geometric Asians, lookbacks and digitals, against QuantLib and against their own limits."""

from __future__ import annotations

import numpy as np
import pytest
import QuantLib as ql

from volsurf.exotics import closed_form as cf

TODAY = ql.Date(14, 10, 2026)
DC = ql.Actual365Fixed()
KINDS = {"down-in": ql.Barrier.DownIn, "down-out": ql.Barrier.DownOut, "up-in": ql.Barrier.UpIn, "up-out": ql.Barrier.UpOut}


def _process(s: float, r: float, q: float, sigma: float) -> ql.BlackScholesMertonProcess:
    ql.Settings.instance().evaluationDate = TODAY
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, q, DC)),
        ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, DC)),
    )


def _cases(n: int, seed: int) -> list[tuple[float, int, float, float, float, bool]]:
    rng = np.random.default_rng(seed)
    return [
        (
            float(rng.uniform(70, 130)),
            int(rng.integers(30, 730)),
            float(rng.uniform(0, 0.08)),
            float(rng.uniform(0, 0.05)),
            float(rng.uniform(0.1, 0.6)),
            bool(rng.random() < 0.5),
        )
        for _ in range(n)
    ]


@pytest.mark.parametrize("kind", list(KINDS))
def test_reiner_rubinstein_barriers_match_quantlib_with_and_without_rebate(kind):
    rng = np.random.default_rng(list(KINDS).index(kind))
    for x, days, r, q, sigma, call in _cases(40, 7):
        h = float(rng.uniform(60, 99)) if kind.startswith("down") else float(rng.uniform(101, 150))
        rebate = float(rng.choice([0.0, 3.0]))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, x)
        option = ql.BarrierOption(KINDS[kind], h, rebate, payoff, ql.EuropeanExercise(TODAY + days))
        option.setPricingEngine(ql.AnalyticBarrierEngine(_process(100.0, r, q, sigma)))
        ours = cf.barrier(kind, 100.0, x, h, days / 365, r, r - q, sigma, call, rebate)  # type: ignore[arg-type]
        assert ours == pytest.approx(option.NPV(), abs=1e-11)


def test_knock_in_plus_knock_out_is_the_vanilla():
    for x, days, r, q, sigma, call in _cases(30, 8):
        t, b = days / 365, r - q
        vanilla = cf.black_scholes(100.0, x, t, r, b, sigma, call)
        for down, h in ((True, 85.0), (False, 115.0)):
            pair = ("down-in", "down-out") if down else ("up-in", "up-out")
            total = sum(cf.barrier(k, 100.0, x, h, t, r, b, sigma, call) for k in pair)  # type: ignore[arg-type]
            assert total == pytest.approx(vanilla, abs=1e-11)


def test_a_barrier_already_reached_is_refused():
    with pytest.raises(ValueError):
        cf.barrier("down-out", 100.0, 100.0, 105.0, 1.0, 0.05, 0.03, 0.2)


def test_geometric_asians_match_quantlib_continuous_and_discrete():
    rng = np.random.default_rng(9)
    for x, days, r, q, sigma, call in _cases(60, 9):
        process = _process(100.0, r, q, sigma)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, x)
        continuous = ql.ContinuousAveragingAsianOption(ql.Average.Geometric, payoff, ql.EuropeanExercise(TODAY + days))
        continuous.setPricingEngine(ql.AnalyticContinuousGeometricAveragePriceAsianEngine(process))
        assert cf.geometric_asian(100.0, x, days / 365, r, r - q, sigma, call) == pytest.approx(continuous.NPV(), abs=1e-11)
        n = int(rng.choice([2, 4, 5, 10]))
        days -= days % n  # QuantLib fixes on whole days: keep them equally spaced
        dates = [TODAY + days * (i + 1) // n for i in range(n)]
        discrete = ql.DiscreteAveragingAsianOption(ql.Average.Geometric, 1.0, 0, dates, payoff, ql.EuropeanExercise(TODAY + days))
        discrete.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))
        assert cf.geometric_asian(100.0, x, days / 365, r, r - q, sigma, call, n) == pytest.approx(discrete.NPV(), abs=1e-11)


def test_discrete_geometric_asian_tends_to_the_continuous_one():
    continuous = cf.geometric_asian(100.0, 100.0, 1.0, 0.05, 0.03, 0.25)
    gaps = [abs(cf.geometric_asian(100.0, 100.0, 1.0, 0.05, 0.03, 0.25, True, n) - continuous) for n in (10, 100, 1000, 10000)]
    assert all(b < a for a, b in zip(gaps, gaps[1:], strict=False)) and gaps[-1] < 1e-3


def test_lookbacks_match_quantlib():
    rng = np.random.default_rng(10)
    for x, days, r, q, sigma, call in _cases(60, 10):
        if abs(r - q) < 1e-3:
            continue
        process = _process(100.0, r, q, sigma)
        low_or_high = float(rng.uniform(80, 100)) if call else float(rng.uniform(100, 120))
        floating = ql.ContinuousFloatingLookbackOption(
            low_or_high, ql.FloatingTypePayoff(ql.Option.Call if call else ql.Option.Put), ql.EuropeanExercise(TODAY + days)
        )
        floating.setPricingEngine(ql.AnalyticContinuousFloatingLookbackEngine(process))
        assert cf.floating_lookback(100.0, low_or_high, days / 365, r, r - q, sigma, call) == pytest.approx(
            floating.NPV(), abs=1e-10
        )
        high_or_low = float(rng.uniform(100, 120)) if call else float(rng.uniform(80, 100))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, x)
        fixed = ql.ContinuousFixedLookbackOption(high_or_low, payoff, ql.EuropeanExercise(TODAY + days))
        fixed.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(process))
        assert cf.fixed_lookback(100.0, x, high_or_low, days / 365, r, r - q, sigma, call) == pytest.approx(
            fixed.NPV(), abs=1e-10
        )


def test_digitals_match_quantlib_and_add_up_to_the_vanilla():
    for x, days, r, q, sigma, call in _cases(40, 11):
        t, b = days / 365, r - q
        kind = ql.Option.Call if call else ql.Option.Put
        cash = ql.VanillaOption(ql.CashOrNothingPayoff(kind, x, 1.0), ql.EuropeanExercise(TODAY + days))
        cash.setPricingEngine(ql.AnalyticEuropeanEngine(_process(100.0, r, q, sigma)))
        asset = ql.VanillaOption(ql.AssetOrNothingPayoff(kind, x), ql.EuropeanExercise(TODAY + days))
        asset.setPricingEngine(ql.AnalyticEuropeanEngine(_process(100.0, r, q, sigma)))
        c, a = cf.cash_or_nothing(100.0, x, t, r, b, sigma, 1.0, call), cf.asset_or_nothing(100.0, x, t, r, b, sigma, call)
        assert c == pytest.approx(cash.NPV(), abs=1e-12)
        assert a == pytest.approx(asset.NPV(), abs=1e-11)
        sign = 1.0 if call else -1.0
        assert sign * (a - x * c) == pytest.approx(cf.black_scholes(100.0, x, t, r, b, sigma, call), abs=1e-11)
