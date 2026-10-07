"""The analytic engine against three references that share nothing with it but the model's definition."""

from __future__ import annotations

import numpy as np
import pytest

from volsurf.analytic import greeks
from volsurf.analytic.published import HAUG_2007, Example
from volsurf.analytic.reference import DEFINITIONS, reference_greeks


@pytest.mark.parametrize("example", HAUG_2007, ids=lambda e: e.title)
def test_haugs_worked_examples_reproduce_to_the_printed_digit(example: Example):
    g = greeks(
        example.spot,
        example.strike,
        example.maturity,
        example.rate,
        example.yield_,
        example.vol,
        example.is_call,
    )
    assert round(float(getattr(g, example.quantity)), 4) == pytest.approx(example.published, abs=1e-12)


def random_contracts(count: int, seed: int) -> list[tuple[float, float, float, float, float, float, bool]]:
    rng = np.random.default_rng(seed)
    return [
        (
            float(rng.uniform(50, 150)),
            float(rng.uniform(50, 150)),
            float(rng.uniform(0.05, 3)),
            float(rng.uniform(-0.02, 0.1)),
            float(rng.uniform(0, 0.06)),
            float(rng.uniform(0.08, 0.8)),
            bool(rng.random() < 0.5),
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize("contract", random_contracts(12, seed=2026), ids=lambda c: f"S{c[0]:.0f}K{c[1]:.0f}")
def test_every_greek_matches_arbitrary_precision_differentiation(contract):
    """Seventeen Greeks, to third order, against mpmath at 50 digits: a sign or a term wrong shows at once."""
    analytic = greeks(*contract).as_dict()
    reference = reference_greeks(*contract)
    for name in DEFINITIONS:
        assert float(analytic[name]) == pytest.approx(reference[name], rel=1e-11, abs=1e-12), name


ql = pytest.importorskip("QuantLib")


def quantlib_greeks(spot, strike, days, rate, yield_, vol, is_call):
    today = ql.Date(15, 3, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, yield_, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, day_count)),
    )
    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        ql.EuropeanExercise(today + days),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "theta": option.theta(),
        "rho": option.rho(),
        "psi": option.dividendRho(),
    }


def test_the_engine_agrees_with_quantlib_on_five_hundred_random_contracts():
    rng = np.random.default_rng(11)
    worst: dict[str, float] = {}
    for _ in range(500):
        spot, strike = rng.uniform(20, 200), rng.uniform(20, 200)
        days = int(rng.integers(2, 5 * 365))
        rate, yield_, vol = rng.uniform(-0.02, 0.12), rng.uniform(0, 0.08), rng.uniform(0.03, 1.2)
        is_call = bool(rng.random() < 0.5)
        theirs = quantlib_greeks(spot, strike, days, rate, yield_, vol, is_call)
        ours = greeks(spot, strike, days / 365.0, rate, yield_, vol, is_call).as_dict()
        for name, value in theirs.items():
            error = abs(float(ours[name]) - value) / max(abs(value), 1.0)
            worst[name] = max(worst.get(name, 0.0), error)
    assert max(worst.values()) < 1e-10, worst
