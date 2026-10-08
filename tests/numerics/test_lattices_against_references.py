"""Lattices: the same trees as QuantLib, their convergence measured, and American prices held to high precision."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volsurf.analytic import greeks
from volsurf.numerics.lattice import LATTICES, Contract, bbsr, price
from volsurf.numerics.references import LONGSTAFF_SCHWARTZ_TABLE

CALL = Contract(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)


def test_every_lattice_converges_to_black_scholes():
    reference = CALL.european()
    for lattice in LATTICES:
        assert price(CALL, 2001, lattice).price == pytest.approx(reference, abs=2e-3), lattice


def test_leisen_reimer_converges_at_second_order_and_crr_at_first():
    reference = CALL.european()
    lr = [abs(price(CALL, n, "leisen-reimer").price - reference) for n in (101, 201, 401)]
    assert lr[0] / lr[1] > 3.5 and lr[1] / lr[2] > 3.5  # error falls fourfold when n doubles
    crr = np.array([abs(price(CALL, n, "crr").price - reference) for n in range(400, 440)])
    rises = int(np.sum(np.diff(crr) > 0))
    assert 10 <= rises <= 30  # adding a step makes the error worse about as often as better: it oscillates


def test_bbsr_cancels_the_leading_error():
    reference = CALL.european()
    assert abs(bbsr(CALL, 200, "crr", "european") - reference) < abs(price(CALL, 400, "crr").price - reference) / 10


def test_tree_greeks_match_the_analytic_ones():
    tree = price(CALL, 2001, "leisen-reimer")
    exact = greeks(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)
    assert tree.delta == pytest.approx(float(exact.delta), abs=1e-4)
    assert tree.gamma == pytest.approx(float(exact.gamma), abs=1e-4)


@pytest.mark.parametrize("row", LONGSTAFF_SCHWARTZ_TABLE[::3], ids=lambda r: f"S{r.spot:.0f}s{r.vol}T{r.maturity:.0f}")
def test_american_puts_reach_the_high_precision_reference(row):
    contract = Contract(row.spot, row.strike, row.maturity, row.rate, 0.0, row.vol, False)
    assert bbsr(contract, 1000, "crr") == pytest.approx(row.american, abs=2e-4)


@pytest.mark.parametrize("row", LONGSTAFF_SCHWARTZ_TABLE[:4], ids=lambda r: f"S{r.spot:.0f}s{r.vol}T{r.maturity:.0f}")
def test_a_bermudan_tree_reproduces_the_bermudan_reference(row):
    contract = Contract(row.spot, row.strike, row.maturity, row.rate, 0.0, row.vol, False)
    dates = [k / 50 for k in range(1, int(50 * row.maturity) + 1)]
    # CRR is first order: at 2,500 steps a year it is good to about 1e-3, the reference (Crank-Nicolson) to 1e-5
    assert price(contract, 2500 * int(row.maturity), "crr", dates).price == pytest.approx(row.bermudan_50, abs=1e-3)


def test_the_longstaff_schwartz_column_is_the_bermudan_price_not_the_american():
    one_year = [r for r in LONGSTAFF_SCHWARTZ_TABLE if r.maturity == 1.0]
    assert all(abs(r.longstaff_schwartz - r.bermudan_50) < 5e-4 for r in one_year)
    assert all(r.american - r.longstaff_schwartz > 2.5e-3 for r in one_year)


ql = pytest.importorskip("QuantLib")


@pytest.mark.parametrize(("ours", "theirs"), [("tian", "tian"), ("leisen-reimer", "lr")])
@pytest.mark.parametrize("exercise", ["european", "american"])
def test_tian_and_leisen_reimer_are_the_same_trees_as_quantlibs(ours, theirs, exercise):
    today = ql.Date(15, 3, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.25, day_count)),
    )
    is_call = exercise == "european"
    exercise_ql = ql.EuropeanExercise(today + 365) if is_call else ql.AmericanExercise(today, today + 365)
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, 105.0), exercise_ql)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, theirs, 301))
    contract = Contract(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, is_call)
    assert price(contract, 301, ours, exercise).price == pytest.approx(option.NPV(), abs=1e-11)


contracts = st.builds(
    Contract,
    st.floats(20.0, 200.0),
    st.floats(20.0, 200.0),
    st.floats(0.05, 3.0),
    st.floats(0.0, 0.1),
    st.floats(0.0, 0.08),
    st.floats(0.1, 0.8),
    st.booleans(),
)


@settings(max_examples=60, deadline=None)
@given(contracts)
def test_early_exercise_is_worth_something_and_never_less_than_intrinsic(contract):
    european = price(contract, 200, "crr").price
    american = price(contract, 200, "crr", "american").price
    bermudan = price(contract, 200, "crr", list(np.linspace(0, contract.maturity, 6)[1:-1])).price
    intrinsic = float(contract.payoff(np.array([contract.spot]))[0])
    assert european - 1e-12 <= bermudan <= american + 1e-12
    assert american >= intrinsic - 1e-12


@settings(max_examples=40, deadline=None)
@given(st.floats(20.0, 200.0), st.floats(20.0, 200.0), st.floats(0.05, 3.0), st.floats(0.0, 0.1), st.floats(0.1, 0.8))
def test_an_american_call_without_dividends_is_never_exercised_early(spot, strike, maturity, rate, vol):
    """Merton (1973): with no dividend, a call is worth more alive than exercised."""
    contract = Contract(spot, strike, maturity, rate, 0.0, vol, True)
    assert price(contract, 300, "crr", "american").price == pytest.approx(price(contract, 300, "crr").price, abs=1e-10)
