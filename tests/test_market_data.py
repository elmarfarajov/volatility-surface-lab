from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volsurf.heston import heston_price_cos
from volsurf.implied_vol import implied_vol_black76
from volsurf.market_data import (
    build_snapshot,
    estimate_forward_discount,
    fetch_yfinance_chain,
    normalise_yfinance_chain,
    select_expiries,
    synthetic_chain,
)

AS_OF = datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)


def test_parity_regression_recovers_carry(clean_snapshot):
    for e in clean_snapshot.expiries:
        assert e.forward_method == "put-call parity"
        assert e.forward == pytest.approx(5000.0 * np.exp((0.04 - 0.013) * e.maturity), rel=2e-5)
        assert e.discount == pytest.approx(np.exp(-0.04 * e.maturity), abs=5e-5)


def test_implied_vols_match_model(clean_snapshot, true_params):
    e = clean_snapshot.expiries[2]
    strikes = e.quotes["strike"].to_numpy()
    is_call = (e.quotes["option_type"] == "C").to_numpy()
    model = heston_price_cos(e.forward, strikes, e.maturity, e.discount, true_params, is_call)
    model_iv = implied_vol_black76(model, e.forward, strikes, e.maturity, e.discount, is_call)
    np.testing.assert_allclose(e.quotes["iv_mid"], model_iv, atol=2e-3)


def test_only_out_of_the_money_quotes_are_kept(clean_snapshot):
    for e in clean_snapshot.expiries:
        q = e.quotes
        assert (q.loc[q["option_type"] == "P", "strike"] < e.forward).all()
        assert (q.loc[q["option_type"] == "C", "strike"] >= e.forward).all()
        assert (q["iv_bid"] <= q["iv_mid"] + 1e-12).all() and (q["iv_mid"] <= q["iv_ask"] + 1e-12).all()


def test_parity_regression_rejects_outliers():
    strikes = np.arange(91.0, 110.0, 1.0)
    forward, discount = 101.3, 0.985
    diff = discount * (forward - strikes)
    diff[[3, 11]] += np.array([2.5, -3.0])
    calls = np.maximum(diff, 0) + 2.0
    puts = calls - diff
    f, d, n, method = estimate_forward_discount(strikes, calls, puts, np.full(strikes.size, 0.1), 100.0, 0.5)
    assert method == "put-call parity" and n == strikes.size - 2
    assert f == pytest.approx(forward, rel=1e-9) and d == pytest.approx(discount, rel=1e-9)


def test_parity_falls_back_when_data_is_insufficient():
    f, d, n, method = estimate_forward_discount(
        np.array([100.0, 101.0]), np.array([2.0, 1.5]), np.array([1.0, 1.4]), np.ones(2), 100.0, 0.5, 0.05
    )
    assert method == "fallback" and f == pytest.approx(100 * np.exp(0.025))


def test_crossed_and_empty_quotes_are_removed(true_params):
    raw = synthetic_chain(true_params, maturities_days=(60,), strikes_per_expiry=21)
    raw.loc[0, ["bid", "ask"]] = [5.0, 4.0]
    raw.loc[1, ["bid", "ask"]] = [0.0, 1.0]
    snapshot = build_snapshot(raw, 5000.0, AS_OF, "SYNTH", "synthetic")
    kept = snapshot.expiries[0].quotes
    assert len(kept) > 5
    assert not ((kept["bid"] <= 0) | (kept["ask"] <= kept["bid"])).any()


def test_expiry_selection_spans_the_term_structure():
    expiries = [
        (AS_OF + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for d in (1, 3, 8, 15, 22, 31, 45, 62, 90, 120, 185, 270, 365, 550, 730, 900)
    ]
    chosen = select_expiries(expiries, AS_OF, max_expiries=6)
    assert len(chosen) == 6
    assert chosen == sorted(chosen)
    assert (AS_OF + pd.Timedelta(days=3)).strftime("%Y-%m-%d") not in chosen


def _yahoo_frame(root: str, flag: str, strikes, bid, oi):
    return pd.DataFrame(
        {
            "contractSymbol": [f"{root}261218{flag}{int(k * 1000):08d}" for k in strikes],
            "strike": strikes,
            "lastPrice": np.asarray(bid) + 0.1,
            "bid": bid,
            "ask": np.asarray(bid) + 0.2,
            "volume": [10.0] * len(strikes),
            "openInterest": oi,
        }
    )


def test_normalise_prefers_dominant_root_and_dedupes():
    calls = pd.concat(
        [
            _yahoo_frame("SPXW", "C", [5000.0, 5050.0, 5100.0], [50.0, 30.0, 15.0], [10, 10, 10]),
            _yahoo_frame("SPX", "C", [5000.0], [51.0], [999]),
        ]
    )
    puts = _yahoo_frame("SPXW", "P", [5000.0, 5050.0], [40.0, 60.0], [5, 5])
    out = normalise_yfinance_chain(calls, puts, "2026-12-18", AS_OF)
    assert len(out) == 5
    assert set(out["option_type"]) == {"C", "P"}
    assert out.loc[(out["strike"] == 5000.0) & (out["option_type"] == "C"), "bid"].item() == 50.0
    assert out["maturity"].iloc[0] == pytest.approx(94 / 365, abs=2 / 365)


def test_fetch_uses_injected_ticker_without_network(true_params):
    expiry = "2026-12-18"
    strikes = np.arange(4600.0, 5400.0, 25.0)
    forward, discount, maturity = 5020.0, 0.99, 94 / 365
    call = heston_price_cos(forward, strikes, maturity, discount, true_params, True)
    put = heston_price_cos(forward, strikes, maturity, discount, true_params, False)

    class FakeTicker:
        options = (expiry,)

        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period):
            return pd.DataFrame({"Close": [4990.0, 5000.0]})

        def option_chain(self, date):
            return SimpleNamespace(
                calls=_yahoo_frame("SPXW", "C", strikes, call - 0.05, [100] * strikes.size),
                puts=_yahoo_frame("SPXW", "P", strikes, put - 0.05, [100] * strikes.size),
            )

    raw, spot, as_of = fetch_yfinance_chain("^SPX", max_expiries=3, ticker_factory=FakeTicker)
    assert spot == 5000.0
    snapshot = build_snapshot(raw, spot, as_of, "^SPX", "fake")
    assert snapshot.expiries[0].forward == pytest.approx(forward, rel=1e-3)
