"""Snapshots: fetched through a stand-in for Yahoo, stored, and loaded into one schema."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from volsurf.market import chains


class _FakeTicker:
    options = ("2026-10-16", "2026-11-20")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, period: str) -> pd.DataFrame:
        return pd.DataFrame({"Close": [7800.0, 7811.5]}, index=pd.to_datetime(["2026-10-08", "2026-10-09"]))

    def option_chain(self, expiry: str) -> SimpleNamespace:
        def side(flag: str) -> pd.DataFrame:
            stamp = expiry.replace("-", "")[2:]
            roots = ["SPX", "SPXW"]
            return pd.DataFrame(
                {
                    "contractSymbol": [f"{r}{stamp}{flag}07800000" for r in roots],
                    "lastTradeDate": ["2026-10-09 19:59:00+00:00"] * 2,
                    "strike": [7800.0, 7800.0],
                    "lastPrice": [100.0, 101.0],
                    "bid": [99.0, None],
                    "ask": [101.0, 103.0],
                    "volume": [5.0, None],
                    "openInterest": [10.0, 3.0],
                    "impliedVolatility": [0.15, 0.16],
                }
            )

        return SimpleNamespace(calls=side("C"), puts=side("P"))


def test_fetch_then_load_round_trip(tmp_path):
    path = chains.fetch("^SPX", root=tmp_path, ticker_factory=_FakeTicker)
    assert path.name == "SPX.csv.gz" and path.parent.name == "2026-10-09"
    assert chains.snapshots(tmp_path) == [("2026-10-09", "SPX")]
    assert chains.latest("^SPX", tmp_path) == "2026-10-09"
    chain = chains.load("^SPX", root=tmp_path)
    q = chain.quotes
    assert chain.spot == 7811.5 and len(q) == 8
    assert set(q["root"]) == {"SPX", "SPXW"}
    assert q["bid"].min() == 0.0 and q["volume"].min() == 0.0  # missing values are no bid, no volume
    # AM-settled SPX expires 6.5 hours before PM-settled SPXW on the same day
    one = q[q["expiry"] == "2026-10-16"].groupby("root")["maturity"].first()
    assert (one["SPXW"] - one["SPX"]) * 365 * 24 == pytest.approx(6.5)
    assert chain.expiries() == ["2026-10-16", "2026-11-20"] and len(chain.expiry("2026-11-20")) == 4


def test_a_missing_snapshot_says_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="volsurf chains fetch"):
        chains.load("SPY", root=tmp_path)
    assert chains.snapshots(tmp_path / "nowhere") == []


def test_data_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLSURF_DATA", str(tmp_path))
    assert chains.data_root() == tmp_path
