"""Option-chain snapshots: fetched once, stored locally, loaded the same way every time.

A market chart is only reproducible if the quotes behind it are fixed, so the live
feed is never read directly by the analysis. ``volsurf chains fetch SPX`` stores
every expiry Yahoo Finance lists as one compressed file per ticker and trading day,

    data/chains/<YYYY-MM-DD>/<TICKER>.csv.gz   with   <TICKER>.json   (spot, fetch time, source)

and :func:`load` turns it into a :class:`Chain` with one schema: maturity in years
to the contract's settlement, strike, side, bid, ask, last trade, volume, open
interest, and the vendor's own implied volatility for comparison.

The quotes are third-party data and are not redistributed with the repository
(``data/`` is ignored by git); the gallery's market charts are drawn from a local
snapshot and skipped where there is none. ``VOLSURF_DATA`` points elsewhere.

Settlement follows the contract: standard monthly SPX options are AM-settled at
Friday's opening print (09:30 New York), SPXW weeklies and equity options settle
at 16:00. Maturity is ACT/365 from the snapshot's close.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_ROOT = re.compile(r"^([A-Z]+)\d")
_KEEP = ["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]


def data_root() -> Path:
    """Where snapshots live: ``$VOLSURF_DATA`` or ``data/chains`` in the repository."""
    configured = os.environ.get("VOLSURF_DATA")
    return Path(configured) if configured else Path(__file__).resolve().parents[3] / "data" / "chains"


@dataclass(frozen=True)
class Chain:
    """Every listed option on one underlying at one close."""

    ticker: str
    spot: float
    as_of: datetime  # the close the quotes belong to, UTC
    quotes: pd.DataFrame  # expiry, root, maturity, strike, is_call, bid, ask, last, last_trade, volume, open_interest, vendor_iv

    def expiries(self) -> list[str]:
        return sorted(self.quotes["expiry"].unique())

    def expiry(self, expiry: str) -> pd.DataFrame:
        return self.quotes[self.quotes["expiry"] == expiry].reset_index(drop=True)


def settlement(expiry: str, root: str) -> datetime:
    """The settlement instant of a contract expiring on ``expiry`` (UTC)."""
    clock = time(9, 30) if root == "SPX" else time(16, 0)
    return datetime.combine(date.fromisoformat(expiry), clock, tzinfo=NEW_YORK).astimezone(timezone.utc)


def snapshots(root: Path | None = None) -> list[tuple[str, str]]:
    """``(day, ticker)`` for every stored snapshot, oldest first."""
    base = root or data_root()
    if not base.exists():
        return []
    return sorted((p.parent.name, p.name.removesuffix(".csv.gz")) for p in base.glob("*/*.csv.gz"))


def latest(ticker: str, root: Path | None = None) -> str | None:
    """The most recent day with a snapshot of ``ticker``, if any."""
    days = [day for day, name in snapshots(root) if name == ticker.replace("^", "")]
    return days[-1] if days else None


def load(ticker: str, day: str | None = None, root: Path | None = None) -> Chain:
    """Read a stored snapshot (the latest when ``day`` is omitted) into the common schema."""
    name = ticker.replace("^", "")
    base = root or data_root()
    day = day or latest(name, base)
    if day is None:
        raise FileNotFoundError(f"no snapshot of {ticker} under {base}; run `volsurf chains fetch {ticker}`")
    meta = json.loads((base / day / f"{name}.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(base / day / f"{name}.csv.gz")
    as_of = datetime.combine(date.fromisoformat(meta["spot_date"]), time(16, 0), tzinfo=NEW_YORK).astimezone(timezone.utc)
    roots = raw["contractSymbol"].str.extract(_ROOT, expand=False)
    pairs = list(zip(raw["expiry"].astype(str), roots, strict=True))
    years = {pair: (settlement(*pair) - as_of).total_seconds() / _SECONDS_PER_YEAR for pair in set(pairs)}
    maturity = np.array([years[pair] for pair in pairs])
    quotes = pd.DataFrame(
        {
            "expiry": raw["expiry"].astype(str),
            "root": roots,
            "maturity": maturity,
            "strike": raw["strike"].astype(float),
            "is_call": raw["option_type"].eq("C").to_numpy(),
            "bid": raw["bid"].astype(float).fillna(0.0),
            "ask": raw["ask"].astype(float).fillna(0.0),
            "last": raw["lastPrice"].astype(float),
            "last_trade": pd.to_datetime(raw["lastTradeDate"], utc=True),
            "volume": raw["volume"].astype(float).fillna(0.0),
            "open_interest": raw["openInterest"].astype(float).fillna(0.0),
            "vendor_iv": raw["impliedVolatility"].astype(float),
        }
    )
    quotes = quotes[quotes["maturity"] > 0].sort_values(["maturity", "root", "strike", "is_call"]).reset_index(drop=True)
    return Chain(meta["ticker"], float(meta["spot"]), as_of, quotes)


def fetch(ticker: str, root: Path | None = None, ticker_factory: object = None) -> Path:  # pragma: no cover - network
    """Download every listed expiry of ``ticker`` from Yahoo Finance and store it as a snapshot."""
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker
    assert callable(ticker_factory)
    handle = ticker_factory(ticker)
    history = handle.history(period="5d")
    if history is None or history.empty:
        raise RuntimeError(f"no price history for {ticker}")
    spot, spot_date = float(history["Close"].iloc[-1]), str(history.index[-1].date())
    frames = []
    for expiry in handle.options:
        sides = handle.option_chain(expiry)
        for frame, flag in ((sides.calls, "C"), (sides.puts, "P")):
            if frame is not None and not frame.empty:
                frames.append(frame[_KEEP].assign(expiry=expiry, option_type=flag))
    if not frames:
        raise RuntimeError(f"no listed options for {ticker}")
    name = ticker.replace("^", "")
    folder = (root or data_root()) / spot_date
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(folder / f"{name}.csv.gz", index=False)
    meta = {"ticker": ticker, "spot": spot, "spot_date": spot_date, "source": "Yahoo Finance"}
    meta["fetched"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder / f"{name}.csv.gz"
