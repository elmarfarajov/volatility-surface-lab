"""An S&P 500 options book: its Greeks, its stress ladder under each smile dynamic, and its historical value at risk.

**The book** is a dealer's typical position: short downside puts sold to clients for
protection, long upside calls bought from clients overwriting their holdings, and the
delta hedged with the index. It is short gamma, short vega and short skew. Positions are
priced off the Day 4 S&P 500 surface, each at its own expiry's rate and carry.

**Revaluation.** After the index moves by ``s = ln(S'/S)``, an option of strike ``K`` and
expiry ``T`` is repriced at

    sigma'(K, T) = sigma(k' + c s, T) + d sigma(T),   k' = ln(K / F'),

where ``c`` is the smile dynamic: ``c = 0`` sticky moneyness (the smile moves with the
index), ``c = 1`` sticky strike (it stays put), ``c = 2`` local volatility's (Day 6), and
the market's own from thirty-six years of the VIX. ``d sigma(T)`` is a volatility shock by
expiry.

**Historical scenarios** apply every daily (or ten-day) move of the index and the VIX since
1990 to today's book. The observed VIX move is the at-the-money move, so the smile is held
in moneyness (``c = 0``) and shifted by it: 30-day volatility moves with the VIX, 91-day with
VIX3M, and longer expiries by VIX3M scaled down as ``(91 days / T)^alpha``, ``alpha`` from the
regression of VIX3M moves on VIX moves. Before VIX3M exists (2006) it is that regression's
prediction. Full revaluation is compared with the Greeks' second-order explain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..analytic.generalized import Greeks, greeks
from ..smile.build import SurfaceFit
from .explain import Explain, explain

Array = NDArray[np.float64]
MULTIPLIER = 100.0  # S&P 500 index options: dollars per index point


@dataclass(frozen=True)
class Position:
    name: str
    expiry_days: int  # target; the nearest listed expiry is used
    moneyness: float  # strike over today's spot
    call: bool
    contracts: float  # positive long, negative short


DEALER_BOOK = (
    Position("1m 95% put", 30, 0.95, False, -200),
    Position("3m 90% put", 91, 0.90, False, -150),
    Position("1y 80% put", 365, 0.80, False, -100),
    Position("3m 105% call", 91, 1.05, True, 150),
    Position("6m 110% call", 182, 1.10, True, 100),
)


@dataclass(frozen=True)
class Book:
    names: tuple[str, ...]
    spot: float
    strike: Array  # in units of today's spot
    expiry: Array  # years
    rate: Array
    yield_: Array
    vol: Array
    calls: NDArray[np.bool_]
    contracts: Array
    hedge: float  # index units (per unit of spot) held to make the book delta-neutral
    surface_vol: Callable[[Array, float], Array]  # (log-moneyness, expiry) -> implied vol

    def greeks(self) -> Greeks:
        return greeks(1.0, self.strike, self.expiry, self.rate, self.yield_, self.vol, self.calls)

    def dollars(self, per_unit: Array) -> Array:
        """Per-unit-of-spot values, times contracts and the multiplier and the index level, summed into dollars."""
        return np.asarray(per_unit * self.spot * MULTIPLIER, dtype=np.float64)


def build(fit: SurfaceFit, positions: Sequence[Position] = DEALER_BOOK) -> Book:
    spot = float(fit.surface.spot or 1.0)
    strikes, expiries, rates, yields, vols, calls = [], [], [], [], [], []
    for p in positions:
        e = min(fit.expiries, key=lambda e: abs(e.maturity * 365 - p.expiry_days))
        t = e.maturity
        r = float(-np.log(e.discount) / t)
        b = float(np.log(e.forward / spot) / t)
        k = float(np.log(p.moneyness) - b * t)
        strikes.append(p.moneyness)
        expiries.append(t)
        rates.append(r)
        yields.append(r - b)
        vols.append(float(fit.surface.implied_vol(np.array([k]), t)[0]))
        calls.append(p.call)
    contracts = np.array([p.contracts for p in positions], dtype=np.float64)
    g = greeks(1.0, np.array(strikes), np.array(expiries), np.array(rates), np.array(yields), np.array(vols), np.array(calls))
    hedge = -float(np.sum(contracts * g.delta))
    return Book(
        tuple(p.name for p in positions),
        spot,
        np.array(strikes),
        np.array(expiries),
        np.array(rates),
        np.array(yields),
        np.array(vols),
        np.array(calls),
        contracts,
        hedge,
        fit.surface.implied_vol,
    )


@dataclass(frozen=True)
class Revaluation:
    pnl: Array  # dollars, per scenario: full revaluation of options and hedge
    explain: Explain  # per scenario and position, per unit of spot and contract
    options: Array  # dollars per scenario and position


def revalue(book: Book, s: Array, d_vol: Array, dt: float = 1.0 / 365.0, c: float = 0.0) -> Revaluation:
    """Reprice the book after index log-moves ``s`` (scenarios) and volatility shocks ``d_vol`` ([scenarios, positions])."""
    s = np.asarray(s, dtype=np.float64)[:, None]
    tau = np.maximum(book.expiry - dt, 1e-6)[None, :]
    b = book.rate - book.yield_
    k_new = np.log(book.strike)[None, :] - s - (b * tau)
    vol_new = np.empty_like(k_new)
    for j, t in enumerate(book.expiry):
        vol_new[:, j] = book.surface_vol(k_new[:, j] + c * s[:, 0], float(t))
    vol_new = np.maximum(vol_new + d_vol, 0.01)
    spot_new = np.exp(s)
    after = greeks(spot_new, book.strike, tau, book.rate, book.yield_, vol_new, book.calls).price
    before = book.greeks()
    change = after - before.price[None, :]
    options = book.dollars(change * book.contracts[None, :])
    hedge = book.dollars(book.hedge * (spot_new[:, 0] - 1.0))
    full = options.sum(axis=1) + hedge
    terms = explain(before, spot_new - 1.0, vol_new - book.vol[None, :], dt, change)
    return Revaluation(np.asarray(full, dtype=np.float64), terms, options)


def explained_dollars(book: Book, rev: Revaluation) -> dict[str, Array]:
    """Each Greek term summed over the book, in dollars per scenario (delta includes the hedge)."""
    out = {name: book.dollars(np.sum(term * book.contracts[None, :], axis=1)) for name, term in rev.explain.terms.items()}
    return out


def stress_ladder(book: Book, moves: Array, dynamics: dict[str, float]) -> pd.DataFrame:
    """Book P&L in dollars for each index move, under each smile dynamic, with no other volatility shock."""
    s = np.log1p(np.asarray(moves, dtype=np.float64))
    zero = np.zeros((s.size, book.expiry.size))
    return pd.DataFrame({name: revalue(book, s, zero, c=c).pnl for name, c in dynamics.items()}, index=np.asarray(moves))


def vol_shocks(d_vix: Array, d_vix3m: Array, expiries: Array, alpha: float) -> Array:
    """Volatility shock by expiry from the VIX (30 days) and VIX3M (91 days): flat before, log-linear between, decaying after."""
    days = np.asarray(expiries) * 365.0
    w = np.clip(np.log(days / 30.0) / np.log(91.0 / 30.0), 0.0, 1.0)
    between = (1.0 - w)[None, :] * d_vix[:, None] + w[None, :] * d_vix3m[:, None]
    beyond = d_vix3m[:, None] * (91.0 / np.maximum(days, 91.0))[None, :] ** alpha
    return np.asarray(np.where(days[None, :] <= 91.0, between, beyond), dtype=np.float64)


@dataclass(frozen=True)
class HistoricalRisk:
    dates: pd.DatetimeIndex
    full: Array
    greeks: Array
    terms: dict[str, Array]
    horizon: int

    def var(self, level: float = 0.99, which: str = "full") -> float:
        pnl = self.full if which == "full" else self.greeks
        return float(-np.quantile(pnl, 1.0 - level))

    def es(self, level: float = 0.975, which: str = "full") -> float:
        pnl = self.full if which == "full" else self.greeks
        cut = np.quantile(pnl, 1.0 - level)
        return float(-pnl[pnl <= cut].mean())


def historical(book: Book, history: pd.DataFrame, horizon: int = 1, term_ratio: float | None = None) -> HistoricalRisk:
    """Every ``horizon``-day move since 1990 applied to today's book (overlapping windows for horizons over a day)."""
    log_spot = np.log(history["spot"].to_numpy(float))
    vix = history["vix"].to_numpy(float)
    vix3m = history["vix3m"].to_numpy(float)
    s = log_spot[horizon:] - log_spot[:-horizon]
    d_vix = vix[horizon:] - vix[:-horizon]
    d3 = vix3m[horizon:] - vix3m[:-horizon]
    if term_ratio is None:
        both = np.isfinite(d3)
        term_ratio = float(np.polyfit(d_vix[both], d3[both], 1)[0])
    d3 = np.where(np.isfinite(d3), d3, term_ratio * d_vix)
    alpha = float(np.log(term_ratio) / np.log(30.0 / 91.0))
    shocks = vol_shocks(d_vix, d3, book.expiry, alpha)
    rev = revalue(book, s, shocks, dt=horizon / 252.0)
    terms = explained_dollars(book, rev)
    terms["delta"] = terms["delta"] + book.dollars(book.hedge * (np.exp(s) - 1.0))
    approx = np.asarray(sum(terms.values()), dtype=np.float64)
    return HistoricalRisk(history.index[horizon:], rev.pnl, approx, terms, horizon)
