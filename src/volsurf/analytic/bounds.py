"""Model-free no-arbitrage bounds on European option prices.

Whatever the model, a European option's price must respect what a static
portfolio of the underlying and a bond can enforce (Merton, 1973):

* a call is worth at least ``max(S e^{-qT} - K e^{-rT}, 0)`` and at most ``S e^{-qT}``;
* a put is worth at least ``max(K e^{-rT} - S e^{-qT}, 0)`` and at most ``K e^{-rT}``;

and, across strikes at one maturity,

* call prices fall as the strike rises, by no more than the discounted strike step:
  ``-e^{-rT} <= dC/dK <= 0``;
* call prices are convex in the strike (a butterfly costs nothing negative).

The checks take quotes from any source - a model, or the market - and report
each violation with its size, so the surface built on Day 4 can refuse an
arbitrage before it is fitted, and this day's model can be shown to respect
them everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def merton_bounds(
    spot: ArrayLike, strike: ArrayLike, maturity: ArrayLike, rate: ArrayLike, yield_: ArrayLike, is_call: ArrayLike = True
) -> tuple[Array, Array]:
    """(lower, upper) bounds on a European price, model-free."""
    spot_, strike_, t, r, q = (np.asarray(v, dtype=np.float64) for v in (spot, strike, maturity, rate, yield_))
    call = np.asarray(is_call, dtype=bool)
    forward_leg, strike_leg = spot_ * np.exp(-q * t), strike_ * np.exp(-r * t)
    lower = np.where(call, np.maximum(forward_leg - strike_leg, 0.0), np.maximum(strike_leg - forward_leg, 0.0))
    upper = np.where(call, forward_leg, strike_leg)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


@dataclass(frozen=True)
class Violation:
    kind: str  # "below lower bound", "above upper bound", "increasing in strike", "too steep", "not convex"
    index: int  # the quote (or the middle quote of a butterfly) that breaks it
    size: float  # by how much, in price


def strike_violations(strikes: ArrayLike, calls: ArrayLike, discount: float, tolerance: float = 1e-12) -> list[Violation]:
    """Monotonicity, slope and convexity of call prices across strikes at one maturity."""
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(calls, dtype=np.float64)
    order = np.argsort(k)
    k, c = k[order], c[order]
    found: list[Violation] = []
    slopes = np.diff(c) / np.diff(k)
    for i, slope in enumerate(slopes):
        if slope > tolerance:
            found.append(Violation("increasing in strike", int(order[i + 1]), float(slope * (k[i + 1] - k[i]))))
        if slope < -discount - tolerance:
            found.append(Violation("too steep", int(order[i + 1]), float((-discount - slope) * (k[i + 1] - k[i]))))
    for i in range(1, len(k) - 1):
        # the value of a (possibly uneven) butterfly centred on strike i, which must not be negative
        left, right = k[i] - k[i - 1], k[i + 1] - k[i]
        butterfly = c[i - 1] * right - c[i] * (left + right) + c[i + 1] * left
        if butterfly < -tolerance * (left + right):
            found.append(Violation("not convex", int(order[i]), float(-butterfly / (left + right))))
    return found


def bound_violations(
    prices: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    yield_: ArrayLike,
    is_call: ArrayLike = True,
    tolerance: float = 1e-12,
) -> list[Violation]:
    """Quotes outside Merton's bounds."""
    p = np.atleast_1d(np.asarray(prices, dtype=np.float64))
    lower, upper = (np.broadcast_to(b, p.shape) for b in merton_bounds(spot, strike, maturity, rate, yield_, is_call))
    below, above = np.flatnonzero(p < lower - tolerance), np.flatnonzero(p > upper + tolerance)
    found = [Violation("below lower bound", int(i), float(lower[i] - p[i])) for i in below]
    found += [Violation("above upper bound", int(i), float(p[i] - upper[i])) for i in above]
    return found
