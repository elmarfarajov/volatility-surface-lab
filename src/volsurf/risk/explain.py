"""P&L explained by Greeks: what a risk report attributes, and what it leaves unexplained.

A desk explains each day's P&L as a sum of Greek terms over the day's moves in spot ``dS``,
the option's own implied volatility ``d sigma`` and time ``dt``:

    dV ~ delta dS + 1/2 gamma dS^2 + vega d sigma + vanna dS d sigma + 1/2 volga d sigma^2
         + theta dt + charm dS dt + veta d sigma dt.

The last two, the time decay of delta and of vega, are often left out; for short-dated
options away from the money they are as large as gamma over a day. What is left, the unexplained P&L, is the
third-order terms the expansion drops. It is small on quiet days and grows with the size of
the move: the explain is a local statement, and a risk measure built on it misstates crashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..analytic.generalized import Greeks

Array = NDArray[np.float64]
TERMS = ("delta", "gamma", "vega", "vanna", "volga", "theta", "charm", "veta")


@dataclass(frozen=True)
class Explain:
    terms: dict[str, Array]  # each Greek's contribution, in the order of TERMS
    full: Array  # full revaluation

    @property
    def explained(self) -> Array:
        return np.asarray(sum(self.terms.values()), dtype=np.float64)

    @property
    def unexplained(self) -> Array:
        return np.asarray(self.full - self.explained, dtype=np.float64)


def explain(g: Greeks, d_spot: ArrayLike, d_vol: ArrayLike, dt: float, full: ArrayLike) -> Explain:
    """Attribute ``full`` (the revalued change in price) to Greek terms, for any broadcastable moves."""
    ds = np.asarray(d_spot, dtype=np.float64)
    dv = np.asarray(d_vol, dtype=np.float64)
    terms = {
        "delta": g.delta * ds,
        "gamma": 0.5 * g.gamma * ds**2,
        "vega": g.vega * dv,
        "vanna": g.vanna * ds * dv,
        "volga": 0.5 * g.volga * dv**2,
        "theta": g.theta * dt * np.ones_like(ds * dv),
        "charm": g.charm * ds * dt,
        "veta": g.veta * dv * dt,
    }
    return Explain({k: np.asarray(v, dtype=np.float64) for k, v in terms.items()}, np.asarray(full, dtype=np.float64))
