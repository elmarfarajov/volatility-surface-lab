"""Cost of carry: one Black-Scholes-Merton model for stocks, indices, futures and currencies.

What distinguishes a stock option from a futures option or a currency option is
not the model but the cost of carrying the underlying: holding a stock earns its
dividends, holding a currency earns the foreign interest rate, and a future costs
nothing to hold. Generalised Black-Scholes-Merton (Haug, 2007) captures this with
one parameter, the cost of carry ``b``, so that the forward is ``S exp(bT)``:

==============================  =======================  ==========================
underlying                      cost of carry b          yield q = r - b
==============================  =======================  ==========================
non-dividend stock (Black-Scholes)  r                    0
stock or index (Merton)         r - q                    continuous dividend yield
future (Black-76)               0                        r
currency (Garman-Kohlhagen)     r_d - r_f                foreign interest rate
margined future (Asay)          0, and r = 0             0
==============================  =======================  ==========================

The library works with the yield ``q`` rather than ``b``: every formula reads
``exp(-qT)`` where Haug writes ``exp((b - r)T)``. The two are the same model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Carry:
    """The rates an option is priced with: the discount rate and the yield of the underlying."""

    rate: float  # continuously compounded rate the payoff is discounted at
    yield_: float  # continuously compounded yield of the underlying: dividends, foreign rate, or the rate itself
    convention: str

    @property
    def cost_of_carry(self) -> float:
        """Haug's b: the forward is spot times exp(b T)."""
        return self.rate - self.yield_

    def forward(self, spot: float, maturity: float) -> float:
        return float(spot * np.exp(self.cost_of_carry * maturity))

    def discount(self, maturity: float) -> float:
        return float(np.exp(-self.rate * maturity))

    @classmethod
    def stock(cls, rate: float, dividend_yield: float = 0.0) -> Carry:
        """Black-Scholes (no dividend) or Merton (a continuous dividend yield)."""
        return cls(rate, dividend_yield, "Black-Scholes" if dividend_yield == 0 else "Merton")

    @classmethod
    def future(cls, rate: float) -> Carry:
        """Black-76: the underlying is a futures price, which costs nothing to hold."""
        return cls(rate, rate, "Black-76")

    @classmethod
    def currency(cls, domestic_rate: float, foreign_rate: float) -> Carry:
        """Garman-Kohlhagen: holding the foreign currency earns the foreign rate."""
        return cls(domestic_rate, foreign_rate, "Garman-Kohlhagen")

    @classmethod
    def margined_future(cls) -> Carry:
        """Asay: a future-style option, margined like the future, so the premium is not paid up front."""
        return cls(0.0, 0.0, "Asay")
