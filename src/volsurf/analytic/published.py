"""Published worked examples the analytic engine must reproduce, as printed.

The examples are from E. G. Haug, *The Complete Guide to Option Pricing
Formulas*, 2nd edition (McGraw-Hill, 2007), chapters 1 and 2, which work the
generalised Black-Scholes-Merton model through each carry convention and each
first-order Greek. The book prints four decimals; the engine must agree to the
last of them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    title: str
    quantity: str  # "price" or a Greek's name in volsurf.analytic.generalized.Greeks
    spot: float
    strike: float
    maturity: float
    rate: float
    cost_of_carry: float  # Haug's b; the engine's yield is r - b
    vol: float
    is_call: bool
    published: float

    @property
    def yield_(self) -> float:
        return self.rate - self.cost_of_carry


HAUG_2007: tuple[Example, ...] = (
    Example("Black-Scholes call", "price", 60.0, 65.0, 0.25, 0.08, 0.08, 0.30, True, 2.1334),
    Example("Merton put on an index paying 5%", "price", 75.0, 70.0, 0.5, 0.10, 0.05, 0.35, False, 4.0870),
    Example("Black-76 call on a future", "price", 19.0, 19.0, 0.75, 0.10, 0.0, 0.28, True, 1.7011),
    Example("Garman-Kohlhagen call on a currency", "price", 1.56, 1.60, 0.5, 0.06, 0.06 - 0.08, 0.12, True, 0.0291),
    Example("Delta of a futures call", "delta", 105.0, 100.0, 0.5, 0.10, 0.0, 0.36, True, 0.5946),
    Example("Delta of a futures put", "delta", 105.0, 100.0, 0.5, 0.10, 0.0, 0.36, False, -0.3566),
    Example("Gamma", "gamma", 55.0, 60.0, 0.75, 0.10, 0.10, 0.30, True, 0.0278),
    Example("Theta of a put on an index (per year)", "theta", 430.0, 405.0, 0.0833, 0.07, 0.02, 0.20, False, -31.1924),
    Example("Rho of a call", "rho", 72.0, 75.0, 1.0, 0.09, 0.09, 0.19, True, 38.7325),
)
