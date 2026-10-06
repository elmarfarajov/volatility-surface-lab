"""The analytic foundation: generalised Black-Scholes-Merton, every Greek, and model-free bounds.

* :mod:`.carry` - one model for stocks, indices, futures and currencies, through the cost of carry;
* :mod:`.generalized` - the price and seventeen Greeks, first to third order, with their limits at expiry;
* :mod:`.bounds` - Merton's no-arbitrage bounds and the strike conditions any surface must meet;
* :mod:`.reference` - every Greek again, by arbitrary-precision differentiation, as an independent check;
* :mod:`.published` - worked examples from Haug (2007) the engine must reproduce as printed.
"""

from .bounds import Violation, bound_violations, merton_bounds, strike_violations
from .carry import Carry
from .generalized import Greeks, greeks, price

__all__ = ["Carry", "Greeks", "Violation", "bound_violations", "greeks", "merton_bounds", "price", "strike_violations"]
