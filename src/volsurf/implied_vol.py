"""Implied volatility for the original modules: a thin layer over :mod:`volsurf.market.implied`.

The first version of this module ran a safeguarded Newton iteration on the price,
started at the Manaster-Koehler inflection point. It was fast and accurate to 4e-12
above a normalised price of 1e-5, which covers every real quote (a 0.05 bid on SPX is
about 6e-6). But its stopping rule was absolute in price, so below that it stopped
early: wrong by up to 6e-8 down to 1e-10, 18% by 1e-20 and a factor of four by
1e-300, measured on Day 3 against 60-digit arithmetic. Model prices reach those
levels. The inversion is now Jaeckel-style, on log-price objectives with third-order
steps, and accurate to a few times the conditioning of the problem everywhere; see
ADR 0010.

Quotes outside the static bounds return NaN; a quote at intrinsic value returns zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .market.implied import implied_vol


def implied_vol_black76(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> np.ndarray:
    """Black-76 implied volatility of discounted prices."""
    return implied_vol(price, forward, strike, maturity, discount, is_call)


def implied_vol_bsm(
    price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    is_call: ArrayLike = True,
) -> np.ndarray:
    """Black-Scholes-Merton implied volatility, through the forward ``S e^{(r-q)T}`` and discount ``e^{-rT}``."""
    spot, maturity, rate, dividend_yield = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (spot, maturity, rate, dividend_yield))
    )
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    return implied_vol_black76(price, forward, strike, maturity, np.exp(-rate * maturity), is_call)
