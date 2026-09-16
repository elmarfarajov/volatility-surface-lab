"""Vectorised, arbitrage-aware implied volatility inversion.

Design choices:

* Every quote is mapped to its out-of-the-money equivalent through put-call parity
  before inversion. OTM prices carry all of the time value, so the inversion is
  well conditioned even for deep in-the-money quotes.
* Prices are normalised by the forward, making tolerances scale-free.
* Newton's method starts at the inflection point of the price-volatility curve,
  sigma* = sqrt(2|ln(F/K)|/T) (Manaster & Koehler, 1982). From there Newton converges
  monotonically because the price is convex below sigma* and concave above it.
* Each iteration is safeguarded by a bisection bracket, so vanishing vega or a
  bad step can never escape the feasible region.
* Quotes violating static no-arbitrage bounds return NaN instead of a fake number.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from .black_scholes import norm_pdf

_VOL_LOWER = 1e-8
_VOL_UPPER = 10.0


def implied_vol_black76(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike = 1.0,
    is_call: ArrayLike = True,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    arrays = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(discount, dtype=float),
        np.asarray(is_call, dtype=bool),
    )
    shape = arrays[0].shape
    price, forward, strike, maturity, discount = (a.ravel().astype(float) for a in arrays[:5])
    is_call = arrays[5].ravel()

    moneyness = strike / forward
    use_call = moneyness >= 1.0
    undiscounted = price / (discount * forward)
    parity_shift = (1.0 - moneyness) * (use_call.astype(float) - is_call.astype(float))
    target = undiscounted + parity_shift
    upper_bound = np.where(use_call, 1.0, moneyness)

    valid = np.isfinite(target) & (target > 0.0) & (target < upper_bound) & (maturity > 0)
    sqrt_t = np.sqrt(np.where(maturity > 0, maturity, 1.0))
    log_m = np.log(moneyness)

    sigma = np.sqrt(2.0 * np.abs(log_m) / np.where(maturity > 0, maturity, 1.0))
    sigma = np.clip(sigma, 0.05, 2.0)
    lo = np.full(price.shape, _VOL_LOWER)
    hi = np.full(price.shape, _VOL_UPPER)
    active = valid.copy()

    for _ in range(max_iter):
        if not active.any():
            break
        s = sigma[active]
        tv = s * sqrt_t[active]
        d1 = -log_m[active] / tv + 0.5 * tv
        d2 = d1 - tv
        m = moneyness[active]
        uc = use_call[active]
        model = np.where(uc, ndtr(d1) - m * ndtr(d2), m * ndtr(-d2) - ndtr(-d1))
        diff = model - target[active]
        vega = norm_pdf(d1) * sqrt_t[active]

        lo_a = np.where(diff < 0, s, lo[active])
        hi_a = np.where(diff > 0, s, hi[active])
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s - diff / vega
        use_newton = np.isfinite(newton) & (newton > lo_a) & (newton < hi_a)
        s_next = np.where(use_newton, newton, 0.5 * (lo_a + hi_a))

        converged = (np.abs(diff) <= tol * np.maximum(target[active], 1e-300) + 1e-15) | (hi_a - lo_a < 1e-14)
        idx = np.flatnonzero(active)
        lo[idx], hi[idx] = lo_a, hi_a
        sigma[idx] = np.where(converged, s, s_next)
        active[idx[converged]] = False

    return np.where(valid, sigma, np.nan).reshape(shape)


def implied_vol_bsm(
    price: ArrayLike,
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    is_call: ArrayLike = True,
    **kwargs,
) -> np.ndarray:
    spot, maturity, rate, dividend_yield = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (spot, maturity, rate, dividend_yield))
    )
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    return implied_vol_black76(price, forward, strike, maturity, np.exp(-rate * maturity), is_call, **kwargs)
