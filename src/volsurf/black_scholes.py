"""Black-76 and Black-Scholes-Merton pricing with analytic Greeks.

The core is written in forward form (Black-76): every European price depends only
on the forward F, the discount factor D, the strike, the maturity and the volatility.
Spot-based Black-Scholes-Merton is the special case F = S*exp((r-q)T), D = exp(-rT).
All functions broadcast over NumPy arrays so a whole option chain is priced in one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _as_float_arrays(*values: ArrayLike) -> list[np.ndarray]:
    return list(np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values)))


def black76_price(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike,
    vol: ArrayLike,
    is_call: ArrayLike = True,
) -> np.ndarray:
    """Discounted Black-76 price. Calls and puts are both computed directly (no parity
    subtraction), which keeps deep out-of-the-money prices accurate to machine precision."""
    forward, strike, maturity, discount, vol = _as_float_arrays(forward, strike, maturity, discount, vol)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), forward.shape)

    total_vol = vol * np.sqrt(maturity)
    positive = total_vol > 0
    safe_total_vol = np.where(positive, total_vol, 1.0)
    d1 = np.log(forward / strike) / safe_total_vol + 0.5 * safe_total_vol
    d2 = d1 - safe_total_vol

    call = forward * ndtr(d1) - strike * ndtr(d2)
    put = strike * ndtr(-d2) - forward * ndtr(-d1)
    undiscounted = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(forward - strike, 0.0), np.maximum(strike - forward, 0.0))
    return discount * np.where(positive, undiscounted, intrinsic)


def bsm_price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
    is_call: ArrayLike = True,
) -> np.ndarray:
    spot, strike, maturity, rate, dividend_yield = _as_float_arrays(spot, strike, maturity, rate, dividend_yield)
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    return black76_price(forward, strike, maturity, np.exp(-rate * maturity), vol, is_call)


def black76_vega(forward: ArrayLike, strike: ArrayLike, maturity: ArrayLike, discount: ArrayLike, vol: ArrayLike) -> np.ndarray:
    forward, strike, maturity, discount, vol = _as_float_arrays(forward, strike, maturity, discount, vol)
    sqrt_t = np.sqrt(maturity)
    total_vol = np.maximum(vol * sqrt_t, 1e-300)
    d1 = np.log(forward / strike) / total_vol + 0.5 * total_vol
    return discount * forward * norm_pdf(d1) * sqrt_t


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of a Black-Scholes-Merton price. Theta is per year, vega/rho per unit (not per 1%)."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray


def bsm_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    dividend_yield: ArrayLike,
    vol: ArrayLike,
    is_call: ArrayLike = True,
) -> Greeks:
    spot, strike, maturity, rate, q, vol = _as_float_arrays(spot, strike, maturity, rate, dividend_yield, vol)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), spot.shape)

    sqrt_t = np.sqrt(maturity)
    total_vol = vol * sqrt_t
    d1 = (np.log(spot / strike) + (rate - q + 0.5 * vol**2) * maturity) / total_vol
    d2 = d1 - total_vol
    df_r = np.exp(-rate * maturity)
    df_q = np.exp(-q * maturity)
    pdf_d1 = norm_pdf(d1)

    price = bsm_price(spot, strike, maturity, rate, q, vol, is_call)
    delta = np.where(is_call, df_q * ndtr(d1), df_q * (ndtr(d1) - 1.0))
    gamma = df_q * pdf_d1 / (spot * total_vol)
    vega = spot * df_q * pdf_d1 * sqrt_t
    decay = -spot * df_q * pdf_d1 * vol / (2.0 * sqrt_t)
    theta = np.where(
        is_call,
        decay - rate * strike * df_r * ndtr(d2) + q * spot * df_q * ndtr(d1),
        decay + rate * strike * df_r * ndtr(-d2) - q * spot * df_q * ndtr(-d1),
    )
    rho = np.where(is_call, strike * maturity * df_r * ndtr(d2), -strike * maturity * df_r * ndtr(-d2))
    vanna = -df_q * pdf_d1 * d2 / vol
    volga = vega * d1 * d2 / vol
    return Greeks(price, delta, gamma, vega, theta, rho, vanna, volga)
