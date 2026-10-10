"""The normalised Black function, accurate to machine precision everywhere it is representable.

With ``x = ln(F/K)`` and total volatility ``s = sigma sqrt(T)``, an undiscounted
Black call divided by ``sqrt(F K)`` is

    b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2).

Out of the money (``x <= 0``) the two terms are nearly equal whenever ``s`` is
small, and computing them separately cancels away every significant digit: at
``x = -0.15, s = 0.004`` the textbook formula returns noise. The same is true of
``F N(d1) - K N(d2)`` in any pricer that evaluates it literally.

Writing ``h = x/s`` and ``t = s/2``, the Gaussian factors combine exactly
(Jaeckel, 2015, eq. 2.4):

    b(x, s) = phi-envelope * [Y(h + t) - Y(h - t)],   envelope = exp(-(h^2 + t^2)/2) / sqrt(2 pi),

where ``Y(z) = N(z)/phi(z)`` is the Mills ratio. The envelope is the normalised
vega, so ``ln b`` is available even when ``b`` itself underflows. What is left is the
difference of Mills ratios, which cancels when ``t`` is small against ``|h|``; there
it is computed as an integral instead,

    Y(h + t) - Y(h - t) = int_{h-t}^{h+t} Y'(z) dz,   Y'(z) = 1 + z Y(z),

by 16-point Gauss-Legendre - exact to rounding, because ``Y'`` is entire and varies
slowly over the interval. ``1 + z Y(z)`` cancels in its turn for ``z << 0``, so
there it comes from Laplace's continued fraction for the Mills ratio, which has no
subtraction at all. Elsewhere the two terms differ by a factor of 1.4 or more and
are evaluated directly in log space.

Puts and in-the-money options follow from ``b_put(x, s) = b_call(-x, s)`` and the
intrinsic value, so only the out-of-the-money branch ever needs care.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfcx, log_ndtr

Array = NDArray[np.float64]

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_SQRT_HALF_PI = np.sqrt(0.5 * np.pi)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_CF_TERMS = 40  # Laplace's continued fraction; converged to rounding for z <= -4 (checked against mpmath)


def mills_ratio(z: ArrayLike) -> Array:
    """``Y(z) = N(z) / phi(z)``, without overflow for negative ``z``."""
    z = np.asarray(z, dtype=np.float64)
    return np.asarray(_SQRT_HALF_PI * erfcx(-z / np.sqrt(2.0)), dtype=np.float64)


def mills_ratio_slope(z: ArrayLike) -> Array:
    """``Y'(z) = 1 + z Y(z)``; from the continued fraction for ``z < -4``, where the sum cancels."""
    z = np.asarray(z, dtype=np.float64)
    slope = 1.0 + z * mills_ratio(z)
    far = z < -4.0
    if np.any(far):
        u = -z[far]
        tail = np.zeros_like(u)
        for k in range(_CF_TERMS, 1, -1):
            tail = k / (u + tail)
        first = 1.0 / (u + tail)  # 1 / (u + 2 / (u + 3 / ...)), so that Y = 1 / (u + first)
        slope[far] = first / (u + first)
    return slope


def _otm_log_value(x: Array, s: Array) -> Array:
    """``ln b(x, s)`` for ``x <= 0`` and ``s > 0``."""
    h, t = x / s, 0.5 * s
    out = np.empty_like(x)
    by_quadrature = t < np.maximum(0.5, 0.25 * np.abs(h))
    q = by_quadrature
    if np.any(q):
        hq, tq = h[q], t[q]
        z = hq[:, None] + tq[:, None] * _GL_NODES[None, :]
        difference = tq * (mills_ratio_slope(z.ravel()).reshape(z.shape) @ _GL_WEIGHTS)
        out[q] = -0.5 * (hq * hq + tq * tq) - _LOG_SQRT_2PI + np.log(difference)
    d = ~q
    if np.any(d):
        xd, hd, td = x[d], h[d], t[d]
        upper, lower = log_ndtr(hd + td), log_ndtr(hd - td)
        # the second term is at most about 0.7 of the first here (it is Y(h - t) / Y(h + t))
        out[d] = 0.5 * xd + upper + np.log1p(-np.exp(lower - upper - xd))
    return out


def log_vega(x: ArrayLike, s: ArrayLike) -> Array:
    """``ln db/ds``: the normalised vega, ``exp(-(h^2 + t^2)/2) / sqrt(2 pi)``."""
    x, s = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(s, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(s > 0, x / np.where(s > 0, s, 1.0), np.where(x == 0, 0.0, -np.inf))
    return np.asarray(-0.5 * (h * h + 0.25 * s * s) - _LOG_SQRT_2PI, dtype=np.float64)


def log_otm_value(x: ArrayLike, s: ArrayLike) -> Array:
    """``ln`` of the out-of-the-money normalised price ``b(-|x|, s)``; ``-inf`` at ``s = 0``."""
    x, s = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(s, dtype=np.float64))
    x = -np.abs(x)
    out = np.full(x.shape, -np.inf)
    live = s > 0
    if np.any(live):
        out[live] = _otm_log_value(x[live], s[live])
    return out


def log_otm_complement(x: ArrayLike, s: ArrayLike) -> Array:
    """``ln(e^{-|x|/2} - b(-|x|, s))``: how far the OTM price is below its upper bound. No cancellation."""
    x, s = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(s, dtype=np.float64))
    x = -np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(s > 0, x / np.where(s > 0, s, 1.0), np.where(x == 0, 0.0, -np.inf))
    t = 0.5 * s
    return np.asarray(np.logaddexp(0.5 * x + log_ndtr(-h - t), -0.5 * x + log_ndtr(h - t)), dtype=np.float64)


def intrinsic(x: ArrayLike, is_call: ArrayLike = True) -> Array:
    """The normalised intrinsic value, ``max(theta (e^{x/2} - e^{-x/2}), 0)``."""
    x = np.asarray(x, dtype=np.float64)
    theta = np.where(np.asarray(is_call, dtype=bool), 1.0, -1.0)
    return np.asarray(np.maximum(2.0 * theta * np.sinh(0.5 * x), 0.0), dtype=np.float64)


def normalised_black(x: ArrayLike, s: ArrayLike, is_call: ArrayLike = True) -> Array:
    """``b(x, s)`` for a call (or the put ``b(-x, s)``): intrinsic value plus the OTM value."""
    x, s, is_call = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(s, dtype=np.float64), np.asarray(is_call, dtype=bool)
    )
    return np.asarray(intrinsic(x, is_call) + np.exp(log_otm_value(x, s)), dtype=np.float64)


def black_price(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    vol: ArrayLike,
    discount: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> Array:
    """The discounted Black-76 price, accurate to rounding in the far tails as well as near the money."""
    f, k, t, v, d = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (forward, strike, maturity, vol, discount)))
    s = v * np.sqrt(np.maximum(t, 0.0))
    return np.asarray(d * np.sqrt(f * k) * normalised_black(np.log(f / k), s, is_call), dtype=np.float64)
