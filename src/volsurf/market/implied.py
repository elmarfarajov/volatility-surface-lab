"""Implied volatility to machine precision, from the far tails to the upper price bound.

The approach follows Jaeckel's *Let's be rational* (2015): invert the normalised
Black function ``b(x, s)`` of :mod:`volsurf.market.black` for the total volatility
``s = sigma sqrt(T)``, on a transformed objective that is nearly linear in ``s``, with
third-order Householder steps whose derivatives are all closed-form,

    b'(s) = exp(-(h^2 + t^2)/2) / sqrt(2 pi),   b''/b' = x^2/s^3 - s/4,   b'''/b' = (b''/b')^2 - 3 x^2/s^4 - 1/4.

Two objectives split the price range at half the upper bound ``b_max = e^{x/2}``:

* **below**: ``f(s) = ln b(s) - ln beta``. In log space a price of ``1e-300`` is as
  easy as one of ``0.1``; the value and every derivative come from ``ln b`` and the
  log-vega, so nothing underflows;
* **above**: ``g(s) = ln(b_max - b(s)) - ln(b_max - beta)``, where the price is
  flattening against its bound and ``b`` itself carries no information about ``s``.
  ``b_max - b`` is a sum of two positive tails, computed without cancellation.

Both objectives are concave and monotone in ``s``, so a Householder step from either
side stays in a bracket that tightens with every evaluation; a step that would
leave it is replaced by bisection. Starting points come from asymptotics - the
small-``s`` expansion ``b ~ envelope * 2t/(1 + h^2)`` below, and
``b_max - b ~ 2 cosh(x/2) N(-s/2)`` above - so a few iterations suffice everywhere.

Where this differs from Jaeckel: his rational-cubic initial guesses reach full
precision in two iterations; the asymptotic starts here take a few more (the
distribution is measured in the gallery). Accuracy, the part that matters for
pricing, is the same: the residual is set by the conditioning of ``b`` itself.

Quotes outside the static bounds - below intrinsic value or above the forward -
have no implied volatility and return NaN. A quote exactly at intrinsic value has
zero time value and returns zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from .black import intrinsic, log_otm_complement, log_otm_value, log_vega

Array = NDArray[np.float64]

_EPS = np.finfo(np.float64).eps
_S_MAX = 60.0  # total volatility above which every price is its upper bound to double precision


@dataclass(frozen=True)
class Inversion:
    """Total implied volatility per quote, the iterations each took, and the branch used (0 below, 1 above)."""

    total_vol: Array
    iterations: NDArray[np.int64]
    branch: NDArray[np.int64]


def _householder(objective: Array, d1: Array, d2: Array, d3: Array) -> Array:
    """The third-order Householder step for a root of ``objective``, given its first three derivatives."""
    n = objective / d1
    gamma, delta = d2 / d1, d3 / d1
    return np.asarray(-n * (1.0 + 0.5 * gamma * n) / (1.0 + n * (gamma + delta * n / 6.0)), dtype=np.float64)


def _lower_start(x: Array, log_beta: Array) -> Array:
    """Solve the small-``s`` asymptote ``ln b ~ -h^2/2 - ln sqrt(2 pi) + ln(s/(1 + h^2))`` for ``s``.

    The right-hand side increases with ``s``, so bisection in ``ln s`` finds its root;
    it is exact as ``s -> 0`` and within a few per cent wherever the lower branch is used.
    """
    lo, hi = np.full_like(x, np.log(1e-12)), np.full_like(x, np.log(4.0))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = np.exp(mid)
        h2 = (x / s) ** 2
        value = -0.5 * h2 - 0.5 * np.log(2.0 * np.pi) + mid - np.log1p(h2)
        above = value > log_beta
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return np.asarray(np.exp(0.5 * (lo + hi)), dtype=np.float64)


def _upper_start(x: Array, log_gap: Array) -> Array:
    """Invert ``b_max - b ~ 2 cosh(x/2) N(-s/2)``, exact at the money."""
    ratio = np.exp(log_gap) / (2.0 * np.cosh(0.5 * x))
    return np.asarray(np.clip(-2.0 * ndtri(np.clip(ratio, 1e-300, 0.5)), 1e-8, _S_MAX), dtype=np.float64)


def invert_normalised(beta: ArrayLike, x: ArrayLike, is_call: ArrayLike = True, max_iter: int = 60) -> Inversion:
    """Total implied volatility ``s`` with ``b(x, s) = beta`` for normalised prices ``beta = P / (D sqrt(F K))``."""
    beta, x, call = np.broadcast_arrays(
        np.asarray(beta, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(is_call, dtype=bool)
    )
    shape = beta.shape
    beta, x, call = beta.ravel(), x.ravel(), call.ravel()
    otm = beta - intrinsic(x, call)  # the out-of-the-money equivalent (put-call parity)
    xo = -np.abs(x)
    b_max = np.exp(0.5 * xo)
    total = np.full(beta.shape, np.nan)
    iterations = np.zeros(beta.shape, dtype=np.int64)
    branch = np.zeros(beta.shape, dtype=np.int64)
    # a time value within rounding of zero (the normalisation itself rounds) is zero time value
    at_intrinsic = np.abs(otm) <= 4.0 * _EPS * np.maximum(beta, np.finfo(np.float64).tiny)
    total[at_intrinsic] = 0.0
    live = np.flatnonzero(~at_intrinsic & (otm > 0) & (otm < b_max) & np.isfinite(otm) & np.isfinite(x))
    if live.size == 0:
        return Inversion(total.reshape(shape), iterations.reshape(shape), branch.reshape(shape))

    xl, bl, bmax = xo[live], otm[live], b_max[live]
    upper = bl > 0.5 * bmax
    target = np.where(upper, np.log(np.where(upper, bmax - bl, 1.0)), np.log(bl))
    s = np.where(upper, _upper_start(xl, target), _lower_start(xl, target))
    lo, hi = np.zeros_like(s), np.full_like(s, _S_MAX)
    active = np.ones(s.shape, dtype=bool)
    count = np.zeros(s.shape, dtype=np.int64)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        si, xi, up = s[idx], xl[idx], upper[idx]
        lv = log_vega(xi, si)
        a = xi * xi / si**3 - 0.25 * si  # b''/b'
        bb = a * a - 3.0 * xi * xi / si**4 - 0.25  # b'''/b'
        level = np.where(up, log_otm_complement(xi, si), log_otm_value(xi, si))
        ratio = np.exp(lv - level)  # b'/b below, b'/(b_max - b) above
        sign = np.where(up, -1.0, 1.0)
        # derivatives of ln b (below) or ln(b_max - b) (above), whose sign flips with b' -> -b'
        d1 = sign * ratio
        d2 = sign * ratio * a - ratio**2
        d3 = sign * ratio * bb - 3.0 * ratio**2 * a + sign * 2.0 * ratio**3
        objective = level - target[idx]
        # tighten the bracket: the objective increases with s below and decreases above
        too_high = sign * objective > 0
        hi[idx] = np.where(too_high, np.minimum(hi[idx], si), hi[idx])
        lo[idx] = np.where(too_high, lo[idx], np.maximum(lo[idx], si))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            step = _householder(objective, d1, d2, d3)
        new = si + step
        # converged when the step, or the objective, is down to the rounding in evaluating it
        noise = 8.0 * _EPS * (1.0 + np.abs(target[idx]))
        done = (np.abs(objective) <= noise) | (np.abs(step) <= 8.0 * _EPS * si) | (hi[idx] - lo[idx] <= 8.0 * _EPS * si)
        outside = ~done & (~np.isfinite(new) | (new < lo[idx]) | (new > hi[idx]))
        # bisect - geometrically while the bracket spans orders of magnitude
        wide = hi[idx] > 4.0 * lo[idx]
        middle = np.where(wide, np.sqrt(np.maximum(lo[idx], 1e-16 * hi[idx]) * hi[idx]), 0.5 * (lo[idx] + hi[idx]))
        new = np.where(outside, middle, np.where(objective == 0.0, si, new))
        s[idx] = new
        count[idx] += 1
        active[idx[done]] = False
    total[live] = s
    iterations[live] = count
    branch[live] = upper.astype(np.int64)
    return Inversion(total.reshape(shape), iterations.reshape(shape), branch.reshape(shape))


def implied_vol(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    discount: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> Array:
    """Black-76 implied volatility of discounted option prices; NaN outside the no-arbitrage bounds."""
    p, f, k, t, d, c = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (price, forward, strike, maturity, discount)), np.asarray(is_call, dtype=bool)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = p / (d * np.sqrt(f * k))
        x = np.log(f / k)
        s = invert_normalised(beta, x, c).total_vol.reshape(p.shape)
        return np.asarray(np.where(t > 0, s / np.sqrt(np.where(t > 0, t, 1.0)), np.nan), dtype=np.float64)
