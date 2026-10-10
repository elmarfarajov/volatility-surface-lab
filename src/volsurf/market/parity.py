"""The forward and the discount factor of each expiry, read from put-call parity.

For European options on the same strike and expiry, a long call and a short put
is a forward contract, whatever the model:

    C(K) - P(K) = D (F - K).

Across strikes the synthetic forward ``C - P`` is a straight line in ``K`` with
slope ``-D`` and intercept ``D F``. Fitting it recovers the forward - and with it
the dividends or carry the market is pricing - and the discount factor of the
option market's own funding, with no rate curve or dividend forecast assumed.

The fit is weighted least squares with weights ``1/w^2``, where ``w`` is the
half-width of the synthetic's bid-ask range, ``(C_ask - C_bid + P_ask - P_bid)/2``:
a quote is trusted in proportion to how tightly it is made. Pairs whose residual
exceeds ``trim`` half-widths are dropped and the fit repeated, so a stale or
mistyped quote cannot tilt the line. Standard errors come from the weighted
residuals, and the forward's by the delta method.

The fit is checked against the market it came from: the fitted line should pass
through the executable range ``[C_bid - P_ask, C_ask - P_bid]`` of every pair it
used (``inside`` is the fraction that do). A line outside the box would be an
arbitrage against the synthetic forward.

American options (single stocks, ETFs) satisfy parity only as an inequality:
early exercise makes the in-the-money put worth more than its European twin, so
``C - P`` bends below the line deep in the money. ``band`` restricts the fit to
strikes near the spot, where the early-exercise premium is small; the residuals
outside the band then show the premium itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ParityFit:
    """The line ``C - P = D (F - K)`` through one expiry's quotes, with its uncertainty."""

    forward: float
    discount: float
    forward_se: float
    discount_se: float
    maturity: float
    spot: float
    strikes: Array = field(repr=False)
    synthetic_bid: Array = field(repr=False)  # C_bid - P_ask
    synthetic_ask: Array = field(repr=False)  # C_ask - P_bid
    used: NDArray[np.bool_] = field(repr=False)
    quoted: NDArray[np.bool_] = field(repr=False)  # both options bid, so the pair is a candidate at all

    @property
    def synthetic_mid(self) -> Array:
        return np.asarray(0.5 * (self.synthetic_bid + self.synthetic_ask), dtype=np.float64)

    @property
    def residuals(self) -> Array:
        return np.asarray(self.synthetic_mid - self.discount * (self.forward - self.strikes), dtype=np.float64)

    @property
    def rate(self) -> float:
        """The continuously compounded rate the options imply, ``-ln D / T``."""
        return float(-np.log(self.discount) / self.maturity)

    @property
    def carry(self) -> float:
        """The yield the forward implies, ``r - ln(F/S)/T``: dividends, or for an index its dividend yield."""
        return float(self.rate - np.log(self.forward / self.spot) / self.maturity)

    @property
    def inside(self) -> float:
        """The fraction of fitted pairs whose bid-ask range contains the fitted line."""
        line = self.discount * (self.forward - self.strikes[self.used])
        within = (line >= self.synthetic_bid[self.used] - 1e-9) & (line <= self.synthetic_ask[self.used] + 1e-9)
        return float(within.mean()) if within.size else float("nan")

    @property
    def pairs(self) -> int:
        return int(self.used.sum())


def fit_forward(
    strikes: Array,
    call_bid: Array,
    call_ask: Array,
    put_bid: Array,
    put_ask: Array,
    spot: float,
    maturity: float,
    band: float | None = None,
    discount: float | None = None,
    trim: float = 3.0,
    min_pairs: int = 5,
) -> ParityFit:
    """Weighted, trimmed least squares of the synthetic forward on the strike.

    The fit starts from the dozen pairs nearest the money - where both options are
    liquid and a stale quote far away cannot pull it - and grows to every pair within
    ``trim`` half-widths of the line, refitting until the set no longer changes.

    ``band``, if given, keeps strikes with ``|ln(K/S)| <= band`` (for American options).
    ``discount``, if given, is held fixed and only the forward is fitted - for American
    options, whose parity slope is bent by early exercise, with ``D`` from a European market.
    Raises ``ValueError`` when fewer than ``min_pairs`` usable pairs remain.
    """
    k = np.asarray(strikes, dtype=np.float64)
    low = np.asarray(call_bid, dtype=np.float64) - np.asarray(put_ask, dtype=np.float64)
    high = np.asarray(call_ask, dtype=np.float64) - np.asarray(put_bid, dtype=np.float64)
    quoted = (np.asarray(call_bid) > 0) & (np.asarray(put_bid) > 0) & (high > low)
    if band is not None:
        quoted &= np.abs(np.log(k / spot)) <= band
    if quoted.sum() < min_pairs:
        raise ValueError(f"only {int(quoted.sum())} call-put pairs with two-sided quotes; need {min_pairs}")
    y, half = 0.5 * (low + high), 0.5 * (high - low)
    fixed = discount is not None
    start = _repeated_median(k[quoted], y[quoted], discount)
    used = quoted & (np.abs(y - (start[0] + start[1] * k)) <= trim * half)
    if used.sum() < min_pairs:  # the robust line misses every box by a little: take the pairs closest to it
        closest = np.argsort(np.where(quoted, np.abs(y - (start[0] + start[1] * k)) / np.where(quoted, half, 1.0), np.inf))
        used = np.zeros(k.shape, dtype=bool)
        used[closest[:min_pairs]] = True
    for _ in range(20):
        coef, cov = _weighted_line(k[used], y[used], half[used], discount)
        scaled = np.where(quoted, (y - (coef[0] + coef[1] * k)) / np.where(quoted, half, 1.0), np.inf)
        keep = quoted & (np.abs(scaled) <= trim)
        if keep.sum() < min_pairs or np.array_equal(keep, used):
            break
        used = keep
    intercept, slope = float(coef[0]), float(coef[1])
    d, forward = -slope, -intercept / slope
    grad = np.array([-1.0 / slope, intercept / slope**2])  # dF/d(intercept, slope)
    forward_se = float(np.sqrt(grad @ cov @ grad))
    return ParityFit(
        forward, d, forward_se, 0.0 if fixed else float(np.sqrt(cov[1, 1])), maturity, spot, k, low, high, used, quoted
    )


def _repeated_median(k: Array, y: Array, discount: float | None) -> Array:
    """Siegel's (1982) repeated-median line: robust to up to half the points, wherever they sit in strike."""
    if discount is not None:
        return np.array([float(np.median(y + discount * k)), -discount])
    dk = k[None, :] - k[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dk != 0, (y[None, :] - y[:, None]) / dk, np.nan)
    slope = float(np.nanmedian(np.nanmedian(slopes, axis=1)))
    return np.array([float(np.median(y - slope * k)), slope])


def _weighted_line(k: Array, y: Array, half: Array, discount: float | None) -> tuple[Array, Array]:
    """Coefficients (intercept, slope) and their covariance, with weights ``1/half^2``; the slope fixed at ``-D`` if given."""
    w = 1.0 / half**2
    if discount is not None:
        # only the intercept D F is free: its weighted mean, given the slope
        intercept = float(np.sum(w * (y + discount * k)) / np.sum(w))
        residual = y - (intercept - discount * k)
        scale = float(np.sum(w * residual**2) / max(k.size - 1, 1))
        cov = np.array([[scale / np.sum(w), 0.0], [0.0, 0.0]])
        return np.array([intercept, -discount]), cov
    design = np.column_stack([np.ones(k.size), k])
    normal = design.T @ (design * w[:, None])
    coef = np.linalg.solve(normal, design.T @ (w * y))
    residual = y - design @ coef
    scale = float(np.sum(w * residual**2) / max(k.size - 2, 1))
    return coef, scale * np.linalg.inv(normal)


@dataclass(frozen=True)
class DiscountCurve:
    """A smooth zero-rate curve through the parity fits: Nelson and Siegel's (1987) three factors,

        r(T) = level + slope L(T/tau) + curvature (L(T/tau) - e^{-T/tau}),   L(x) = (1 - e^{-x}) / x.

    A single expiry pins its discount factor to within ``D_se``, so its rate to within
    ``D_se / (D T)``: a few basis points a year out, several per cent a week out. One
    curve fitted to every reliable expiry with weights ``1/se(r)^2`` lends the long
    expiries' precision to the short ones.
    """

    level: float
    slope: float
    curvature: float
    tau: float
    expiries: int

    def rate(self, maturity: ArrayLike) -> Array:
        return np.asarray(_loadings(np.asarray(maturity, dtype=np.float64), self.tau) @ self.coefficients, dtype=np.float64)

    @property
    def coefficients(self) -> Array:
        return np.array([self.level, self.slope, self.curvature])

    def discount(self, maturity: ArrayLike) -> Array:
        return np.asarray(np.exp(-self.rate(maturity) * np.asarray(maturity, dtype=np.float64)), dtype=np.float64)


def fit_curve(fits: list[ParityFit], tau: float = 1.0, min_pairs: int = 10, min_inside: float = 0.8) -> DiscountCurve:
    """Weighted least squares of each reliable expiry's implied rate on the Nelson-Siegel loadings."""
    good = [f for f in fits if f.pairs >= min_pairs and f.inside >= min_inside and f.discount_se > 0]
    if len(good) < 2:
        raise ValueError(f"{len(good)} reliable expiries; a curve needs two")
    t = np.array([f.maturity for f in good])
    r = np.array([f.rate for f in good])
    se = np.array([f.discount_se / (f.discount * f.maturity) for f in good])
    design = _loadings(t, tau)
    w = 1.0 / se**2
    coef = np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * r))
    return DiscountCurve(float(coef[0]), float(coef[1]), float(coef[2]), tau, len(good))


def _loadings(maturity: Array, tau: float) -> Array:
    x = np.maximum(np.atleast_1d(maturity), 1e-12) / tau
    slope = -np.expm1(-x) / x
    return np.column_stack([np.ones_like(x), slope, slope - np.exp(-x)]).reshape(*np.shape(maturity), 3)
