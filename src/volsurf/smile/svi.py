"""SVI smiles: three parameterisations, analytic derivatives, and arbitrage checks on the whole real line.

Raw SVI (Gatheral, 2004) gives total implied variance ``w = sigma^2 T`` in log-moneyness
``k = ln(K/F)``:

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)).

Gatheral and Jacquier (2014) add two equivalent forms: the *natural* parameters
``(Delta, mu, rho, omega, zeta)`` and the *jump-wings* parameters ``(v, psi, p, c, v~)``,
which a trader reads directly: ATM variance, ATM skew, the two wing slopes and the
minimum variance. Conversions between all three are exact.

**Butterfly arbitrage.** A slice is free of it if and only if the density of the
terminal log-price is non-negative, i.e. Durrleman's function

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2

is non-negative everywhere and ``d+(k) -> -inf`` as ``k -> inf`` (Roger Lee). Checking
``g`` on a grid misses arbitrage outside it. Here:

* the wings are settled analytically: as ``k -> +-inf``, ``g -> 1/4 - b^2(1 +- rho)^2/16``,
  non-negative if and only if the wing slope ``b(1 +- rho) <= 2`` (Lee's bound);
* the finite part is searched on a grid uniform in ``asinh((k - m)/sigma)``, which
  puts points where the curvature is, across 60 smile widths either side, and every
  local minimum is refined by Brent's method.

**Calendar arbitrage** between two slices means ``w_2(k) < w_1(k)`` somewhere. Far out,
the difference is linear in ``|k|`` with the slopes' difference; the finite part is
searched the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RawSVI:
    """One slice: ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0 or not -1 < self.rho < 1 or self.sigma <= 0:
            raise ValueError(f"not an SVI slice: b={self.b}, rho={self.rho}, sigma={self.sigma}")

    def total_variance(self, k: ArrayLike) -> Array:
        d = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.a + self.b * (self.rho * d + np.hypot(d, self.sigma)), dtype=np.float64)

    def dw(self, k: ArrayLike) -> Array:
        d = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.b * (self.rho + d / np.hypot(d, self.sigma)), dtype=np.float64)

    def d2w(self, k: ArrayLike) -> Array:
        d = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(self.b * self.sigma**2 / np.hypot(d, self.sigma) ** 3, dtype=np.float64)

    def implied_vol(self, k: ArrayLike, maturity: float) -> Array:
        return np.asarray(np.sqrt(np.maximum(self.total_variance(k), 0.0) / maturity), dtype=np.float64)

    def g(self, k: ArrayLike) -> Array:
        """Durrleman's function; the slice is butterfly-free if and only if it is non-negative everywhere."""
        return durrleman(np.asarray(k, dtype=np.float64), self.total_variance(k), self.dw(k), self.d2w(k))

    @property
    def wing_slopes(self) -> tuple[float, float]:
        """Slopes of ``w`` as ``k -> -inf`` and ``k -> +inf``: ``b(1 - rho)`` and ``b(1 + rho)``."""
        return self.b * (1.0 - self.rho), self.b * (1.0 + self.rho)

    @property
    def min_variance(self) -> float:
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    # ---- the other parameterisations (Gatheral and Jacquier, 2014, section 3)

    def natural(self) -> NaturalSVI:
        omega = 2.0 * self.b * self.sigma / np.sqrt(1.0 - self.rho**2)
        zeta = np.sqrt(1.0 - self.rho**2) / self.sigma
        mu = self.m + self.rho * self.sigma / np.sqrt(1.0 - self.rho**2)
        delta = self.a - 0.5 * omega * (1.0 - self.rho**2)
        return NaturalSVI(float(delta), float(mu), self.rho, float(omega), float(zeta))

    def jump_wings(self, maturity: float) -> JumpWings:
        w0 = float(self.total_variance(0.0))
        v = w0 / maturity
        root = np.sqrt(w0)
        psi = self.b / (2.0 * root) * (-self.m / np.hypot(self.m, self.sigma) + self.rho)
        p = self.b * (1.0 - self.rho) / root
        c = self.b * (1.0 + self.rho) / root
        v_min = self.min_variance / maturity
        return JumpWings(v, float(psi), float(p), float(c), float(v_min), maturity)


@dataclass(frozen=True)
class NaturalSVI:
    """``w(k) = Delta + omega/2 (1 + zeta rho (k - mu) + sqrt((zeta (k - mu) + rho)^2 + 1 - rho^2))``."""

    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def raw(self) -> RawSVI:
        root = np.sqrt(1.0 - self.rho**2)
        a = self.delta + 0.5 * self.omega * (1.0 - self.rho**2)
        b = 0.5 * self.omega * self.zeta
        m = self.mu - self.rho / self.zeta
        return RawSVI(float(a), float(b), self.rho, float(m), float(root / self.zeta))


@dataclass(frozen=True)
class JumpWings:
    """ATM variance ``v``, ATM skew ``psi``, put and call wing slopes ``p`` and ``c``, minimum variance ``v_min``."""

    v: float
    psi: float
    p: float
    c: float
    v_min: float
    maturity: float

    def raw(self) -> RawSVI:
        """Inverse of :meth:`RawSVI.jump_wings` (Gatheral and Jacquier, 2014, Lemma 3.2)."""
        w0 = self.v * self.maturity
        root = np.sqrt(w0)
        b = 0.5 * root * (self.c + self.p)
        rho = 1.0 - self.p * root / b
        beta = rho - 2.0 * self.psi * root / b  # = m / sqrt(m^2 + sigma^2)
        if abs(beta) < 1e-12:  # m = 0: w(0) = a + b sigma and v_min T = a + b sigma sqrt(1 - rho^2)
            if abs(rho) < 1e-12:
                raise ValueError("a symmetric slice with its minimum at the money: jump-wings do not determine sigma")
            m = 0.0
            sigma = (w0 - self.v_min * self.maturity) / (b * (1.0 - np.sqrt(1.0 - rho**2)))
        else:
            alpha = np.sign(beta) * np.sqrt(1.0 / beta**2 - 1.0)
            m = (
                (self.v - self.v_min)
                * self.maturity
                / (b * (-rho + np.sign(alpha) * np.sqrt(1.0 + alpha**2) - alpha * np.sqrt(1.0 - rho**2)))
            )
            sigma = alpha * m
        a = self.v_min * self.maturity - b * sigma * np.sqrt(1.0 - rho**2)
        return RawSVI(float(a), float(b), float(rho), float(m), float(sigma))


def durrleman(k: Array, w: Array, dw: Array, d2w: Array) -> Array:
    """``g`` from total variance and its first two derivatives in ``k``; ``-inf`` where ``w <= 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (1.0 - k * dw / (2.0 * w)) ** 2 - 0.25 * dw**2 * (1.0 / w + 0.25) + 0.5 * d2w
    return np.asarray(np.where(w > 0, value, -np.inf), dtype=np.float64)


@dataclass(frozen=True)
class Check:
    """The result of an arbitrage check: the worst value found, where, and whether it passes."""

    worst: float
    at: float
    passed: bool
    reason: str


def _search_grid(m: float, sigma: float, span: float = 60.0, points: int = 4001) -> Array:
    """Points uniform in ``asinh((k - m)/sigma)`` out to ``span`` smile widths, plus the real line out to |k| = 10."""
    y = np.linspace(-np.arcsinh(span), np.arcsinh(span), points)
    return np.asarray(np.unique(np.concatenate([m + sigma * np.sinh(y), np.linspace(-10.0, 10.0, 2001)])), dtype=np.float64)


def _refine_minimum(f: object, grid: Array, values: Array) -> tuple[float, float]:
    """Global minimum of ``f`` from a grid: every local minimum refined by Brent's method."""
    assert callable(f)
    interior = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1
    best_k, best = float(grid[int(np.argmin(values))]), float(np.min(values))
    for i in interior[np.argsort(values[interior])][:12]:
        result = minimize_scalar(lambda x: float(f(np.array([x]))[0]), bounds=(grid[i - 1], grid[i + 1]), method="bounded")
        if result.fun < best:
            best_k, best = float(result.x), float(result.fun)
    return best_k, best


def butterfly_check(slice_: RawSVI, tolerance: float = 1e-10) -> Check:
    """Whether ``slice_`` is free of butterfly arbitrage on the whole real line."""
    left, right = slice_.wing_slopes
    if max(left, right) > 2.0 + 1e-12:
        side = "right" if right > left else "left"
        return Check(
            float(0.25 - max(left, right) ** 2 / 16.0),
            float("inf") if side == "right" else float("-inf"),
            False,
            f"{side} wing slope above Lee's bound of 2",
        )
    grid = _search_grid(slice_.m, slice_.sigma)
    if np.any(slice_.total_variance(grid) <= 0):
        i = int(np.argmin(slice_.total_variance(grid)))
        return Check(float(slice_.total_variance(grid)[i]), float(grid[i]), False, "negative total variance")
    at, worst = _refine_minimum(slice_.g, grid, slice_.g(grid))
    wing_limit = 0.25 - max(left, right) ** 2 / 16.0
    if wing_limit < worst:
        worst, at = wing_limit, float("inf") if right >= left else float("-inf")
    return Check(worst, at, worst >= -tolerance, "density negative" if worst < -tolerance else "")


def calendar_check(earlier: RawSVI, later: RawSVI, tolerance: float = 1e-12) -> Check:
    """Whether ``later`` lies on or above ``earlier`` in total variance for every ``k``."""
    (l1, r1), (l2, r2) = earlier.wing_slopes, later.wing_slopes
    if l2 < l1 - 1e-12 or r2 < r1 - 1e-12:
        side = "left" if l2 < l1 else "right"
        return Check(
            float(min(l2 - l1, r2 - r1)), float("-inf") if side == "left" else float("inf"), False, f"{side} wing crosses"
        )
    centre, width = 0.5 * (earlier.m + later.m), max(min(earlier.sigma, later.sigma), 1e-4)
    grid = np.unique(
        np.concatenate([_search_grid(earlier.m, earlier.sigma), _search_grid(later.m, later.sigma), _search_grid(centre, width)])
    )

    def gap(k: Array) -> Array:
        return np.asarray(later.total_variance(k) - earlier.total_variance(k), dtype=np.float64)

    at, worst = _refine_minimum(gap, grid, gap(grid))
    return Check(worst, at, worst >= -tolerance, "slices cross" if worst < -tolerance else "")
