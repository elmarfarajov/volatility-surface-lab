"""SSVI and extended SSVI: whole surfaces that are free of static arbitrage by construction.

**SSVI** (Gatheral and Jacquier, 2014) writes every slice in terms of its ATM total variance
``theta_t``:

    w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)).

With the power-law ``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))``, the surface
is free of static arbitrage whenever

* ``theta_t`` is non-decreasing in ``t`` (Theorem 4.1, calendar);
* ``0 < gamma <= 1/2``, so that ``0 <= d(theta phi)/d theta <= phi``, which is at most the
  theorem's bound ``(1 + sqrt(1 - rho^2)) phi / rho^2``;
* ``eta (1 + |rho|) <= 2`` (Remark 4.4, butterfly), which gives both
  ``theta phi (1 + |rho|) < 4`` and ``theta phi^2 (1 + |rho|) <= 4``.

The calibration below enforces these through its parameterisation, so every fit is
arbitrage-free by the theorem, and the checks of :mod:`volsurf.smile.svi` confirm it.

**eSSVI** (Hendriks and Martini, 2019; Corbetta, Cohort, Laachir and Martini, 2019) lets each
slice have its own correlation: slice ``i`` is ``(theta_i, rho_i, psi_i)`` with
``psi = theta phi``. Slices are fitted in order of maturity under

* butterfly: ``psi (1 + |rho|) <= 4`` and ``psi^2 (1 + |rho|) <= 4 theta``;
* calendar against the previous slice: ``theta_i >= theta_{i-1}`` and
  ``|rho_i psi_i - rho_{i-1} psi_{i-1}| <= psi_i - psi_{i-1}``, so that both wing slopes rise.

The calendar condition used here is the one of Corbetta et al.; every fitted pair is then
checked on the whole real line rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .svi import RawSVI

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SSVISlice:
    """One (e)SSVI slice: ATM total variance ``theta``, correlation ``rho`` and ``psi = theta phi``."""

    theta: float
    rho: float
    psi: float

    def total_variance(self, k: ArrayLike) -> Array:
        phi = self.psi / self.theta
        x = phi * np.asarray(k, dtype=np.float64)
        return np.asarray(
            0.5 * self.theta * (1.0 + self.rho * x + np.sqrt((x + self.rho) ** 2 + 1.0 - self.rho**2)), dtype=np.float64
        )

    def raw(self) -> RawSVI:
        """The same slice as raw SVI (Gatheral and Jacquier, 2014, section 4)."""
        phi = self.psi / self.theta
        return RawSVI(
            a=0.5 * self.theta * (1.0 - self.rho**2),
            b=0.5 * self.psi,
            rho=self.rho,
            m=-self.rho / phi,
            sigma=float(np.sqrt(1.0 - self.rho**2) / phi),
        )

    def butterfly_margin(self) -> float:
        """The slack in the two butterfly conditions; non-negative means arbitrage-free."""
        lee = 4.0 - self.psi * (1.0 + abs(self.rho))
        curvature = 4.0 * self.theta - self.psi**2 * (1.0 + abs(self.rho))
        return float(min(lee, curvature))


def calendar_margin(earlier: SSVISlice, later: SSVISlice) -> float:
    """The slack in Corbetta et al.'s calendar condition between consecutive slices."""
    rise = later.psi - earlier.psi - abs(later.rho * later.psi - earlier.rho * earlier.psi)
    return float(min(later.theta - earlier.theta, rise))


@dataclass(frozen=True)
class SSVI:
    """A whole SSVI surface with the power-law ``phi``: ``rho``, ``eta``, ``gamma`` and ``theta`` per maturity."""

    rho: float
    eta: float
    gamma: float
    maturities: Array
    thetas: Array

    def phi(self, theta: ArrayLike) -> Array:
        th = np.asarray(theta, dtype=np.float64)
        return np.asarray(self.eta / (th**self.gamma * (1.0 + th) ** (1.0 - self.gamma)), dtype=np.float64)

    def slice(self, i: int) -> SSVISlice:
        theta = float(self.thetas[i])
        return SSVISlice(theta, self.rho, float(theta * self.phi(theta)))

    def slices(self) -> list[SSVISlice]:
        return [self.slice(i) for i in range(len(self.thetas))]


@dataclass(frozen=True)
class Quotes:
    """One expiry's market: log-moneyness, mid implied volatility, and a weight per quote."""

    maturity: float
    k: Array
    iv: Array
    weight: Array


def _residuals(slice_: SSVISlice | RawSVI, q: Quotes) -> Array:
    w = np.maximum(slice_.total_variance(q.k), 1e-12)
    return np.asarray(np.sqrt(q.weight) * (np.sqrt(w / q.maturity) - q.iv), dtype=np.float64)


def fit_ssvi(quotes: list[Quotes]) -> SSVI:
    """Least squares in implied volatility over every slice at once, arbitrage-free by parameterisation.

    Parameters: ``rho``; ``u`` in ``(0, 1]`` with ``eta = 2u/(1 + |rho|)``; ``gamma`` in ``(0, 1/2]``;
    the first ``theta`` and non-negative increments of the rest.
    """
    quotes = sorted(quotes, key=lambda q: q.maturity)
    atm = np.array([max(float(np.interp(0.0, q.k, q.iv)) ** 2 * q.maturity, 1e-6) for q in quotes])
    atm = np.maximum.accumulate(atm)
    n = len(quotes)

    def unpack(p: Array) -> SSVI:
        rho, u, gamma = p[0], p[1], p[2]
        thetas = np.cumsum(p[3:])
        return SSVI(float(rho), float(2.0 * u / (1.0 + abs(rho))), float(gamma), np.array([q.maturity for q in quotes]), thetas)

    def residuals(p: Array) -> Array:
        surface = unpack(p)
        return np.concatenate([_residuals(surface.slice(i), q) for i, q in enumerate(quotes)])

    start = np.concatenate([[-0.7, 0.6, 0.4], [atm[0]], np.maximum(np.diff(atm), 1e-8)])
    lower = np.concatenate([[-0.999, 1e-4, 1e-3], np.full(n, 0.0)])
    upper = np.concatenate([[0.999, 1.0, 0.5], np.full(n, np.inf)])
    lower[3] = 1e-8
    result = least_squares(
        residuals, np.clip(start, lower + 1e-9, upper - 1e-9), bounds=(lower, upper), x_scale="jac", max_nfev=5000
    )
    return unpack(result.x)


def psi_bounds(rho: float, theta: float, previous: SSVISlice | None) -> tuple[float, float]:
    """The interval of ``psi`` allowed at ``(rho, theta)``: rising wing slopes below, the butterfly conditions above."""
    if previous is None:
        low = 0.0
    else:
        low = previous.psi * max((1.0 - previous.rho) / (1.0 - rho), (1.0 + previous.rho) / (1.0 + rho))
    high = min(4.0 / (1.0 + abs(rho)), 2.0 * np.sqrt(theta / (1.0 + abs(rho))))
    return float(low), float(high)


def _asinh_grid(centre: float, width: float, points: int = 241) -> Array:
    y = np.linspace(-np.arcsinh(40.0), np.arcsinh(40.0), points)
    return np.asarray(centre + width * np.sinh(y), dtype=np.float64)


def fit_essvi(quotes: list[Quotes], calendar_penalty: float = 1e4) -> list[SSVISlice]:
    """Fit eSSVI slices in order of maturity, each arbitrage-free on its own and against the slice before.

    Each slice has three parameters, mapped so that the box bounds of a trust-region least
    squares carry the conditions: ``theta >= theta_prev``; ``rho`` in ``(-1, 1)``; and ``u`` in
    ``[0, 1]`` placing ``psi`` in the interval of :func:`psi_bounds`, which holds the butterfly
    conditions and Corbetta et al.'s rising wing slopes. Rising wings and ATM variance do not
    stop two slices from crossing in between, so a penalty on a grid around the smile keeps
    the new slice above the old one, and every pair is checked on the whole line afterwards.
    """
    quotes = sorted(quotes, key=lambda q: q.maturity)
    fitted: list[SSVISlice] = []
    for q in quotes:
        previous = fitted[-1] if fitted else None
        atm = max(float(np.interp(0.0, q.k, q.iv)) ** 2 * q.maturity, 1e-8)
        theta_floor = previous.theta if previous else 1e-10
        scale = np.sqrt(1.0 / (np.mean(q.weight) * q.k.size))
        grid: Array | None = None
        floor_w: Array | None = None
        if previous is not None:
            raw = previous.raw()
            grid = _asinh_grid(raw.m, max(raw.sigma, 1e-4))
            floor_w = previous.total_variance(grid)

        def unpack(p: Array, previous: SSVISlice | None = previous) -> tuple[SSVISlice, float]:
            theta, rho, u = float(p[0]), float(p[1]), float(p[2])
            low, high = psi_bounds(rho, theta, previous)
            infeasible = max(low - high, 0.0)
            psi = low + u * max(high - low, 0.0) if high > low else high
            return SSVISlice(theta, rho, max(psi, 1e-12)), infeasible

        def residuals(
            p: Array,
            q: Quotes = q,
            grid: Array | None = grid,
            scale: float = scale,
            atm: float = atm,
            floor_w: Array | None = floor_w,
        ) -> Array:
            slice_, infeasible = unpack(p)
            parts = [scale * _residuals(slice_, q), np.atleast_1d(1e3 * infeasible)]
            if grid is not None and floor_w is not None:
                gap = np.maximum(floor_w - slice_.total_variance(grid), 0.0)
                parts.append(np.sqrt(calendar_penalty) * gap / max(atm, 1e-8))
            return np.concatenate(parts)

        best = None
        for rho0 in ((previous.rho,) if previous else ()) + (-0.7, -0.4, -0.1):
            for u0 in (0.2, 0.6):
                start = np.array([max(atm, theta_floor * (1 + 1e-9)), rho0, u0])
                lower, upper = np.array([theta_floor, -0.999, 0.0]), np.array([np.inf, 0.999, 1.0])
                result = least_squares(
                    residuals, np.clip(start, lower, upper), bounds=(lower, upper), x_scale="jac", max_nfev=400
                )
                if best is None or result.cost < best.cost:
                    best = result
        assert best is not None
        slice_, infeasible = unpack(best.x)
        if infeasible > 0:
            raise RuntimeError(f"no feasible eSSVI slice at T = {q.maturity:.4f}")
        fitted.append(slice_)
    return fitted
