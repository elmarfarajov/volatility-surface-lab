"""The risk-neutral density a smile implies, and the checks every density must pass.

For a slice with total variance ``w(k)``, the density of the terminal log-moneyness
``y = ln(S_T / F)`` is (Gatheral and Jacquier, 2014, equation 2.2)

    p(k) = g(k) / sqrt(2 pi w(k)) * exp(-d_-(k)^2 / 2),   d_- = -k / sqrt(w) - sqrt(w) / 2,

with ``g`` Durrleman's function. A density built from prices that admit no arbitrage
integrates to one, and its exponential integrates to one as well - the forward is the
mean of ``S_T``. Both are checked here by quadrature on a grid that follows the smile, and
the density is compared with Breeden and Litzenberger's second difference of call prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..market.black import normalised_black
from .svi import RawSVI

Array = NDArray[np.float64]


def density(slice_: RawSVI, k: Array) -> Array:
    """The density of ``ln(S_T / F)`` at ``k``."""
    w = slice_.total_variance(k)
    root = np.sqrt(np.maximum(w, 1e-300))
    d_minus = -k / root - 0.5 * root
    return np.asarray(slice_.g(k) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus**2), dtype=np.float64)


def option_price(total_variance: Array, k: Array, is_call: ArrayLike = True) -> Array:
    """The undiscounted call ``E[(e^y - e^k)^+]`` or put ``E[(e^k - e^y)^+]`` over the forward, accurate in the far wings."""
    s = np.sqrt(np.maximum(total_variance, 0.0))
    return np.asarray(np.exp(0.5 * k) * normalised_black(-k, s, is_call), dtype=np.float64)


def call_price(total_variance: Array, k: Array) -> Array:
    """The undiscounted call over the forward, ``E[(e^y - e^k)^+]``."""
    return option_price(total_variance, k, True)


def support_grid(slice_: RawSVI, points: int = 20001, spread: float = 12.0) -> Array:
    """A uniform grid in ``k`` out to where ``|k| = spread * sqrt(w(k))`` on each side.

    The ends follow the wings: where the smile steepens, the distribution's tail is
    heavier and the grid reaches further.
    """
    ends = []
    for side in (-1.0, 1.0):
        k = side * spread * float(np.sqrt(max(slice_.total_variance(0.0).item(), 1e-8)))
        for _ in range(30):
            k = side * spread * float(np.sqrt(max(slice_.total_variance(k).item(), 1e-8)))
        ends.append(float(np.clip(k, -60.0, 60.0)))
    return np.asarray(np.linspace(ends[0], ends[1], points), dtype=np.float64)


@dataclass(frozen=True)
class Moments:
    mass: float  # integral of p: 1
    forward: float  # integral of e^y p: 1, the martingale condition
    breeden_litzenberger: float  # largest gap to the density from call prices, relative to the peak


def moments(slice_: RawSVI) -> Moments:
    k = support_grid(slice_)
    p = density(slice_, k)
    mass = float(np.trapezoid(p, k))
    forward = float(np.trapezoid(np.exp(k) * p, k))
    # Breeden-Litzenberger in log-strike: p(k) = e^{-k} (c'' - c'), by central differences on a uniform grid
    root = float(np.sqrt(slice_.total_variance(0.0).item()))
    u = np.linspace(-6.0 * root, 6.0 * root, 4001)
    h = u[1] - u[0]
    c = call_price(slice_.total_variance(u), u)
    numeric = np.exp(-u[1:-1]) * ((c[2:] - 2.0 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2.0 * h))
    exact = density(slice_, u[1:-1])
    return Moments(mass, forward, float(np.max(np.abs(numeric - exact)) / np.max(exact)))
