"""The Heston characteristic function, its moment strip, and the cumulants that size every Fourier method.

Under Heston (1993),

    dS/S = (r - q) dt + sqrt(v) dW,   dv = kappa (theta - v) dt + sigma sqrt(v) dZ,   dW dZ = rho dt,

the log forward-moneyness ``X = ln(S_T / F_T)`` has the characteristic function
``phi(u) = E[e^{iuX}] = exp(A(u) + B(u) v0)``. Heston's own formula takes a complex
logarithm that jumps branches for long maturities; the "little trap" form of Albrecher,
Mayer, Schoutens and Tistaert (2007) used here is continuous along the real line and,
as Lord and Kahl (2010) show, along the shifted contours ``u - i a`` that the pricing
integrals need.

**The moment strip.** ``E[S_T^p]`` is finite only for ``p`` in an interval around [0, 1]
(Andersen and Piterbarg, 2007). Beyond it ``phi(-ip)`` explodes. The pricing contour
must stay inside, and its edges are found here by bisection on where ``phi(-ip)`` stops
being a finite positive number. The edges are also what fixes the slopes of the implied
volatility wings, through Lee's moment formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Complex = NDArray[np.complex128]


@dataclass(frozen=True)
class Heston:
    """Heston parameters: initial variance, mean reversion, long-run variance, vol of vol, correlation."""

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0 or self.kappa <= 0 or self.theta <= 0 or self.sigma <= 0 or not -1 < self.rho < 1:
            raise ValueError(f"not a Heston parameter set: {self}")

    @property
    def feller(self) -> float:
        """``2 kappa theta / sigma^2``: at least one, and the variance never reaches zero."""
        return 2.0 * self.kappa * self.theta / self.sigma**2

    def as_array(self) -> Array:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])


def exponents(u: ArrayLike, t: float, h: Heston) -> tuple[Complex, Complex]:
    """``(A(u), B(u))`` with ``phi(u) = exp(A + B v0)``, the little-trap form; ``u`` may be complex."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    beta = h.kappa - h.rho * h.sigma * iu
    d = np.sqrt(beta * beta + h.sigma**2 * (iu + u * u))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * t)
    a = h.kappa * h.theta / h.sigma**2 * ((beta - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    b = (beta - d) / h.sigma**2 * (1.0 - e) / (1.0 - g * e)
    return np.asarray(a, dtype=np.complex128), np.asarray(b, dtype=np.complex128)


def cf(u: ArrayLike, t: float, h: Heston) -> Complex:
    """``E[exp(i u ln(S_T / F_T))]``."""
    a, b = exponents(u, t, h)
    return np.asarray(np.exp(a + b * h.v0), dtype=np.complex128)


def cumulants(t: float, h: Heston) -> tuple[float, float]:
    """Mean and variance of ``ln(S_T / F_T)``.

    The mean is exact, ``-E[int_0^T v dt] / 2``. The variance is the second derivative of
    ``ln phi`` at zero by a five-point stencil (error of order ``h^4``): it only sizes the
    integration ranges, and a closed form recalled from the literature for it was tested
    here and found wrong, so the derivative of the function itself is used.
    """
    e = np.exp(-h.kappa * t)
    mean_iv = h.theta * t + (h.v0 - h.theta) * (1.0 - e) / h.kappa
    step = 1e-3
    log_phi = np.log(cf(np.array([-2 * step, -step, 0.0, step, 2 * step]), t, h))
    second = (-log_phi[0] + 16 * log_phi[1] - 30 * log_phi[2] + 16 * log_phi[3] - log_phi[4]) / (12 * step**2)
    return float(-0.5 * mean_iv), float(-second.real)


def explosion_time(p: float, h: Heston) -> float:
    """The time at which ``E[S_T^p]`` becomes infinite (Andersen and Piterbarg, 2007).

    ``E[S_T^p] = exp(phi(T) + psi(T) v0)`` with ``psi' = sigma^2 psi^2 / 2 - k psi + p(p - 1)/2``,
    ``psi(0) = 0``, ``k = kappa - rho sigma p``. With ``D = k^2 - sigma^2 p (p - 1)``, the
    Riccati solution blows up at

    * never, if ``D >= 0`` and ``k >= 0``;
    * ``ln((|k| + sqrt D) / (|k| - sqrt D)) / sqrt D``, if ``D >= 0`` and ``k < 0``;
    * ``2 (pi/2 + arctan(k / gamma)) / gamma`` with ``gamma = sqrt(-D)``, if ``D < 0`` - which
      tends to infinity as ``D`` rises to zero with ``k > 0``, as it must.

    Derived here from the Riccati equation; a form recalled from the literature for the
    last case did not have that limit and was rejected.
    """
    k = h.kappa - h.rho * h.sigma * p
    d = k * k - h.sigma**2 * p * (p - 1.0)
    if d >= 0:
        if k >= 0:
            return float("inf")
        root = np.sqrt(d)
        return float(np.log((-k + root) / (-k - root)) / root) if root < -k else 0.0
    gamma = np.sqrt(-d)
    return float(2.0 / gamma * (0.5 * np.pi + np.arctan(k / gamma)))


def moment_strip(t: float, h: Heston, limit: float = 200.0) -> tuple[float, float]:
    """The interval of ``p`` for which ``E[(S_T/F_T)^p]`` is finite: ``(p_minus, p_plus)`` with ``p_minus < 0 < 1 < p_plus``.

    Each edge solves ``explosion_time(p) = t`` by bisection: the explosion time falls
    monotonically as ``p`` moves away from [0, 1]. (Testing where the characteristic
    function stops being finite and real does not work: beyond the explosion the
    little-trap formula's continuation turns real and positive again.)
    """
    edges = []
    for inside, outward in ((1.0, 1.0), (0.0, -1.0)):
        far = inside + outward * limit
        if explosion_time(far, h) > t:
            edges.append(float(far))
            continue
        lo, hi = inside, far
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if explosion_time(mid, h) > t else (lo, mid)
        edges.append(float(lo))
    return edges[1], edges[0]
