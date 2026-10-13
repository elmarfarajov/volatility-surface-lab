"""SABR: Hagan's expansion, its corrections, an exact reference, and where the expansion fails.

The SABR model (Hagan, Kumar, Lesniewski and Woodward, 2002) moves the forward and its
volatility together:

    dF = alpha F^beta dW,   d alpha = nu alpha dZ,   dW dZ = rho dt.

Its appeal is a closed-form implied volatility from a singular-perturbation expansion,
accurate when ``nu^2 T`` is small. Three facts about that formula matter in practice:

* **Obloj's (2008) correction** replaces the leading term so that the formula is right as
  ``beta -> 1`` and far from the money; both versions are here.
* **The expansion is not a model.** Far in the money, at low strikes and for long
  maturities, the implied volatilities it returns are not those of any distribution: the
  density they imply turns negative. :func:`density_from_vols` shows where.
* **An exact reference** is available for ``beta = 1``: given the path of ``alpha``, the
  log-forward is normal (Willard's conditioning), so the price is a Black price averaged
  over volatility paths, which only the integral of ``alpha^2`` needs to be simulated for.
  Its error is Monte Carlo noise plus the trapezoid rule on that integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from ..market.black import black_price
from ..market.implied import implied_vol

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SABR:
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or not 0 <= self.beta <= 1 or not -1 < self.rho < 1 or self.nu < 0:
            raise ValueError(f"not a SABR parameter set: {self}")


def hagan_vol(f: float, strikes: ArrayLike, t: float, p: SABR, obloj: bool = False) -> Array:
    """Hagan's lognormal implied volatility (2002, eq. A.69), or Obloj's (2008) correction of its leading term."""
    k = np.asarray(strikes, dtype=np.float64)
    a, b, r, n = p.alpha, p.beta, p.rho, p.nu
    one = 1.0 - b
    log_fk = np.log(f / k)
    fk_mid = (f * k) ** (0.5 * one)
    # the time correction, common to both forms
    correction = 1.0 + t * (
        one**2 * a * a / (24.0 * fk_mid**2) + 0.25 * r * b * n * a / fk_mid + (2.0 - 3.0 * r * r) * n * n / 24.0
    )

    def chi(z: Array) -> Array:
        return np.asarray(np.log((np.sqrt(1.0 - 2.0 * r * z + z * z) + z - r) / (1.0 - r)), dtype=np.float64)

    small = np.abs(log_fk) < 1e-7
    if obloj:
        # Obloj: the leading term nu ln(F/K) / chi(z), with z = nu (F^(1-b) - K^(1-b)) / (alpha (1-b))
        # (nu ln(F/K) / alpha at b = 1)
        z = n * (f**one - k**one) / (a * one) if one > 0 else n * log_fk / a
        with np.errstate(divide="ignore", invalid="ignore"):
            lead = np.where(small, a / f**one, n * log_fk / chi(np.where(small, 1.0, z)))
        return np.asarray(lead * correction, dtype=np.float64)
    z = n / a * fk_mid * log_fk
    denominator = fk_mid * (1.0 + one**2 / 24.0 * log_fk**2 + one**4 / 1920.0 * log_fk**4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(z) < 1e-7, 1.0 - 0.5 * r * z, z / chi(np.where(np.abs(z) < 1e-7, 1.0, z)))
    return np.asarray(a / denominator * ratio * correction, dtype=np.float64)


def density_from_vols(f: float, strikes: Array, t: float, vols: Array) -> Array:
    """The density of ``F_T`` implied by a smile through Breeden and Litzenberger: ``d^2 C / dK^2`` by central differences."""
    calls = black_price(f, strikes, t, vols, 1.0, True)
    h = np.diff(strikes)
    second = (
        2.0 * (calls[2:] * h[:-1] - calls[1:-1] * (h[:-1] + h[1:]) + calls[:-2] * h[1:]) / (h[:-1] * h[1:] * (h[:-1] + h[1:]))
    )
    return np.asarray(second, dtype=np.float64)


@dataclass(frozen=True)
class Reference:
    vols: Array
    stderr: Array  # of the implied volatility, by the delta method


def reference_vols(
    f: float, strikes: ArrayLike, t: float, p: SABR, paths: int = 200_000, steps: int = 400, seed: int = 1
) -> Reference:
    """Exact-in-law reference for ``beta = 1``: Black prices averaged over simulated volatility paths.

    ``ln F_T = ln F + rho/nu (alpha_T - alpha) - rho^2/2 V + sqrt((1 - rho^2) V) N`` with ``V = int alpha^2 dt``,
    so conditionally on the volatility path the call is a Black call on the adjusted forward
    ``F exp(rho/nu (alpha_T - alpha) - rho^2 V/2)`` with total variance ``(1 - rho^2) V``.
    Antithetic volatility paths halve the noise.
    """
    if p.beta != 1.0:
        raise ValueError("the conditional reference is exact for beta = 1 only")
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    rng = np.random.default_rng(seed)
    dt = t / steps
    half = paths // 2
    z = rng.standard_normal((steps, half))
    z = np.concatenate([z, -z], axis=1)
    log_alpha = np.log(p.alpha) + np.cumsum(p.nu * np.sqrt(dt) * z - 0.5 * p.nu**2 * dt, axis=0)
    alpha_path = np.vstack([np.full((1, z.shape[1]), p.alpha), np.exp(log_alpha)])
    integral = dt * (0.5 * alpha_path[0] ** 2 + np.sum(alpha_path[1:-1] ** 2, axis=0) + 0.5 * alpha_path[-1] ** 2)
    adjusted = (
        f * np.exp(p.rho / p.nu * (alpha_path[-1] - p.alpha) - 0.5 * p.rho**2 * integral) if p.nu > 0 else np.full(z.shape[1], f)
    )
    total = np.sqrt((1.0 - p.rho**2) * integral)
    prices = black_price(adjusted[None, :], k[:, None], 1.0, total[None, :], 1.0, (k >= f)[:, None])  # out of the money
    mean = prices.mean(axis=1)
    err = prices.std(axis=1, ddof=1) / np.sqrt(prices.shape[1])
    vols = implied_vol(mean, f, k, t, 1.0, k >= f)
    vega = black_price(f, k, t, vols + 1e-6, 1.0, k >= f) - black_price(f, k, t, vols - 1e-6, 1.0, k >= f)
    return Reference(vols, np.asarray(err / (vega / 2e-6), dtype=np.float64))


@dataclass(frozen=True)
class SABRFit:
    params: SABR
    rmse: float  # implied volatility


def calibrate(
    f: float, strikes: Array, t: float, vols: Array, weights: Array | None = None, beta: float = 1.0, obloj: bool = True
) -> SABRFit:
    """Fit ``alpha, rho, nu`` at a fixed ``beta`` to one expiry's implied volatilities."""
    w = np.ones_like(vols) if weights is None else np.sqrt(weights / np.mean(weights))
    atm = float(np.interp(f, strikes, vols))

    def residuals(x: Array) -> Array:
        p = SABR(float(x[0]), beta, float(x[1]), float(x[2]))
        return np.asarray(w * (hagan_vol(f, strikes, t, p, obloj) - vols), dtype=np.float64)

    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for nu0 in (0.5, 1.5, 4.0):
            start = np.array([atm * f ** (1.0 - beta), rho0, nu0])
            result = least_squares(residuals, start, bounds=([1e-6, -0.999, 1e-4], [10.0, 0.999, 20.0]), x_scale="jac")
            if best is None or result.cost < best.cost:
                best = result
    assert best is not None
    p = SABR(float(best.x[0]), beta, float(best.x[1]), float(best.x[2]))
    return SABRFit(p, float(np.sqrt(np.mean((hagan_vol(f, strikes, t, p, obloj) - vols) ** 2))))
