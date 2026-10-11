"""Calibrating Heston to a smile surface, and what the data can and cannot pin down.

The fit minimises the bid-ask-weighted error in implied volatility: model prices from the
contour engine of :mod:`~volsurf.stochvol.pricing`, inverted with the Day 3 inversion,
against the mid, each residual divided by the quote's bid-ask width in volatility. A
residual of one is one spread.

Five parameters do not mean five things the market reveals. Mean reversion ``kappa``,
long-run variance ``theta`` and vol-of-vol ``sigma`` trade off along a valley: a slower
pull towards a higher level produces nearly the same term structure. Three diagnostics
make that visible:

* **multistart** from a scrambled Sobol' design: the minima found, and how far apart in
  parameters fits of equal quality lie;
* the **singular values of the Jacobian** in log-parameters at the optimum: the ratio of
  the largest to the smallest is how much better one direction is determined than the
  worst, and the worst direction says which combination it is;
* a **profile** of the error in ``kappa``: re-optimise everything else with ``kappa`` held
  fixed, across a range of values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.stats import qmc

from ..market.implied import implied_vol
from .characteristic import Heston
from .pricing import otm_price

Array = NDArray[np.float64]

LOWER = np.array([1e-4, 0.05, 1e-3, 0.05, -0.99])
UPPER = np.array([0.5, 15.0, 0.5, 3.0, 0.5])
NAMES = ("v0", "kappa", "theta", "sigma", "rho")


@dataclass(frozen=True)
class Expiry:
    """The quotes of one expiry: log-moneyness, mid implied volatility and its bid-ask width."""

    maturity: float
    k: Array
    iv: Array
    width: Array


@dataclass(frozen=True)
class Calibration:
    heston: Heston
    rmse: float  # implied volatility, unweighted
    weighted: float  # root mean square residual in bid-ask widths
    residuals: list[Array]  # model minus mid implied volatility, per expiry
    evaluations: int


def model_vols(h: Heston, expiries: list[Expiry]) -> list[Array]:
    """Heston implied volatilities at every quote."""
    out = []
    for e in expiries:
        otm = otm_price(e.k, e.maturity, h)
        out.append(implied_vol(otm, 1.0, np.exp(e.k), e.maturity, 1.0, e.k >= 0))
    return out


def _pack(x: Array) -> Heston:
    return Heston(*(float(v) for v in x))


def calibrate(expiries: list[Expiry], start: Heston | None = None, fixed: dict[str, float] | None = None) -> Calibration:
    """Weighted least squares in implied volatility; ``fixed`` holds parameters at given values."""
    fixed = fixed or {}
    free = [i for i, name in enumerate(NAMES) if name not in fixed]
    base = (start or Heston(0.02, 2.0, 0.04, 0.8, -0.7)).as_array()
    for name, value in fixed.items():
        base[NAMES.index(name)] = value
    count = 0

    def full(z: Array) -> Array:
        x = base.copy()
        x[free] = z
        return np.asarray(x, dtype=np.float64)

    def residuals(z: Array) -> Array:
        nonlocal count
        count += 1
        vols = model_vols(_pack(full(z)), expiries)
        r = np.concatenate([(v - e.iv) / e.width for v, e in zip(vols, expiries, strict=True)])
        return np.asarray(np.where(np.isfinite(r), r, 50.0), dtype=np.float64)

    z0 = np.clip(base[free], LOWER[free] + 1e-9, UPPER[free] - 1e-9)
    result = least_squares(residuals, z0, bounds=(LOWER[free], UPPER[free]), x_scale="jac", diff_step=1e-6, max_nfev=200)
    h = _pack(full(result.x))
    vols = model_vols(h, expiries)
    raw = [v - e.iv for v, e in zip(vols, expiries, strict=True)]
    flat = np.concatenate(raw)
    return Calibration(h, float(np.sqrt(np.nanmean(flat**2))), float(np.sqrt(np.mean(result.fun**2))), raw, count)


def multistart(expiries: list[Expiry], starts: int = 16, seed: int = 7) -> list[Calibration]:
    """Calibrate from a scrambled Sobol' design over the parameter box; best first."""
    design = qmc.Sobol(5, scramble=True, seed=seed).random(starts)
    lo, hi = LOWER.copy(), UPPER.copy()
    hi[0], hi[2], hi[1], hi[3] = 0.1, 0.15, 8.0, 2.0  # starts in the plausible part of the box
    fits = [calibrate(expiries, _pack(lo + (hi - lo) * d)) for d in design]
    return sorted(fits, key=lambda c: c.weighted)


@dataclass(frozen=True)
class Identifiability:
    singular_values: Array  # of the weighted Jacobian in log-parameters (rho in its own units)
    weakest: Array  # the right singular vector of the smallest singular value
    condition: float


def identifiability(expiries: list[Expiry], h: Heston, step: float = 1e-4) -> Identifiability:
    """The Jacobian of the weighted residuals at ``h``, in ``(ln v0, ln kappa, ln theta, ln sigma, rho)``."""
    x = h.as_array()

    def transformed(y: Array) -> Array:
        params = np.concatenate([np.exp(y[:4]), y[4:]])
        vols = model_vols(_pack(params), expiries)
        return np.concatenate([(v - e.iv) / e.width for v, e in zip(vols, expiries, strict=True)])

    y0 = np.concatenate([np.log(x[:4]), x[4:]])
    columns = []
    for j in range(5):
        dy = np.zeros(5)
        dy[j] = step
        columns.append((transformed(y0 + dy) - transformed(y0 - dy)) / (2 * step))
    jac = np.column_stack(columns)
    _, sv, vt = np.linalg.svd(jac[np.all(np.isfinite(jac), axis=1)], full_matrices=False)
    return Identifiability(sv, vt[-1], float(sv[0] / sv[-1]))


def profile(expiries: list[Expiry], best: Heston, kappas: Array) -> list[Calibration]:
    """Re-optimise the other four parameters with ``kappa`` held at each value."""
    fits = []
    start = best
    for kappa in kappas:
        fit = calibrate(expiries, start, fixed={"kappa": float(kappa)})
        fits.append(fit)
        start = fit.heston
    return fits
