"""Stochastic Volatility Inspired (SVI) smiles with static-arbitrage control.

Raw SVI (Gatheral, 2004) parameterises total implied variance w = sigma_BS^2 * T as a
function of log forward-moneyness k = ln(K/F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Fitting strategy:

1. Quasi-explicit initialisation (Zeliade Systems, 2009). For fixed (m, s) the model is
   linear in (a, b*s, rho*b*s), so a coarse (m, s) grid is scanned with ordinary least
   squares and the best candidates seed the non-linear step.
2. Bounded least squares in implied-volatility units with penalty residuals that
   enforce, on a dense strike grid:
     * non-negative total variance,
     * Roger Lee's (2004) moment bound on the wings, b * (1 + |rho|) <= 2,
     * Durrleman's butterfly condition g(k) >= 0 (a non-negative risk-neutral density),
     * no calendar arbitrage against the previous (shorter) expiry, w_T2(k) >= w_T1(k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVISlice:
    a: float
    b: float
    rho: float
    m: float
    s: float
    maturity: float

    def total_variance(self, k: ArrayLike) -> np.ndarray:
        dk = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * dk + np.sqrt(dk * dk + self.s * self.s))

    def dw_dk(self, k: ArrayLike) -> np.ndarray:
        dk = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + dk / np.sqrt(dk * dk + self.s * self.s))

    def d2w_dk2(self, k: ArrayLike) -> np.ndarray:
        dk = np.asarray(k, dtype=float) - self.m
        return self.b * self.s * self.s / (dk * dk + self.s * self.s) ** 1.5

    def implied_vol(self, k: ArrayLike) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / self.maturity)

    def durrleman_g(self, k: ArrayLike) -> np.ndarray:
        """g(k) >= 0 for all k  <=>  the slice admits no butterfly arbitrage."""
        k = np.asarray(k, dtype=float)
        w = np.maximum(self.total_variance(k), 1e-12)
        w1 = self.dw_dk(k)
        w2 = self.d2w_dk2(k)
        with np.errstate(over="ignore", invalid="ignore"):
            g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
        return np.where(np.isfinite(g), g, -1e6)

    def risk_neutral_density(self, k: ArrayLike) -> np.ndarray:
        """Density of ln(S_T/F) implied by the smile (Gatheral & Jacquier, 2014)."""
        k = np.asarray(k, dtype=float)
        w = np.maximum(self.total_variance(k), 1e-12)
        d_minus = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        return self.durrleman_g(k) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus * d_minus)

    @property
    def min_variance(self) -> float:
        return self.a + self.b * self.s * np.sqrt(1.0 - self.rho**2)

    @property
    def lee_wing_slope(self) -> float:
        return self.b * (1.0 + abs(self.rho))

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "s": self.s, "T": self.maturity}


@dataclass(frozen=True)
class SVIFit:
    slice: SVISlice
    rmse_vol: float
    max_abs_error_vol: float
    min_durrleman_g: float
    calendar_violation: float
    n_quotes: int


def _quasi_explicit_candidates(k: np.ndarray, w: np.ndarray, weights: np.ndarray, maturity: float, n_best: int = 4):
    k_span = max(k.max() - k.min(), 0.05)
    m_grid = np.linspace(k.min() - 0.1 * k_span, k.max() + 0.1 * k_span, 9)
    s_grid = np.geomspace(0.01, 1.0, 9) * max(np.sqrt(maturity), 0.1)
    scored: list[tuple[float, np.ndarray]] = []
    sw = np.sqrt(weights)
    for m in m_grid:
        for s in s_grid:
            y = (k - m) / s
            design = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
            coef, *_ = np.linalg.lstsq(design * sw[:, None], w * sw, rcond=None)
            a, d, c = coef
            c = float(np.clip(c, 1e-6, 2.0 * s))
            d = float(np.clip(d, -0.999 * c, 0.999 * c))
            b = c / s
            rho = d / c
            b = min(b, 1.999 / (1.0 + abs(rho)))
            params = np.array([a, b, rho, m, s])
            fitted = SVISlice(*params, maturity=maturity).total_variance(k)
            scored.append((float(np.sum(weights * (fitted - w) ** 2)), params))
    scored.sort(key=lambda item: item[0])
    return [p for _, p in scored[:n_best]]


def fit_svi_slice(
    log_moneyness: ArrayLike,
    implied_vols: ArrayLike,
    maturity: float,
    weights: ArrayLike | None = None,
    previous: SVISlice | None = None,
    penalty: float = 1e3,
) -> SVIFit:
    k = np.asarray(log_moneyness, dtype=float)
    iv = np.asarray(implied_vols, dtype=float)
    mask = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[mask], iv[mask]
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)[mask]
    wts = wts / wts.mean()
    if k.size < 5:
        raise ValueError("At least five quotes are needed to fit an SVI slice")

    w_market = iv * iv * maturity
    k_span = k.max() - k.min()
    k_grid = np.linspace(k.min() - 0.5 * k_span - 0.1, k.max() + 0.5 * k_span + 0.1, 81)
    sqrt_pen = np.sqrt(penalty)
    sqrt_w = np.sqrt(wts)

    def residuals(p: np.ndarray) -> np.ndarray:
        sl = SVISlice(*p, maturity=maturity)
        w_model = np.maximum(sl.total_variance(k), 1e-12)
        fit = sqrt_w * (np.sqrt(w_model / maturity) - iv)
        w_grid = sl.total_variance(k_grid)
        pen = [
            sqrt_pen * np.maximum(-w_grid, 0.0),
            sqrt_pen * np.atleast_1d(max(sl.lee_wing_slope - 2.0, 0.0)),
            sqrt_pen * np.maximum(-sl.durrleman_g(k_grid), 0.0) * 0.1,
        ]
        if previous is not None:
            pen.append(sqrt_pen * np.maximum(previous.total_variance(k_grid) - w_grid, 0.0))
        return np.concatenate([fit, *pen])

    w_max = float(w_market.max())
    # Curvature on a scale finer than the strike grid is not identified by the quotes and
    # produces spurious spikes in the implied density, so s is bounded by the grid spacing.
    s_floor = max(float(np.median(np.diff(np.sort(k)))), 1e-3)
    lower = [-w_max, 0.0, -0.999, k.min() - 1.0, s_floor]
    upper = [w_max, 2.0, 0.999, k.max() + 1.0, 2.0]

    best = None
    for start in _quasi_explicit_candidates(k, w_market, wts, maturity):
        start = np.clip(start, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
        result = least_squares(residuals, start, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=2000)
        if best is None or result.cost < best.cost:
            best = result

    sl = SVISlice(*best.x, maturity=maturity)
    errors = sl.implied_vol(k) - iv
    calendar = (
        0.0 if previous is None else float(np.max(np.maximum(previous.total_variance(k_grid) - sl.total_variance(k_grid), 0.0)))
    )
    return SVIFit(
        slice=sl,
        rmse_vol=float(np.sqrt(np.mean(errors**2))),
        max_abs_error_vol=float(np.max(np.abs(errors))),
        min_durrleman_g=float(np.min(sl.durrleman_g(k_grid))),
        calendar_violation=calendar,
        n_quotes=int(k.size),
    )
