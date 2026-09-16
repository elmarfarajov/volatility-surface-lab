"""Arbitrage-aware implied volatility surface and Dupire local volatility.

Slices are joined in total variance along lines of constant log forward-moneyness
using monotone (PCHIP) interpolation in maturity, anchored at w(k, 0) = 0. Because
PCHIP preserves monotonicity, a calendar-arbitrage-free set of slices yields a
calendar-arbitrage-free surface, and its first maturity derivative is continuous,
which keeps the local-volatility surface free of the saw-tooth artefacts produced by
piecewise-linear interpolation.

Local variance follows Gatheral's form of Dupire's equation in total-variance terms:

    sigma_loc^2(k, T) = (dw/dT) / g(k, T),
    g = (1 - k w_k / (2w))^2 - (w_k^2 / 4)(1/w + 1/4) + w_kk / 2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from .svi import SVISlice


@dataclass
class VolSurface:
    slices: list[SVISlice]
    forwards: np.ndarray
    discounts: np.ndarray
    spot: float
    _maturities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        order = np.argsort([s.maturity for s in self.slices])
        self.slices = [self.slices[i] for i in order]
        self.forwards = np.asarray(self.forwards, dtype=float)[order]
        self.discounts = np.asarray(self.discounts, dtype=float)[order]
        self._maturities = np.array([s.maturity for s in self.slices])

    @property
    def maturities(self) -> np.ndarray:
        return self._maturities

    def forward(self, maturity: ArrayLike) -> np.ndarray:
        """Log-linear interpolation of forwards (constant carry between expiries)."""
        t_nodes = np.concatenate([[0.0], self._maturities])
        log_f = np.concatenate([[np.log(self.spot)], np.log(self.forwards)])
        t = np.asarray(maturity, dtype=float)
        slope = (log_f[-1] - log_f[-2]) / (t_nodes[-1] - t_nodes[-2])
        return np.exp(np.where(t <= t_nodes[-1], np.interp(t, t_nodes, log_f), log_f[-1] + slope * (t - t_nodes[-1])))

    def _node_variances(self, k: np.ndarray) -> np.ndarray:
        nodes = np.array([s.total_variance(k) for s in self.slices])
        return np.maximum.accumulate(np.maximum(nodes, 0.0), axis=0)

    def total_variance(self, k: ArrayLike, maturity: ArrayLike) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        t = np.asarray(maturity, dtype=float)
        k_b, t_b = np.broadcast_arrays(k, t)
        flat_k = k_b.ravel()
        flat_t = t_b.ravel()
        out = np.empty_like(flat_k)
        for t_value in np.unique(flat_t):
            sel = flat_t == t_value
            out[sel] = self._total_variance_at(flat_k[sel], float(t_value))[0]
        return out.reshape(k_b.shape)

    def _total_variance_at(self, k: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        nodes = self._node_variances(k)
        t_last = self._maturities[-1]
        if t > t_last:
            slope = nodes[-1] / t_last
            return nodes[-1] + slope * (t - t_last), slope
        t_nodes = np.concatenate([[0.0], self._maturities])
        values = np.vstack([np.zeros_like(k), nodes])
        interp = PchipInterpolator(t_nodes, values, axis=0, extrapolate=False)
        return interp(t), interp.derivative()(t)

    def implied_vol(self, k: ArrayLike, maturity: ArrayLike) -> np.ndarray:
        t = np.asarray(maturity, dtype=float)
        return np.sqrt(np.maximum(self.total_variance(k, t), 0.0) / t)

    def implied_vol_strike(self, strike: ArrayLike, maturity: float) -> np.ndarray:
        return self.implied_vol(np.log(np.asarray(strike, dtype=float) / self.forward(maturity)), maturity)

    def local_vol(self, k: ArrayLike, maturity: float, dk: float = 1e-3) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        w, w_t = self._total_variance_at(k, maturity)
        w_up, _ = self._total_variance_at(k + dk, maturity)
        w_dn, _ = self._total_variance_at(k - dk, maturity)
        w_k = (w_up - w_dn) / (2.0 * dk)
        w_kk = (w_up - 2.0 * w + w_dn) / (dk * dk)
        w_safe = np.maximum(w, 1e-12)
        g = (1.0 - k * w_k / (2.0 * w_safe)) ** 2 - 0.25 * w_k * w_k * (1.0 / w_safe + 0.25) + 0.5 * w_kk
        with np.errstate(divide="ignore", invalid="ignore"):
            local_var = np.where(g > 1e-8, w_t / g, np.nan)
        return np.sqrt(np.where(local_var > 0, local_var, np.nan))

    def calendar_violations(self, k: ArrayLike) -> np.ndarray:
        """Largest decrease in raw slice total variance between consecutive expiries."""
        k = np.asarray(k, dtype=float)
        raw = np.array([s.total_variance(k) for s in self.slices])
        return np.maximum(raw[:-1] - raw[1:], 0.0).max(axis=1) if len(self.slices) > 1 else np.array([])
