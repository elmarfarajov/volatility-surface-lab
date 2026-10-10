"""Between and beyond the fitted expiries, without creating arbitrage.

Interpolating total variance at fixed log-moneyness, as the original ``VolSurface`` does,
keeps the surface calendar-free when the slices are, but not butterfly-free: on the S&P
500 surface of 9 October 2026 its density between expiries reached ``g = -3.9``.

**Between two expiries** Gatheral and Jacquier (2014, section 5.2) interpolate *prices*.
With ``c_i(k)`` the undiscounted call over the forward at log-moneyness ``k`` and ``theta``
the ATM total variance, interpolated linearly in time,

    c(k, t) = alpha_t c_i(k) + (1 - alpha_t) c_{i+1}(k),
    alpha_t = (sqrt(theta_{i+1}) - sqrt(theta_t)) / (sqrt(theta_{i+1}) - sqrt(theta_i)).

A convex combination of call-price functions that are convex, decreasing and bounded is
another such function, so there is no butterfly arbitrage; ``alpha_t`` falls with ``t`` and
``c_{i+1} >= c_i``, so there is no calendar arbitrage either. Before the first expiry the
lower slice is expiry itself, ``(1 - e^k)^+`` with ``theta = 0``.

**Beyond the last expiry** the same device would need a slice of larger variance. Instead,
the last terminal distribution is convolved with independent log-normal noise of
variance ``theta_t - theta_n`` and mean one: the extended process is still a martingale
and its distributions grow in convex order, which is exactly freedom from static
arbitrage. Its calls are a Gauss-Hermite average of the last slice's calls.

Implied volatilities come from these prices through the inversion of Day 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..market.implied import invert_normalised
from .density import density, option_price
from .svi import RawSVI

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Surface:
    """Arbitrage-free slices at their maturities, joined by price interpolation."""

    maturities: Array
    slices: list[RawSVI]
    forwards: Array | None = None
    spot: float | None = None
    _thetas: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        thetas = np.array([float(s.total_variance(0.0)) for s in self.slices])
        if np.any(np.diff(self.maturities) <= 0) or np.any(np.diff(thetas) < 0):
            raise ValueError("maturities must increase and ATM total variance must not fall")
        object.__setattr__(self, "_thetas", thetas)

    @property
    def thetas(self) -> Array:
        return self._thetas

    def theta(self, t: float) -> float:
        t_nodes = np.concatenate([[0.0], self.maturities])
        th_nodes = np.concatenate([[0.0], self._thetas])
        if t <= self.maturities[-1]:
            return float(np.interp(t, t_nodes, th_nodes))
        return float(self._thetas[-1] * t / self.maturities[-1])

    def otm(self, k: ArrayLike, t: float) -> Array:
        """The out-of-the-money option over the forward - a put below the forward, a call above.

        Calls and puts differ by the intrinsic value, which is linear in the slice, so the
        same convex combination interpolates both. Out-of-the-money prices carry the time
        value alone, so nothing cancels when they are inverted deep in the money.
        """
        k = np.asarray(k, dtype=np.float64)
        call = k >= 0
        if t <= 0:
            return np.zeros_like(k)
        i = int(np.searchsorted(self.maturities, t))
        if i < len(self.maturities) and np.isclose(t, self.maturities[i], rtol=0, atol=1e-14):
            return option_price(self.slices[i].total_variance(k), k, call)
        if i == len(self.maturities):
            return self._extrapolate(k, t)
        upper = option_price(self.slices[i].total_variance(k), k, call)
        theta_hi, theta_lo = self._thetas[i], (self._thetas[i - 1] if i > 0 else 0.0)
        lower = option_price(self.slices[i - 1].total_variance(k), k, call) if i > 0 else np.zeros_like(k)
        if theta_hi == theta_lo:
            return upper
        alpha = (np.sqrt(theta_hi) - np.sqrt(self.theta(t))) / (np.sqrt(theta_hi) - np.sqrt(theta_lo))
        return np.asarray(alpha * lower + (1.0 - alpha) * upper, dtype=np.float64)

    def call(self, k: ArrayLike, t: float) -> Array:
        """The undiscounted call over the forward at log-moneyness ``k`` and maturity ``t``."""
        k = np.asarray(k, dtype=np.float64)
        return np.asarray(self.otm(k, t) + np.maximum(1.0 - np.exp(k), 0.0), dtype=np.float64)

    def _extrapolate(self, k: Array, t: float) -> Array:
        """``E[e^z o_n(k - z)]`` with ``z ~ N(-v/2, v)``, ``v = theta_t - theta_n``, by 60-point Gauss-Hermite.

        ``S_t = S_n e^z`` with ``z`` independent: the call (or put) on ``S_t`` at ``e^k`` is ``e^z``
        calls (puts) on ``S_n`` at ``e^{k - z}``, averaged over ``z``. Puts are used below the
        forward, calls above, so the result is the out-of-the-money price.
        """
        last = self.slices[-1]
        extra = self.theta(t) - float(self._thetas[-1])
        nodes, weights = np.polynomial.hermite_e.hermegauss(60)
        z = -0.5 * extra + np.sqrt(extra) * nodes
        weights = weights / weights.sum()
        call = k >= 0
        total = np.zeros_like(k)
        for zi, wi in zip(z, weights, strict=True):
            shifted = k - zi
            total += wi * np.exp(zi) * option_price(last.total_variance(shifted), shifted, call)
        return np.asarray(total, dtype=np.float64)

    def total_variance(self, k: ArrayLike, t: float) -> Array:
        """Total implied variance, by inverting the interpolated out-of-the-money prices."""
        k = np.asarray(k, dtype=np.float64)
        s = invert_normalised(self.otm(k, t) * np.exp(-0.5 * k), -k, k >= 0).total_vol
        return np.asarray(s**2, dtype=np.float64)

    def implied_vol(self, k: ArrayLike, t: float) -> Array:
        return np.asarray(np.sqrt(self.total_variance(k, t) / t), dtype=np.float64)

    def density(self, k: ArrayLike, t: float) -> Array:
        """The density of ``ln(S_t / F_t)``: between expiries, the same convex combination of the slice densities."""
        k = np.asarray(k, dtype=np.float64)
        i = int(np.searchsorted(self.maturities, t))
        if i == len(self.maturities) or i == 0 and t < self.maturities[0]:
            h = 1e-3
            c = [self.call(k + d, t) for d in (-h, 0.0, h)]
            return np.asarray(np.exp(-k) * ((c[2] - 2 * c[1] + c[0]) / h**2 - (c[2] - c[0]) / (2 * h)), dtype=np.float64)
        if np.isclose(t, self.maturities[i], rtol=0, atol=1e-14):
            return density(self.slices[i], k)
        theta_hi, theta_lo = self._thetas[i], self._thetas[i - 1]
        alpha = (np.sqrt(theta_hi) - np.sqrt(self.theta(t))) / (np.sqrt(theta_hi) - np.sqrt(theta_lo))
        return np.asarray(alpha * density(self.slices[i - 1], k) + (1.0 - alpha) * density(self.slices[i], k), dtype=np.float64)
