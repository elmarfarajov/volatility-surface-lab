"""An option's value and hedge ratios along simulated paths, under Heston and under local volatility.

Hedging a book path by path needs the option's value and Greeks at every rebalancing
date on every path: millions of evaluations. Both models are therefore solved once on a
grid and read off by interpolation.

**Heston.** The undiscounted call over the forward is a function ``c(k, v; tau)`` of
log-moneyness ``k = ln(K/F)``, instantaneous variance ``v`` and time to expiry. It is
priced on a grid in ``(v, k)`` at each rebalancing date by the Day 5 contour integral,
and a bicubic spline gives the price and its derivatives. With ``F = S e^{b tau}`` and
discount ``D``:

    C = D F c,    dC/dS = D e^{b tau} (c - c_k),    dC/dv = D F c_v.

**Local volatility.** In ``X = ln(S/F_t)``, ``F_t`` the forward to ``t``, the undiscounted
option value ``u(X, t)`` solves

    u_t + (sigma^2/2)(u_XX - u_X) = 0,    u(X, T) = payoff(F_T e^X),

with ``sigma^2 = sigma_loc^2(X, t)`` from Dupire (Day 6). It is solved backwards once by
Crank-Nicolson after Rannacher steps; ``C = D u`` and ``dC/dS = D u_X / S``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.linalg import solve_banded

from ..localvol.dupire import local_variance
from ..smile.interpolation import Surface
from ..stochvol.characteristic import Heston, cf, cumulants

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Valuation:
    value: Array
    delta: Array  # dV/dS
    vega_v: Array  # dV/dv, instantaneous variance (Heston); zero where the model has none


# ---------------------------------------------------------------- Heston


@dataclass(frozen=True)
class HestonGrid:
    """A European call of strike ``strike`` (in units of today's spot) priced on a (time, variance, log-moneyness) grid."""

    model: Heston
    strike: float
    expiry: float
    r: float
    b: float
    times: Array
    splines: tuple[RectBivariateSpline, ...]
    v_range: tuple[float, float]
    k_range: tuple[float, float]

    @classmethod
    def build(
        cls,
        model: Heston,
        strike: float,
        expiry: float,
        r: float,
        b: float,
        times: Array,
        v_max: float = 0.5,
        v_points: int = 36,
        k_range: tuple[float, float] = (-1.0, 0.6),
        k_points: int = 321,
    ) -> HestonGrid:
        v = np.linspace(0.0, np.sqrt(v_max), v_points) ** 2  # denser near zero, where the price bends most
        v[0] = 1e-7
        k = np.linspace(*k_range, k_points)
        splines = []
        for t in times:
            tau = expiry - float(t)
            if tau <= 1e-10:
                # at expiry the price is the payoff, and in k it is the hockey stick
                rows = np.tile(np.maximum(1.0 - np.exp(k), 0.0), (v.size, 1))
            else:
                rows = np.vstack([cos_calls(k, tau, _with_v0(model, x)) for x in v])
            splines.append(RectBivariateSpline(v, k, rows, kx=3, ky=3))
        return cls(
            model, strike, expiry, r, b, np.asarray(times, dtype=np.float64), tuple(splines), (float(v[0]), float(v[-1])), k_range
        )

    def at(self, i: int, spot: Array, variance: Array) -> Valuation:
        """Value, delta and dV/dv at grid time ``times[i]`` for spots and variances (one per path)."""
        tau = self.expiry - float(self.times[i])
        forward = spot * np.exp(self.b * tau)
        discount = np.exp(-self.r * tau)
        k = np.clip(np.log(self.strike / forward), *self.k_range)
        v = np.clip(variance, *self.v_range)
        spline = self.splines[i]
        c = spline.ev(v, k)
        c_k = spline.ev(v, k, dy=1)
        c_v = spline.ev(v, k, dx=1)
        value = discount * forward * c
        delta = discount * np.exp(self.b * tau) * (c - c_k)
        return Valuation(np.asarray(value), np.asarray(delta), np.asarray(discount * forward * c_v))


def cos_calls(k: Array, tau: float, h: Heston, terms: int = 512, width: float = 12.0) -> Array:
    """Undiscounted calls over the forward at log-moneyness ``k`` by Fang and Oosterlee's COS method, all strikes at once.

    Within 1e-6 of the Day 5 contour integral down to ``v0 = 0`` and a day to expiry; the grid
    needs thousands of smiles, and this is the fast one.
    """
    c1, c2 = cumulants(tau, h)
    a, b = c1 - width * np.sqrt(c2), c1 + width * np.sqrt(c2)
    n = np.arange(terms)
    u = n * np.pi / (b - a)
    phi = cf(u, tau, h) * np.exp(-1j * u * a)
    top = np.clip(k, a, b)[:, None]
    span = u[None, :] * (top - a)
    chi = (np.exp(top) * (np.cos(span) + u * np.sin(span)) - np.exp(a)) / (1.0 + u * u)
    psi = np.where(n == 0, top - a, np.sin(span) / np.where(n == 0, 1.0, u))
    coefficients = 2.0 / (b - a) * (np.exp(k)[:, None] * psi - chi)
    coefficients[:, 0] *= 0.5
    put = (coefficients @ phi).real
    return np.asarray(put + 1.0 - np.exp(k), dtype=np.float64)


def _with_v0(h: Heston, v0: float) -> Heston:
    return Heston(float(v0), h.kappa, h.theta, h.sigma, h.rho)


def minimum_variance_delta(h: Heston, spot: Array, delta: Array, vega_v: Array) -> Array:
    """``dC/dS + (rho sigma / S) dC/dv``: the delta that also hedges the part of a variance shock correlated with spot."""
    return np.asarray(delta + h.rho * h.sigma * vega_v / spot, dtype=np.float64)


# ---------------------------------------------------------------- local volatility


@dataclass(frozen=True)
class LocalVolGrid:
    """A European option under Dupire local volatility, solved backwards once on ``X = ln(S/F_t)``."""

    strike: float
    expiry: float
    r: float
    b: float
    x: Array
    times: Array  # the solver's time nodes, increasing
    values: Array  # undiscounted value, [time, x]

    @classmethod
    def build(
        cls,
        surface: Surface,
        strike: float,
        expiry: float,
        r: float,
        b: float,
        call: bool = True,
        space_steps: int = 600,
        steps_per_year: int = 1000,
        width: float = 1.6,
        rannacher: int = 4,
    ) -> LocalVolGrid:
        """Spot and strike in units of today's spot; the surface's log-moneyness is relative to the forward to each date."""
        x = np.linspace(-width, width, space_steps + 1)
        h = x[1] - x[0]
        n = max(int(np.ceil(expiry * steps_per_year)), 20)
        times = np.linspace(0.0, expiry, n + 1)
        forward_t = np.exp(b * expiry)
        u = np.maximum(forward_t * np.exp(x) - strike, 0.0) if call else np.maximum(strike - forward_t * np.exp(x), 0.0)
        out = np.empty((n + 1, x.size))
        out[n] = u
        for j in range(n, 0, -1):
            dt = times[j] - times[j - 1]
            t_mid = 0.5 * (times[j] + times[j - 1])
            var = local_variance(surface, x, max(t_mid, 1e-6))
            var = np.where(np.isfinite(var), np.clip(var, 1e-6, 25.0), 1e-6)
            # operator L u = (var/2)(u_xx - u_x), central differences
            lower = 0.5 * var * (1.0 / h**2 + 0.5 / h)
            upper = 0.5 * var * (1.0 / h**2 - 0.5 / h)
            diag = -var / h**2
            # Rannacher: the first steps from the kinked payoff as two implicit half-steps each, then Crank-Nicolson
            theta, sub = (1.0, 2) if (n - j) < rannacher else (0.5, 1)
            step = dt / sub
            band = np.zeros((3, x.size))
            band[0, 2:] = -theta * step * upper[1:-1]
            band[1, 1:-1] = 1.0 - theta * step * diag[1:-1]
            band[2, :-2] = -theta * step * lower[1:-1]
            band[1, 0] = band[1, -1] = 1.0
            for _ in range(sub):
                rhs = u.copy()  # far out the value is its intrinsic value on the forward, which does not change in X
                if theta < 1.0:
                    rhs[1:-1] += (1 - theta) * step * (lower[1:-1] * u[:-2] + diag[1:-1] * u[1:-1] + upper[1:-1] * u[2:])
                u = solve_banded((1, 1), band, rhs)
            out[j - 1] = u
        return cls(strike, expiry, r, b, x, times, out)

    def at(self, t: float, spot: Array) -> Valuation:
        """Value and delta at time ``t`` (years from today) for spots in units of today's spot, linear in time between nodes."""
        j = int(np.clip(np.searchsorted(self.times, t), 1, self.times.size - 1))
        w = float(np.clip((t - self.times[j - 1]) / (self.times[j] - self.times[j - 1]), 0.0, 1.0))
        spline = CubicSpline(self.x, (1.0 - w) * self.values[j - 1] + w * self.values[j])
        xs = np.clip(np.log(spot) - self.b * t, self.x[0], self.x[-1])
        discount = np.exp(-self.r * (self.expiry - t))
        value = discount * spline(xs)
        delta = discount * spline(xs, 1) / spot
        return Valuation(np.asarray(value, dtype=np.float64), np.asarray(delta, dtype=np.float64), np.zeros_like(value))
