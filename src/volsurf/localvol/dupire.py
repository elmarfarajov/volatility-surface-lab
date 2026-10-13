"""Dupire local volatility from the arbitrage-free surface, in closed form, and the forward equation that checks it.

For the undiscounted call over the forward, ``c(k, t) = E[(S_t/F_t - e^k)^+]``, Dupire's
equation in log-moneyness reads

    dc/dt = sigma_loc^2(k, t) / 2 * (c_kk - c_k),     c_kk - c_k = e^k p(k, t),

where ``p`` is the density of ``ln(S_t/F_t)``. So local variance is twice the time
derivative of the call over its density. Taken by finite differences from a fitted
surface, both are noisy, and a small or negative density turns the ratio into nonsense:
the original ``VolSurface.local_vol`` returns NaN wherever that happens.

On the Day 4 surface neither is approximated. Between two expiries the call is the
price combination ``c = alpha c_lo + (1 - alpha) c_hi`` with ``theta`` linear in time, so

    dc/dt = -alpha'(t) (c_hi - c_lo),     alpha'(t) = -theta' / (2 sqrt(theta_t) (sqrt(theta_hi) - sqrt(theta_lo))),

which is non-negative because the slices are free of calendar arbitrage. The density is
the same combination of the slices' closed-form densities. Local variance is therefore
positive wherever the density is, by construction. Two consequences:

* ``theta'`` jumps at every expiry, so local volatility is piecewise smooth in time with a
  jump at each fitted expiry. That is the price of an interpolation that is free of
  arbitrage; it is shown, not smoothed.
* the price combination of two log-normal laws is not log-normal: between the expiries of a
  flat 20% surface it puts a smile of 2.8 volatility points at ``ln(K/F) = -0.4`` (a fifth
  of the way between expiries a quarter-year apart), and local volatility follows. The
  interpolation is free of arbitrage, not minimal.
* beyond the last expiry, where the distribution is convolved with log-normal noise, local
  variance is the noise's rate, ``theta_n / t_n``, at every strike;
* **before the first expiry** the interpolation mixes the first slice with expiry itself,
  ``c = alpha (1 - e^k)^+ + (1 - alpha) c_1``: the distribution keeps an atom of weight
  ``alpha`` at the forward. No diffusion can carry an atom, local variance is infinite at
  the forward and grows like ``1/t`` as ``t -> 0``, and a forward solve from ``t = 0``
  diverged as its grid was refined. Before the first expiry, local volatility therefore
  comes from the self-similar short end ``w(k, t) = (t/t_1) w_1(k)``, which reaches the first
  slice exactly and is a diffusion's;
* between closely spaced expiries - SPX lists one every trading day - ``theta`` rises by a
  little, any disagreement between neighbouring slices is divided by a small time step,
  and local volatility in the wings swings widely (from 8% to 264% at ``k = -0.1`` within
  the first two weeks of the 9 October 2026 surface). Local volatility from a dense set of
  weeklies is noise in its wings; for dynamics, monthly expiries are used.

The check is the **forward Dupire equation**: solved from ``c(k, 0) = (1 - e^k)^+`` with this
local volatility, it must return the surface's prices at every expiry, strike by strike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from ..market.implied import invert_normalised
from ..smile.density import density, option_price
from ..smile.interpolation import Surface

Array = NDArray[np.float64]


def short_end_variance(surface: Surface, k: ArrayLike, t: float) -> Array:
    """Local variance before the first expiry on the self-similar short end ``w(k, t) = (t/t_1) w_1(k)``.

    In total variance Dupire reads ``sigma_loc^2 = (dw/dt) / g(k, w)``; here ``dw/dt = w_1(k)/t_1``
    and ``g`` is Durrleman's function of the scaled slice.
    """
    k = np.asarray(k, dtype=np.float64)
    first, t1 = surface.slices[0], float(surface.maturities[0])
    scale = t / t1
    w, w1, w2 = scale * first.total_variance(k), scale * first.dw(k), scale * first.d2w(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2
        return np.asarray(np.where(g > 0, first.total_variance(k) / t1 / g, np.nan), dtype=np.float64)


def local_variance(surface: Surface, k: ArrayLike, t: float, short_end: str = "self-similar") -> Array:
    """Dupire local variance at log-moneyness ``k`` and time ``t``, from the price interpolation in closed form.

    Before the first expiry the price interpolation carries an atom (see the module), so by
    default the self-similar short end of :func:`short_end_variance` is used there instead;
    ``short_end="interpolation"`` keeps the interpolation's own, singular, local variance.
    """
    k = np.asarray(k, dtype=np.float64)
    times, thetas = surface.maturities, surface.thetas
    if t > times[-1]:
        return np.full(k.shape, float(thetas[-1] / times[-1]))
    if t <= times[0] and short_end == "self-similar":
        return short_end_variance(surface, k, t)
    i = int(np.searchsorted(times, t))  # t in (times[i-1], times[i]]
    hi = surface.slices[i]
    theta_hi = float(thetas[i])
    t_lo = float(times[i - 1]) if i > 0 else 0.0
    theta_lo = float(thetas[i - 1]) if i > 0 else 0.0
    slope = (theta_hi - theta_lo) / (float(times[i]) - t_lo)
    theta_t = surface.theta(t)
    span = np.sqrt(theta_hi) - np.sqrt(theta_lo)
    alpha = (np.sqrt(theta_hi) - np.sqrt(theta_t)) / span
    d_alpha = -slope / (2.0 * np.sqrt(theta_t) * span)
    call = k >= 0
    hi_price = option_price(hi.total_variance(k), k, call)
    if i > 0:
        lo = surface.slices[i - 1]
        lo_price = option_price(lo.total_variance(k), k, call)
        lo_density = density(lo, k)
    else:
        lo_price = np.zeros_like(k)  # expiry: the out-of-the-money option is worth nothing
        lo_density = np.zeros_like(k)  # and the density is a point mass at zero
    d_call = -d_alpha * (hi_price - lo_price)  # out-of-the-money prices differ from calls by intrinsic value only
    mixed = alpha * lo_density + (1.0 - alpha) * density(hi, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(mixed > 0, 2.0 * d_call / (np.exp(k) * mixed), np.nan), dtype=np.float64)


def local_vol(surface: Surface, k: ArrayLike, t: float) -> Array:
    return np.asarray(np.sqrt(local_variance(surface, k, t)), dtype=np.float64)


@dataclass(frozen=True)
class ForwardSolution:
    k: Array
    times: Array
    calls: Array  # undiscounted calls over the forward, one row per requested time


def solve_forward(
    variance: object,
    times: Array,
    k_min: float = -3.0,
    k_max: float = 1.5,
    space_steps: int = 900,
    steps_per_year: int = 800,
    rannacher: int = 4,
    start: tuple[float, object] | None = None,
) -> ForwardSolution:
    """Solve ``c_t = v(k, t)/2 (c_kk - c_k)`` forward, reporting ``c`` at ``times``.

    From ``(1 - e^k)^+`` at ``t = 0`` by default, or from ``start = (t0, calls(k))``.
    ``variance(k, t)`` is the local variance. Crank-Nicolson after ``rannacher`` implicit
    half-steps, which damp the kink of a payoff; the boundaries are the call's limits,
    ``1 - e^k`` deep in the money and zero far out.
    """
    assert callable(variance)
    k = np.linspace(k_min, k_max, space_steps + 1)
    h = k[1] - k[0]
    if start is None:
        t, c = 0.0, np.maximum(1.0 - np.exp(k), 0.0)
    else:
        t, initial = start
        assert callable(initial)
        c = np.asarray(initial(k), dtype=np.float64)
    out = np.empty((len(times), k.size))
    n = k.size
    taken = 0  # full steps so far: the first rannacher/2 of them are two implicit half-steps each
    for index, target in enumerate(np.asarray(times, dtype=np.float64)):
        steps = max(int(np.ceil((target - t) * steps_per_year)), 1)
        dt = (target - t) / steps
        for _ in range(steps):
            schedule = [(0.5 * dt, 1.0), (0.5 * dt, 1.0)] if taken < rannacher // 2 else [(dt, 0.5)]
            taken += 1
            for sub_dt, theta in schedule:
                v_now = np.asarray(variance(k, t + sub_dt), dtype=np.float64)
                v_now = np.where(np.isfinite(v_now), v_now, 0.0)
                a = 0.5 * v_now * (1.0 / h**2 + 0.5 / h)  # coefficient of c_{j-1}
                b = -0.5 * v_now * (2.0 / h**2)  # of c_j
                d = 0.5 * v_now * (1.0 / h**2 - 0.5 / h)  # of c_{j+1}
                rhs = c.copy()
                if theta < 1.0:
                    v_old = np.asarray(variance(k, t), dtype=np.float64)
                    v_old = np.where(np.isfinite(v_old), v_old, 0.0)
                    a0, b0, d0 = 0.5 * v_old * (1.0 / h**2 + 0.5 / h), -v_old / h**2, 0.5 * v_old * (1.0 / h**2 - 0.5 / h)
                    rhs[1:-1] += (1.0 - theta) * sub_dt * (a0[1:-1] * c[:-2] + b0[1:-1] * c[1:-1] + d0[1:-1] * c[2:])
                band = np.zeros((3, n))
                band[0, 2:] = -theta * sub_dt * d[1:-1]
                band[1, 1:-1] = 1.0 - theta * sub_dt * b[1:-1]
                band[2, :-2] = -theta * sub_dt * a[1:-1]
                band[1, 0] = band[1, -1] = 1.0
                rhs[0], rhs[-1] = 1.0 - np.exp(k[0]), 0.0
                c = solve_banded((1, 1), band, rhs)
                t += sub_dt
        out[index] = c
    return ForwardSolution(k, np.asarray(times, dtype=np.float64), out)


def implied_from_calls(k: Array, calls: Array, t: float) -> Array:
    """Implied volatility of undiscounted calls over the forward (out-of-the-money side inverted)."""
    otm = np.where(k >= 0, calls, calls - (1.0 - np.exp(k)))
    s = invert_normalised(otm * np.exp(-0.5 * k), -k, k >= 0).total_vol
    return np.asarray(s / np.sqrt(t), dtype=np.float64)
