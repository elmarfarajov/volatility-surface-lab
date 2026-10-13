"""Monte Carlo for exotics: discrete monitoring, the Brownian bridge, continuity corrections, control variates.

**Discrete against continuous monitoring.** A barrier checked once a day is crossed less
often than one watched continuously. Broadie, Glasserman and Kou (1997) showed that the
discretely monitored price is, to first order, the continuous price with the barrier moved
away from the spot by ``exp(beta sigma sqrt(dt))``, ``beta = -zeta(1/2)/sqrt(2 pi) = 0.5826``.

**The Brownian bridge.** Between two monitoring dates a log-normal path crosses a barrier
``H`` with known probability, ``exp(-2 ln(S_i/H) ln(S_{i+1}/H) / (sigma^2 dt))``. Multiplying
the knock-out payoff by the probability of surviving every interval gives an estimator of
the *continuous* price from a coarse grid, unbiased under Black-Scholes and smoother than
the indicator.

**Control variates.** The arithmetic average has no closed form; the geometric average of
the same path does (Kemna and Vorst, 1990), and the two are almost perfectly correlated.

Paths come from three models on one interface: Black-Scholes (exact log-normal steps),
Heston (Andersen's QE with the martingale correction), and Dupire local volatility (log-Euler
on the forward-normalised price), so the same exotic can be priced under each.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import zeta

from ..localvol.dupire import local_variance
from ..smile.interpolation import Surface
from ..stochvol.characteristic import Heston
from ..stochvol.simulation import qe_variance
from . import closed_form as cf

Array = NDArray[np.float64]
BETA = float(-zeta(0.5) / np.sqrt(2.0 * np.pi))  # 0.5826...


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float


def _estimate(samples: Array) -> Estimate:
    return Estimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size)))


# ---------------------------------------------------------------- paths


def black_scholes_paths(s: float, t: float, r: float, b: float, sigma: float, steps: int, paths: int, seed: int = 1) -> Array:
    """Spot at ``steps`` equally spaced dates after today (columns), exact log-normal steps, antithetic."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((paths // 2, steps))
    z = np.vstack([z, -z])
    log_steps = (b - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return np.asarray(s * np.exp(np.cumsum(log_steps, axis=1)), dtype=np.float64)


def heston_paths(s: float, t: float, r: float, b: float, h: Heston, steps: int, paths: int, seed: int = 1) -> Array:
    """Spot at ``steps`` dates under Heston, by QE with Andersen's martingale correction."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    x = np.zeros(paths)
    v = np.full(paths, h.v0)
    out = np.empty((paths, steps))
    k1 = 0.5 * dt * (h.kappa * h.rho / h.sigma - 0.5) - h.rho / h.sigma
    k2 = 0.5 * dt * (h.kappa * h.rho / h.sigma - 0.5) + h.rho / h.sigma
    k3 = k4 = 0.5 * dt * (1.0 - h.rho**2)
    big_a = k2 + 0.5 * k4
    for j in range(steps):
        v_new, quadratic, first, second = qe_variance(v, dt, h, rng.standard_normal(paths), rng.random(paths))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            quad_mgf = np.exp(big_a * second * first / (1.0 - 2.0 * big_a * first)) / np.sqrt(1.0 - 2.0 * big_a * first)
            exp_mgf = first + second * (1.0 - first) / (second - big_a)
        k0 = -np.log(np.where(quadratic, quad_mgf, exp_mgf)) - (k1 + 0.5 * k3) * v
        x = x + k0 + k1 * v + k2 * v_new + np.sqrt(np.maximum(k3 * v + k4 * v_new, 0.0)) * rng.standard_normal(paths)
        v = v_new
        out[:, j] = x
    forward = s * np.exp(b * dt * np.arange(1, steps + 1))
    return np.asarray(forward[None, :] * np.exp(out), dtype=np.float64)


def local_vol_paths(s: float, t: float, r: float, b: float, surface: Surface, steps: int, paths: int, seed: int = 1) -> Array:
    """Spot at ``steps`` dates under Dupire local volatility: log-Euler on ``X = ln(S/F_t)``, a martingale in ``e^X``."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    x = np.zeros(paths)
    out = np.empty((paths, steps))
    for j in range(steps):
        variance = local_variance(surface, x, max(j * dt, 1e-6))
        variance = np.where(np.isfinite(variance), np.clip(variance, 0.0, 25.0), 0.0)
        x = x - 0.5 * variance * dt + np.sqrt(variance * dt) * rng.standard_normal(paths)
        out[:, j] = x
    forward = s * np.exp(b * dt * np.arange(1, steps + 1))
    return np.asarray(forward[None, :] * np.exp(out), dtype=np.float64)


# ---------------------------------------------------------------- barriers


def barrier_payoff(paths: Array, s: float, kind: str, x: float, h: float, call: bool) -> Array:
    """Payoff of a discretely monitored barrier option on the path's monitoring dates (no rebate)."""
    full = np.column_stack([np.full(paths.shape[0], s), paths])
    hit = (full.min(axis=1) <= h) if kind.startswith("down") else (full.max(axis=1) >= h)
    vanilla = np.maximum(paths[:, -1] - x, 0.0) if call else np.maximum(x - paths[:, -1], 0.0)
    alive = ~hit if kind.endswith("out") else hit
    return np.asarray(np.where(alive, vanilla, 0.0), dtype=np.float64)


def survival_bridge(paths: Array, s: float, h: float, sigma: float, dt: float, down: bool) -> Array:
    """Probability that a log-normal path avoided ``h`` between every pair of monitoring dates (Brownian bridge)."""
    full = np.column_stack([np.full(paths.shape[0], s), paths])
    a, c = np.log(full[:, :-1] / h), np.log(full[:, 1:] / h)
    on_side = (a > 0) & (c > 0) if down else (a < 0) & (c < 0)
    cross = np.where(on_side, np.exp(-2.0 * a * c / (sigma**2 * dt)), 1.0)
    return np.asarray(np.prod(1.0 - cross, axis=1), dtype=np.float64)


def barrier_mc(
    kind: str,
    s: float,
    x: float,
    h: float,
    t: float,
    r: float,
    b: float,
    sigma: float,
    call: bool = True,
    monitoring: int = 252,
    paths: int = 200_000,
    bridge: bool = False,
    seed: int = 1,
) -> Estimate:
    """Monte Carlo price of a knock-out (or knock-in by parity) barrier option.

    ``bridge=False``: monitored on ``monitoring`` dates. ``bridge=True``: continuous monitoring
    estimated on that grid, the knock-out weighted by the Brownian-bridge survival probability.
    """
    p = black_scholes_paths(s, t, r, b, sigma, monitoring, paths, seed)
    disc = np.exp(-r * t)
    if bridge:
        vanilla = np.maximum(p[:, -1] - x, 0.0) if call else np.maximum(x - p[:, -1], 0.0)
        # an endpoint beyond the barrier makes that interval's crossing certain, so discrete hits are included
        knock_out = vanilla * survival_bridge(p, s, h, sigma, t / monitoring, kind.startswith("down"))
        samples = disc * (vanilla - knock_out if kind.endswith("in") else knock_out)
    else:
        samples = disc * barrier_payoff(p, s, kind, x, h, call)
    return _estimate(samples)


def bgk_barrier(
    kind: str, s: float, x: float, h: float, t: float, r: float, b: float, sigma: float, call: bool, monitoring: int
) -> float:
    """Broadie, Glasserman and Kou's continuity correction: the continuous price with the barrier shifted away from spot."""
    shift = np.exp(BETA * sigma * np.sqrt(t / monitoring))
    moved = h / shift if kind.startswith("down") else h * shift
    return cf.barrier(kind, s, x, moved, t, r, b, sigma, call)  # type: ignore[arg-type]


# ---------------------------------------------------------------- Asians


@dataclass(frozen=True)
class AsianEstimate:
    plain: Estimate
    controlled: Estimate
    variance_ratio: float  # plain variance over controlled variance


def arithmetic_asian(
    s: float,
    x: float,
    t: float,
    r: float,
    b: float,
    sigma: float,
    fixings: int,
    call: bool = True,
    paths: int = 100_000,
    seed: int = 1,
) -> AsianEstimate:
    """Arithmetic-average-price option over ``fixings`` equally spaced dates, plain and with the geometric control variate."""
    p = black_scholes_paths(s, t, r, b, sigma, fixings, paths, seed)
    disc = np.exp(-r * t)
    arithmetic = p.mean(axis=1)
    geometric = np.exp(np.log(p).mean(axis=1))
    sign = 1.0 if call else -1.0
    y = disc * np.maximum(sign * (arithmetic - x), 0.0)
    g = disc * np.maximum(sign * (geometric - x), 0.0)
    exact = cf.geometric_asian(s, x, t, r, b, sigma, call, fixings)
    cov = np.cov(y, g)
    beta = cov[0, 1] / cov[1, 1]
    controlled = y - beta * (g - exact)
    return AsianEstimate(_estimate(y), _estimate(controlled), float(y.var() / controlled.var()))


# ---------------------------------------------------------------- generic payoffs


def price(paths: Array, payoff: Callable[[Array], Array], discount: float) -> Estimate:
    return _estimate(discount * payoff(paths))
