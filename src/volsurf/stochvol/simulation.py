"""Simulating Heston, on both sides of the Feller boundary.

When ``2 kappa theta < sigma^2`` (Feller's condition fails, as it does for most fitted
equity surfaces) the variance process reaches zero with positive probability, and a naive
discretisation spends much of its time there or below it.

* **Euler with full truncation** (Lord, Koekkoek and van Dijk, 2010) uses ``v^+`` in every
  coefficient and lets the variance itself go negative. It converges, at first order and
  slowly when Feller fails.
* **Quadratic-exponential, QE** (Andersen, 2008) matches the first two moments of the exact
  variance transition with a squared normal where the variance is far from zero and with a
  mass at zero plus an exponential tail where it is near. The log-price step integrates the
  variance by the trapezoid rule and, with Andersen's martingale correction, keeps
  ``E[S_T] = F`` exactly whatever the step.

The exact transition is known: ``v_{t+dt} | v_t`` is a scaled non-central chi-squared
variable, which is what QE is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .characteristic import Heston

Array = NDArray[np.float64]
PSI_CRITICAL = 1.5


def exact_variance(v: Array, dt: float, h: Heston, rng: np.random.Generator) -> Array:
    """Draw ``v_{t+dt}`` from the exact non-central chi-squared transition."""
    c = h.sigma**2 * (1.0 - np.exp(-h.kappa * dt)) / (4.0 * h.kappa)
    df = 4.0 * h.kappa * h.theta / h.sigma**2
    return np.asarray(c * rng.noncentral_chisquare(df, v * np.exp(-h.kappa * dt) / c), dtype=np.float64)


def exact_variance_cdf(x: Array, v: float, dt: float, h: Heston) -> Array:
    c = h.sigma**2 * (1.0 - np.exp(-h.kappa * dt)) / (4.0 * h.kappa)
    df = 4.0 * h.kappa * h.theta / h.sigma**2
    return np.asarray(stats.ncx2.cdf(x / c, df, v * np.exp(-h.kappa * dt) / c), dtype=np.float64)


def qe_variance(v: Array, dt: float, h: Heston, z: Array, u: Array) -> tuple[Array, Array, Array, Array]:
    """One QE step: the new variance, and the branch's (a, b^2) or (p, beta) for the martingale correction."""
    e = np.exp(-h.kappa * dt)
    m = h.theta + (v - h.theta) * e
    s2 = v * h.sigma**2 * e * (1.0 - e) / h.kappa + h.theta * h.sigma**2 * (1.0 - e) ** 2 / (2.0 * h.kappa)
    psi = s2 / np.maximum(m * m, 1e-300)
    quadratic = psi <= PSI_CRITICAL
    inv = 2.0 / np.where(quadratic, psi, 1.0)
    b2 = np.where(quadratic, inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0)), 0.0)
    a = np.where(quadratic, m / (1.0 + b2), 0.0)
    p = np.where(quadratic, 0.0, (psi - 1.0) / (psi + 1.0))
    beta = np.where(quadratic, 0.0, (1.0 - p) / np.maximum(m, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = np.where(u > p, np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / np.where(beta > 0, beta, 1.0), 0.0)
    new = np.where(quadratic, a * (np.sqrt(b2) + z) ** 2, tail)
    return np.asarray(new, dtype=np.float64), quadratic, np.where(quadratic, a, p), np.where(quadratic, b2, beta)


@dataclass(frozen=True)
class Paths:
    log_moneyness: Array  # ln(S_T / F_T) per path
    variance: Array  # v_T per path


def simulate(h: Heston, t: float, steps: int, paths: int, scheme: str = "qe", seed: int = 1) -> Paths:
    """Terminal ``ln(S_T/F_T)`` and ``v_T`` under ``scheme``: ``"qe"`` or ``"euler"`` (full truncation)."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    x = np.zeros(paths)
    v = np.full(paths, h.v0)
    root = np.sqrt(1.0 - h.rho**2)
    for _ in range(steps):
        z_v = rng.standard_normal(paths)
        z_s = rng.standard_normal(paths)
        if scheme == "euler":
            vp = np.maximum(v, 0.0)
            x = x - 0.5 * vp * dt + np.sqrt(vp * dt) * (h.rho * z_v + root * z_s)
            v = v + h.kappa * (h.theta - vp) * dt + h.sigma * np.sqrt(vp * dt) * z_v
            continue
        if scheme != "qe":
            raise ValueError(f"unknown scheme {scheme!r}")
        u = rng.random(paths)
        v_new, quadratic, first, second = qe_variance(v, dt, h, z_v, u)
        g1 = g2 = 0.5
        k1 = g1 * dt * (h.kappa * h.rho / h.sigma - 0.5) - h.rho / h.sigma
        k2 = g2 * dt * (h.kappa * h.rho / h.sigma - 0.5) + h.rho / h.sigma
        k3, k4 = g1 * dt * (1.0 - h.rho**2), g2 * dt * (1.0 - h.rho**2)
        big_a = k2 + 0.5 * k4
        # Andersen's martingale correction: K0 = -ln E[exp(A v_new) | v] - (K1 + K3/2) v
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            quad_mgf = np.exp(big_a * second * first / (1.0 - 2.0 * big_a * first)) / np.sqrt(1.0 - 2.0 * big_a * first)
            exp_mgf = first + second * (1.0 - first) / (second - big_a)
        mgf = np.where(quadratic, quad_mgf, exp_mgf)
        k0 = -np.log(mgf) - (k1 + 0.5 * k3) * v
        x = x + k0 + k1 * v + k2 * v_new + np.sqrt(np.maximum(k3 * v + k4 * v_new, 0.0)) * z_s
        v = v_new
    return Paths(x, v)


@dataclass(frozen=True)
class Estimate:
    value: Array
    stderr: Array


def otm_estimate(paths: Paths, log_moneyness: Array) -> Estimate:
    """Monte Carlo out-of-the-money prices over the forward, with standard errors."""
    s = np.exp(paths.log_moneyness)
    values, errors = [], []
    for k in np.atleast_1d(log_moneyness):
        payoff = np.maximum(s - np.exp(k), 0.0) if k >= 0 else np.maximum(np.exp(k) - s, 0.0)
        values.append(payoff.mean())
        errors.append(payoff.std(ddof=1) / np.sqrt(payoff.size))
    return Estimate(np.array(values), np.array(errors))
