"""Monte Carlo engines.

* Geometric Brownian motion with antithetic sampling and an S_T control variate.
* Heston paths with the Quadratic-Exponential (QE) scheme of Andersen (2008), including
  his martingale correction, so the discounted asset price is an exact martingale
  in the discretised model.
* Longstaff-Schwartz (2001) least-squares Monte Carlo for American options.

Every estimator reports a standard error computed from independent antithetic pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .heston import HestonParams


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    def contains(self, value: float, n_std: float = 4.0, abs_tol: float = 0.0) -> bool:
        return abs(self.price - value) <= n_std * self.std_error + abs_tol


def _antithetic_normals(rng: np.random.Generator, n_pairs: int, *shape: int) -> np.ndarray:
    z = rng.standard_normal((n_pairs, *shape))
    return np.concatenate([z, -z], axis=0)


def _pair_means(samples: np.ndarray) -> np.ndarray:
    n_pairs = samples.shape[0] // 2
    return 0.5 * (samples[:n_pairs] + samples[n_pairs:])


def _control_variate_estimate(payoff: np.ndarray, control: np.ndarray, control_mean: float) -> tuple[float, float]:
    y = _pair_means(payoff)
    x = _pair_means(control)
    var_x = np.var(x)
    beta = np.cov(y, x, ddof=0)[0, 1] / var_x if var_x > 0 else 0.0
    adjusted = y - beta * (x - control_mean)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / np.sqrt(adjusted.size))


def simulate_gbm_terminal(
    spot: float, maturity: float, rate: float, dividend_yield: float, vol: float, n_paths: int, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = _antithetic_normals(rng, n_paths // 2)
    drift = (rate - dividend_yield - 0.5 * vol**2) * maturity
    return spot * np.exp(drift + vol * np.sqrt(maturity) * z)


def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Exact log-normal paths, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    z = _antithetic_normals(rng, n_paths // 2, n_steps)
    increments = (rate - dividend_yield - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
    return spot * np.exp(log_paths)


def mc_european_gbm(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    is_call: bool = True,
    n_paths: int = 200_000,
    seed: int | None = None,
    control_variate: bool = True,
) -> MCResult:
    terminal = simulate_gbm_terminal(spot, maturity, rate, dividend_yield, vol, n_paths, seed)
    discount = np.exp(-rate * maturity)
    payoff = discount * np.maximum((terminal - strike) if is_call else (strike - terminal), 0.0)
    if control_variate:
        price, se = _control_variate_estimate(payoff, discount * terminal, spot * np.exp(-dividend_yield * maturity))
    else:
        pairs = _pair_means(payoff)
        price, se = float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(pairs.size))
    return MCResult(price, se, terminal.size)


@dataclass(frozen=True)
class HestonPaths:
    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray


def simulate_heston_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    params: HestonParams,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    psi_critical: float = 1.5,
) -> HestonPaths:
    """Andersen (2008) QE discretisation with the martingale-corrected log-price step."""
    rng = np.random.default_rng(seed)
    kappa, theta, sigma, rho = params.kappa, params.theta, params.sigma, params.rho
    dt = maturity / n_steps
    n_paths = 2 * (n_paths // 2)

    exp_k = np.exp(-kappa * dt)
    k1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = 0.5 * dt * (1.0 - rho**2)
    k4 = k3
    big_a = k2 + 0.5 * k4
    k0_plain = -rho * kappa * theta / sigma * dt

    log_s = np.empty((n_paths, n_steps + 1))
    var = np.empty((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(spot)
    var[:, 0] = params.v0

    for step in range(n_steps):
        v = var[:, step]
        z_v = _antithetic_normals(rng, n_paths // 2)
        z_s = _antithetic_normals(rng, n_paths // 2)

        m = theta + (v - theta) * exp_k
        s2 = v * sigma**2 * exp_k / kappa * (1.0 - exp_k) + theta * sigma**2 / (2.0 * kappa) * (1.0 - exp_k) ** 2
        psi = s2 / np.maximum(m * m, 1e-300)

        v_next = np.empty_like(v)
        k0 = np.empty_like(v)

        quad = psi <= psi_critical
        if quad.any():
            inv_psi = 1.0 / psi[quad]
            b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi) * np.sqrt(np.maximum(2.0 * inv_psi - 1.0, 0.0))
            a = m[quad] / (1.0 + b2)
            v_next[quad] = a * (np.sqrt(b2) + z_v[quad]) ** 2
            ok = big_a < 1.0 / (2.0 * a)
            corrected = -big_a * b2 * a / (1.0 - 2.0 * big_a * a) + 0.5 * np.log(np.where(ok, 1.0 - 2.0 * big_a * a, 1.0))
            k0[quad] = np.where(ok, corrected - (k1 + 0.5 * k3) * v[quad], k0_plain)

        expo = ~quad
        if expo.any():
            p = (psi[expo] - 1.0) / (psi[expo] + 1.0)
            beta = (1.0 - p) / m[expo]
            uniform = ndtr(z_v[expo])
            v_next[expo] = np.where(uniform <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - uniform, 1e-300)) / beta)
            ok = big_a < beta
            ratio = np.where(ok, p + beta * (1.0 - p) / np.where(ok, beta - big_a, 1.0), 1.0)
            k0[expo] = np.where(ok, -np.log(ratio) - (k1 + 0.5 * k3) * v[expo], k0_plain)

        diffusion = np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0))
        log_s[:, step + 1] = log_s[:, step] + (rate - dividend_yield) * dt + k0 + k1 * v + k2 * v_next + diffusion * z_s
        var[:, step + 1] = v_next

    return HestonPaths(times=np.linspace(0.0, maturity, n_steps + 1), spot=np.exp(log_s), variance=var)


def mc_european_heston(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    params: HestonParams,
    is_call: bool = True,
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    paths = simulate_heston_paths(spot, maturity, rate, dividend_yield, params, n_steps, n_paths, seed)
    terminal = paths.spot[:, -1]
    discount = np.exp(-rate * maturity)
    payoff = discount * np.maximum((terminal - strike) if is_call else (strike - terminal), 0.0)
    price, se = _control_variate_estimate(payoff, discount * terminal, spot * np.exp(-dividend_yield * maturity))
    return MCResult(price, se, terminal.size)


def lsm_american_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    is_call: bool = False,
    n_exercise_dates: int = 50,
    n_paths: int = 100_000,
    degree: int = 3,
    seed: int | None = None,
) -> MCResult:
    """Longstaff-Schwartz: regress discounted continuation values on polynomials of S/K
    using in-the-money paths only, exercising when the immediate payoff is larger."""
    paths = simulate_gbm_paths(spot, maturity, rate, dividend_yield, vol, n_exercise_dates, n_paths, seed)
    dt = maturity / n_exercise_dates
    step_discount = np.exp(-rate * dt)
    sign = 1.0 if is_call else -1.0

    cash_flow = np.maximum(sign * (paths[:, -1] - strike), 0.0)
    for t in range(n_exercise_dates - 1, 0, -1):
        cash_flow *= step_discount
        exercise_value = np.maximum(sign * (paths[:, t] - strike), 0.0)
        itm = exercise_value > 0.0
        if itm.sum() <= degree + 1:
            continue
        x = paths[itm, t] / strike
        basis = np.vander(x, degree + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(basis, cash_flow[itm], rcond=None)
        continuation = basis @ coefficients
        exercise_now = exercise_value[itm] > continuation
        idx = np.flatnonzero(itm)[exercise_now]
        cash_flow[idx] = exercise_value[idx]

    discounted = cash_flow * step_discount
    pairs = _pair_means(discounted)
    price = float(pairs.mean())
    immediate = max(sign * (spot - strike), 0.0)
    return MCResult(max(price, immediate), float(pairs.std(ddof=1) / np.sqrt(pairs.size)), discounted.size)
