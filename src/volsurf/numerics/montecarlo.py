"""Monte Carlo that states its own error: quasi-random numbers, and early exercise bracketed from both sides.

**European options.** Pseudo-random sampling converges at ``N^(-1/2)``. Scrambled
Sobol' points (Owen scrambling, Joe and Kuo's direction numbers) fill the unit
interval evenly, and for a smooth one-dimensional integrand converge close to
``N^(-1)``. A quasi-random estimate has no standard error of its own, so the
sequence is randomised several times (randomised QMC) and the spread of the
independent estimates gives an honest error.

**Early exercise.** Longstaff and Schwartz (2001) estimate when to exercise by
regressing continuation values on functions of the spot. Priced on the same paths
the regression was fitted on, the estimate borrows foresight and is biased
upwards. Done properly it gives two bounds:

* **a lower bound**: the exercise rule, fitted on one set of paths, followed on an
  independent set - the value of a feasible policy, which no policy can beat the
  optimum by;
* **an upper bound** by duality (Rogers, 2002; Haugh and Kogan, 2004; Andersen and
  Broadie, 2004): for any martingale ``M`` starting at zero, the option is worth at
  most ``E[max_k (Z_k - M_k)]``, with ``Z`` the discounted exercise value. The
  martingale is built from the fitted value function, its increments estimated by
  inner simulations one exercise date ahead.

The true price lies between them; the width of the gap is how much the fitted rule
still leaves on the table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from .lattice import Contract

Array = NDArray[np.float64]
Sampling = Literal["pseudo", "antithetic", "control", "sobol"]


@dataclass(frozen=True)
class Estimate:
    price: float
    std_error: float
    samples: int

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        return self.price - z * self.std_error, self.price + z * self.std_error


def _terminal(c: Contract, normals: Array) -> Array:
    drift = (c.rate - c.yield_ - 0.5 * c.vol**2) * c.maturity
    return np.asarray(c.spot * np.exp(drift + c.vol * np.sqrt(c.maturity) * normals), dtype=np.float64)


def european(
    contract: Contract, samples: int, sampling: Sampling = "pseudo", seed: int = 0, randomisations: int = 16
) -> Estimate:
    """A European price by simulation of the terminal spot, with the chosen sampling."""
    c = contract
    rng = np.random.default_rng(seed)
    discount = float(np.exp(-c.rate * c.maturity))
    if sampling == "sobol":
        per = max(samples // randomisations, 2)
        m = int(np.ceil(np.log2(per)))
        estimates = []
        for r in range(randomisations):
            points = qmc.Sobol(d=1, scramble=True, seed=rng.integers(2**63) + r).random_base2(m)[:, 0]
            normals = ndtri(np.clip(points, 1e-16, 1 - 1e-16))
            estimates.append(discount * float(np.mean(c.payoff(_terminal(c, normals)))))
        values = np.array(estimates)
        return Estimate(float(values.mean()), float(values.std(ddof=1) / np.sqrt(randomisations)), randomisations * 2**m)
    if sampling == "pseudo":
        payoff = discount * c.payoff(_terminal(c, rng.standard_normal(samples)))
        return Estimate(float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(samples)), samples)
    z = rng.standard_normal(samples // 2)
    paired = 0.5 * discount * (c.payoff(_terminal(c, z)) + c.payoff(_terminal(c, -z)))
    if sampling == "antithetic":
        return Estimate(float(paired.mean()), float(paired.std(ddof=1) / np.sqrt(paired.size)), 2 * paired.size)
    if sampling == "control":
        # the discounted terminal spot is a control with known mean S e^{-qT}
        control = 0.5 * discount * (_terminal(c, z) + _terminal(c, -z))
        mean = c.spot * np.exp(-c.yield_ * c.maturity)
        beta = float(np.cov(paired, control)[0, 1] / np.var(control, ddof=1))
        adjusted = paired - beta * (control - mean)
        return Estimate(float(adjusted.mean()), float(adjusted.std(ddof=1) / np.sqrt(adjusted.size)), 2 * paired.size)
    raise ValueError(f"unknown sampling {sampling!r}")


# ---------------------------------------------------------------------------- early exercise
@dataclass(frozen=True)
class EarlyExercise:
    lower: Estimate  # the fitted rule followed on independent paths
    upper: Estimate  # the dual bound
    in_sample: float  # Longstaff-Schwartz as usually reported: rule fitted and priced on the same paths
    dates: int

    @property
    def gap(self) -> float:
        return self.upper.price - self.lower.price


def _paths(c: Contract, times: Array, n: int, rng: np.random.Generator) -> Array:
    dt = np.diff(np.concatenate([[0.0], times]))
    z = rng.standard_normal((n // 2, times.size))
    z = np.concatenate([z, -z])
    steps = (c.rate - c.yield_ - 0.5 * c.vol**2) * dt + c.vol * np.sqrt(dt) * z
    return np.asarray(c.spot * np.exp(np.cumsum(steps, axis=1)), dtype=np.float64)


def _basis(spot: Array, strike: float, degree: int, european: Array | None = None) -> Array:
    """Powers of S/K, and - when given - the European value of the remaining option, which carries most of the shape."""
    powers = np.vander(spot / strike, degree + 1, increasing=True)
    if european is None:
        return np.asarray(powers, dtype=np.float64)
    return np.asarray(np.column_stack([powers, european / strike]), dtype=np.float64)


def longstaff_schwartz(
    contract: Contract,
    exercise_times: Sequence[float],
    regression_paths: int = 100_000,
    pricing_paths: int = 100_000,
    dual_paths: int = 4_000,
    inner_paths: int = 200,
    degree: int = 3,
    seed: int = 0,
    european_basis: bool = True,
) -> EarlyExercise:
    """Bermudan (or, with dense dates, American) option bracketed by a lower and a dual upper bound."""
    c = contract
    times = np.asarray(sorted(t for t in exercise_times if 0 < t <= c.maturity), dtype=np.float64)
    if times.size == 0 or times[-1] < c.maturity:
        times = np.append(times, c.maturity)
    rng = np.random.default_rng(seed)
    discounts = np.exp(-c.rate * times)

    def basis(k: int, spot: Array) -> Array:
        if not european_basis:
            return _basis(spot, c.strike, degree)
        from ..black_scholes import bsm_price  # the price alone: the full Greek set would be wasted here

        remaining = c.maturity - times[k]
        value = np.asarray(bsm_price(spot, c.strike, remaining, c.rate, c.yield_, c.vol, c.is_call), dtype=np.float64)
        return _basis(spot, c.strike, degree, value)

    # 1. fit the exercise rule (in the money paths) and a value function (all paths), backwards
    fit = _paths(c, times, regression_paths, rng)
    cash = discounts[-1] * c.payoff(fit[:, -1])  # discounted cash flow of the policy, per path
    exercise_coef: list[Array | None] = [None] * times.size
    value_coef: list[Array | None] = [None] * times.size
    for k in range(times.size - 2, -1, -1):
        spot = fit[:, k]
        exercise_value = discounts[k] * c.payoff(spot)
        value_coef[k] = np.linalg.lstsq(basis(k, spot), cash, rcond=None)[0]
        itm = exercise_value > 0
        if itm.sum() > degree + 2:
            coef = np.linalg.lstsq(basis(k, spot[itm]), cash[itm], rcond=None)[0]
            exercise_coef[k] = coef
            go = np.zeros_like(itm)
            go[itm] = exercise_value[itm] > basis(k, spot[itm]) @ coef
            cash = np.where(go, exercise_value, cash)
    in_sample = max(float(cash.mean()), float(c.payoff(np.array([c.spot]))[0]))

    def exercise_now(k: int, spot: Array) -> Array:
        value = discounts[k] * c.payoff(spot)
        if k == times.size - 1:
            return np.asarray(value > 0)
        coef = exercise_coef[k]
        if coef is None:
            return np.zeros(spot.shape, dtype=bool)
        return np.asarray((value > 0) & (value > basis(k, spot) @ coef))

    def value_function(k: int, spot: Array) -> Array:
        exercise_value = discounts[k] * c.payoff(spot)
        if k == times.size - 1:
            return np.asarray(exercise_value, dtype=np.float64)
        coef = value_coef[k]
        continuation = basis(k, spot) @ coef if coef is not None else np.zeros_like(spot)
        return np.asarray(np.where(exercise_now(k, spot), exercise_value, continuation), dtype=np.float64)

    # 2. the lower bound: follow the rule on independent paths
    price_paths = _paths(c, times, pricing_paths, rng)
    alive = np.ones(pricing_paths, dtype=bool)
    received = np.zeros(pricing_paths)
    for k in range(times.size):
        go = alive & exercise_now(k, price_paths[:, k])
        received[go] = discounts[k] * c.payoff(price_paths[go, k])
        alive &= ~go
    half = pricing_paths // 2
    pairs = 0.5 * (received[:half] + received[half:])
    lower = Estimate(float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(half)), pricing_paths)

    # 3. the upper bound by duality, the martingale's increments from inner one-step simulations
    outer = _paths(c, times, dual_paths, rng)
    martingale = np.zeros(dual_paths)
    # at time zero nothing has happened: the martingale starts at 0 and the first increment uses S0
    previous_spot = np.full(dual_paths, c.spot)
    previous_time = 0.0
    best = np.full(dual_paths, -np.inf)
    for k in range(times.size):
        dt = times[k] - previous_time
        z = rng.standard_normal((dual_paths, inner_paths // 2))
        z = np.concatenate([z, -z], axis=1)  # antithetic inner draws
        inner = previous_spot[:, None] * np.exp((c.rate - c.yield_ - 0.5 * c.vol**2) * dt + c.vol * np.sqrt(dt) * z)
        expected = value_function(k, inner.ravel()).reshape(inner.shape).mean(axis=1)
        martingale += value_function(k, outer[:, k]) - expected
        best = np.maximum(best, discounts[k] * c.payoff(outer[:, k]) - martingale)
        previous_spot, previous_time = outer[:, k], float(times[k])
    upper = Estimate(float(best.mean()), float(best.std(ddof=1) / np.sqrt(dual_paths)), dual_paths)
    return EarlyExercise(lower, upper, in_sample, int(times.size))
