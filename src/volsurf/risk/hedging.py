"""Discrete hedging on simulated paths: any model's world, any hedge, and the P&L it leaves.

A dealer sells an option for a premium and holds hedging instruments - the underlying, and
optionally other options - rebalanced on a schedule. Cash earns the rate and the underlying
pays its dividend yield. At expiry the book is closed and its P&L, discounted to today, is
recorded path by path. Nothing about the world is assumed by the accounting: the same
function measures a Black-Scholes delta hedge in a Heston world and a local-volatility hedge
in a Black-Scholes one.

**Two results the module is held to.**

* *Derman and Kamal (1999).* Delta-hedging a call N times in a Black-Scholes world leaves a
  P&L whose standard deviation is ``sqrt(pi/4) vega sigma / sqrt(N)`` to leading order.
  Adding a gamma hedge with a second option makes it fall faster - measured here as about
  ``N^-0.7``, not the ``1/N`` of the leading-order argument, because near expiry the gamma
  of paths close to the strike changes within a single rebalancing interval.
* *Hedging at the wrong volatility.* A dealer who sells at volatility ``sigma_h`` and
  delta-hedges at it, on a path that realises something else, earns

      P&L = 1/2 sum_i e^{-r t_i} Gamma_i S_i^2 [ sigma_h^2 dt - (dS_i / S_i)^2 ],

  with ``Gamma`` at ``sigma_h``: the option's gamma times the gap between hedging and
  realised variance, weighted by where the spot was when the variance was realised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..analytic.generalized import greeks

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Instrument:
    """A hedging instrument's value on every path and date (``[paths, dates]``); the underlying is built in."""

    name: str
    values: Array


@dataclass(frozen=True)
class Outcome:
    strategy: str
    rebalances: int
    premium: float
    pnl: Array  # discounted to today, per path

    def stats(self) -> dict[str, float]:
        pnl = self.pnl
        tail = np.sort(pnl)[: max(1, int(0.05 * pnl.size))]
        return {
            "mean": float(pnl.mean()),
            "std": float(pnl.std(ddof=1)),
            "std_pct_premium": float(100.0 * pnl.std(ddof=1) / self.premium),
            "es_05": float(tail.mean()),
        }


def hedge(
    spot: Array,
    times: Array,
    r: float,
    q: float,
    premium: float,
    payoff: Array,
    stock: Array,
    others: Sequence[tuple[Instrument, Array]] = (),
    every: int = 1,
) -> Array:
    """Discounted P&L of selling an option for ``premium`` and holding ``stock`` shares (and ``others``) between rebalances.

    ``spot`` and the instruments' values are ``[paths, dates]`` with ``times[0] = 0`` and
    ``times[-1]`` the option's expiry. ``stock`` and each holding array are ``[paths, dates - 1]``:
    the position chosen at each date; only every ``every``-th date is used, and positions are
    held in between. Instruments are valued at expiry on the last column.
    """
    n_paths, n_dates = spot.shape
    hedge_dates = np.arange(0, n_dates - 1, every)
    position = np.zeros(n_paths)
    held = [np.zeros(n_paths) for _ in others]
    cash = np.full(n_paths, premium)
    last = 0.0
    for i in range(n_dates - 1):
        dt = times[i] - last
        if i > 0:
            cash = cash * np.exp(r * dt) + position * spot[:, i] * (np.exp(q * dt) - 1.0)
        last = times[i]
        if i in hedge_dates:
            new = stock[:, i]
            cash -= (new - position) * spot[:, i]
            position = new
            for j, (instrument, holding) in enumerate(others):
                cash -= (holding[:, i] - held[j]) * instrument.values[:, i]
                held[j] = holding[:, i]
    dt = times[-1] - last
    cash = cash * np.exp(r * dt) + position * spot[:, -1] * (np.exp(q * dt) - 1.0)
    final = cash + position * spot[:, -1] - payoff
    for j, (instrument, _) in enumerate(others):
        final += held[j] * instrument.values[:, -1]
    return np.asarray(final * np.exp(-r * times[-1]), dtype=np.float64)


# ---------------------------------------------------------------- Black-Scholes worlds


def gbm_paths(t: float, r: float, q: float, sigma: float, steps: int, paths: int, seed: int = 1) -> tuple[Array, Array]:
    """Spot (from 1) on ``steps + 1`` equally spaced dates, exact log-normal steps, antithetic."""
    rng = np.random.default_rng(seed)
    dt = t / steps
    z = rng.standard_normal((paths // 2, steps))
    z = np.vstack([z, -z])
    log = np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    spot = np.column_stack([np.ones(z.shape[0]), np.exp(log)])
    return np.asarray(spot, dtype=np.float64), np.linspace(0.0, t, steps + 1)


def bs_values(
    spot: Array, times: Array, strike: float, expiry: float, r: float, q: float, sigma: float | Array
) -> tuple[Array, Array, Array]:
    """Value, delta and gamma of a call on every path and date (``times`` from today; ``expiry`` may be later)."""
    tau = np.maximum(expiry - times, 0.0)[None, :]
    g = greeks(spot, strike, tau, r, q, sigma, True)
    return g.price, g.delta, g.gamma


def black_scholes_study(
    sigma: float = 0.2,
    strike: float = 1.0,
    expiry: float = 0.25,
    r: float = 0.04,
    q: float = 0.01,
    steps: int = 504,
    paths: int = 20_000,
    rebalances: Sequence[int] = (8, 16, 32, 63, 126, 252, 504),
    hedge_expiry: float = 0.5,
    seed: int = 7,
    show: int = 63,
) -> BlackScholesStudy:
    """Delta and delta-gamma hedging of a call in its own Black-Scholes world, against rebalancing frequency.

    The gamma hedge is a call of the same strike and a longer expiry, ``hedge_expiry``. The
    P&L of both hedges is kept for ``show`` rebalances.
    """
    spot, times = gbm_paths(expiry, r, q, sigma, steps, paths, seed)
    value, delta, gamma = bs_values(spot, times, strike, expiry, r, q, sigma)
    h_value, h_delta, h_gamma = bs_values(spot, times, strike, hedge_expiry, r, q, sigma)
    payoff = np.maximum(spot[:, -1] - strike, 0.0)
    premium = float(value[0, 0])
    vega = float(greeks(1.0, strike, expiry, r, q, sigma).vega)
    ratio = gamma[:, :-1] / h_gamma[:, :-1]
    option = Instrument("longer call", h_value)
    rows = []
    kept: dict[str, Array] = {}
    for n in rebalances:
        every = steps // n
        plain = hedge(spot, times, r, q, premium, payoff, delta[:, :-1], every=every)
        both = hedge(spot, times, r, q, premium, payoff, delta[:, :-1] - ratio * h_delta[:, :-1], [(option, ratio)], every=every)
        if n == show:
            kept = {"delta": plain, "delta and gamma": both}
        theory = float(np.sqrt(np.pi / 4.0) * vega * sigma / np.sqrt(n)) * np.exp(-r * expiry)
        rows.append(
            {
                "rebalances": n,
                "delta": float(plain.std(ddof=1)),
                "delta_gamma": float(both.std(ddof=1)),
                "derman_kamal": theory,
                "premium": premium,
            }
        )
    return BlackScholesStudy(pd.DataFrame(rows), kept, show)


@dataclass(frozen=True)
class BlackScholesStudy:
    table: pd.DataFrame  # rebalances, delta, delta_gamma, derman_kamal, premium
    pnl: dict[str, Array]  # each hedge's P&L at ``shown`` rebalances
    shown: int


def gamma_pnl(spot: Array, times: Array, strike: float, expiry: float, r: float, q: float, sigma_hedge: float) -> Array:
    """``1/2 sum e^{-r t_i} Gamma_i S_i^2 [sigma_h^2 dt - (dS_i/S_i)^2]``: a short option's P&L "
    "hedged at ``sigma_hedge``, by path."""
    _, _, gamma = bs_values(spot[:, :-1], times[:-1], strike, expiry, r, q, sigma_hedge)
    ret = np.diff(spot, axis=1) / spot[:, :-1]
    carry = (r - q) * np.diff(times)[None, :]
    realised = (ret - carry) ** 2  # the drift of the hedged underlying is financed; only its variance is left
    target = sigma_hedge**2 * np.diff(times)[None, :]
    weight = 0.5 * np.exp(-r * times[:-1])[None, :] * gamma * spot[:, :-1] ** 2
    return np.asarray(np.sum(weight * (target - realised), axis=1), dtype=np.float64)


@dataclass(frozen=True)
class WrongVolatility:
    spot: Array
    times: Array
    pnl: Array
    predicted: Array
    realised_vol: Array
    premium: float


def wrong_volatility(
    sigma_true: float = 0.25,
    sigma_hedge: float = 0.2,
    strike: float = 1.0,
    expiry: float = 0.25,
    r: float = 0.04,
    q: float = 0.01,
    steps: int = 252,
    paths: int = 20_000,
    seed: int = 11,
) -> WrongVolatility:
    """Sell at ``sigma_hedge`` and delta-hedge at it, in a world whose volatility is ``sigma_true``."""
    spot, times = gbm_paths(expiry, r, q, sigma_true, steps, paths, seed)
    value, delta, _ = bs_values(spot, times, strike, expiry, r, q, sigma_hedge)
    payoff = np.maximum(spot[:, -1] - strike, 0.0)
    premium = float(value[0, 0])
    pnl = hedge(spot, times, r, q, premium, payoff, delta[:, :-1])
    realised = np.sqrt(np.sum(np.diff(np.log(spot), axis=1) ** 2, axis=1) / expiry)
    return WrongVolatility(spot, times, pnl, gamma_pnl(spot, times, strike, expiry, r, q, sigma_hedge), realised, premium)
