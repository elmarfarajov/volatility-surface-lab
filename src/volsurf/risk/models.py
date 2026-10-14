"""Hedging when volatility is not constant: Heston's world, the S&P 500's surface, and hedges from the wrong model.

**Heston's world** has two sources of risk, the spot and its variance, and a delta hedge
reaches only one. However often it is rebalanced, the P&L keeps the variance risk: the
error stops falling. Three deltas are compared, and one hedge with a second option:

* Black-Scholes delta at the option's implied volatility on the day it was sold (the
  trader's sticky-strike delta);
* Heston's delta, ``dC/dS`` with the variance held;
* the minimum-variance delta, ``dC/dS + (rho sigma / S) dC/dv``, which also hedges the
  part of the variance move that is correlated with the spot (Bakshi, Cao and Chen, 1997);
* delta and vega: a longer option of the same strike cancels ``dC/dv``, and the stock what
  is left of the delta. The market is complete again, and the error falls with frequency.

**The S&P 500.** A three-month at-the-money call is sold at the market's price and
hedged in two worlds fitted to the same surface: Heston as calibrated on Day 5, and
Dupire local volatility from the Day 4 surface. In each world it is hedged with
Black-Scholes, local-volatility and Heston deltas. Hedging in a world with the other
model's delta is model risk made concrete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..analytic.generalized import greeks
from ..exotics.monte_carlo import heston_paths_with_variance, local_vol_paths
from ..market.implied import implied_vol
from ..smile.interpolation import Surface
from ..stochvol.characteristic import Heston
from .hedging import Instrument, hedge
from .pricers import HestonGrid, LocalVolGrid, minimum_variance_delta

Array = NDArray[np.float64]


@dataclass(frozen=True)
class HestonWorld:
    spot: Array  # [paths, dates], from 1
    variance: Array
    times: Array


def heston_world(h: Heston, expiry: float, b: float, steps: int, paths: int, seed: int = 1) -> HestonWorld:
    s, v = heston_paths_with_variance(1.0, expiry, b, h, steps, paths, seed)
    spot = np.column_stack([np.ones(paths), s])
    variance = np.column_stack([np.full(paths, h.v0), v])
    return HestonWorld(spot, variance, np.linspace(0.0, expiry, steps + 1))


def _grid_values(grid: HestonGrid, world: HestonWorld, dates: int) -> tuple[Array, Array, Array]:
    """Value, delta and dV/dv from a Heston grid on the first ``dates`` dates of the world."""
    shape = (world.spot.shape[0], dates)
    value, delta, vega = np.empty(shape), np.empty(shape), np.empty(shape)
    for i in range(dates):
        val = grid.at(i, world.spot[:, i], world.variance[:, i])
        value[:, i], delta[:, i], vega[:, i] = val.value, val.delta, val.vega_v
    return value, delta, vega


def heston_study(
    h: Heston,
    strike: float = 1.0,
    expiry: float = 0.25,
    r: float = 0.04,
    q: float = 0.01,
    steps: int = 64,
    paths: int = 20_000,
    rebalances: Sequence[int] = (2, 4, 8, 16, 32, 64),
    hedge_expiry: float = 0.5,
    seed: int = 5,
) -> HestonStudy:
    """Hedging error (standard deviation of discounted P&L) of each strategy against rebalancing frequency."""
    b = r - q
    world = heston_world(h, expiry, b, steps, paths, seed)
    v_max = float(np.quantile(world.variance, 0.999)) * 1.2
    target = HestonGrid.build(h, strike, expiry, r, b, world.times[:-1], v_max=v_max)
    longer = HestonGrid.build(h, strike, hedge_expiry, r, b, world.times, v_max=v_max)
    value, delta, vega = _grid_values(target, world, steps)
    h_value, h_delta, h_vega = _grid_values(longer, world, steps + 1)
    premium = float(value[0, 0])
    payoff = np.maximum(world.spot[:, -1] - strike, 0.0)
    implied = float(
        implied_vol(np.array([premium]), np.exp(b * expiry), np.array([strike]), expiry, np.exp(-r * expiry), True)[0]
    )
    bs = greeks(world.spot[:, :-1], strike, (expiry - world.times[:-1])[None, :], r, q, implied).delta
    mv = minimum_variance_delta(h, world.spot[:, :-1], delta, vega)
    ratio = vega / h_vega[:, :-1]
    option = Instrument("longer call", h_value)
    strategies = {
        "Black-Scholes delta at the implied volatility": (bs, []),
        "Heston delta": (delta, []),
        "minimum-variance delta": (mv, []),
        "delta and vega, with a longer call": (delta - ratio * h_delta[:, :-1], [(option, ratio)]),
    }
    rows = []
    kept: dict[str, Array] = {}
    for n in rebalances:
        for name, (stock, others) in strategies.items():
            pnl = hedge(world.spot, world.times, r, q, premium, payoff, stock, others, every=steps // n)
            rows.append(
                {"rebalances": n, "strategy": name, "std": float(pnl.std(ddof=1)), "mean": float(pnl.mean()), "premium": premium}
            )
            if n == max(rebalances):
                kept[name] = pnl
    return HestonStudy(pd.DataFrame(rows), kept, implied)


@dataclass(frozen=True)
class HestonStudy:
    table: pd.DataFrame  # rebalances, strategy, std, mean, premium
    pnl: dict[str, Array]  # each strategy's P&L at the most frequent rebalancing
    implied: float  # the option's Black-Scholes implied volatility when sold


# ---------------------------------------------------------------- the S&P 500


@dataclass(frozen=True)
class CrossModel:
    table: pd.DataFrame  # world, strategy, mean, std, std_pct_premium, es_05
    premium: float
    implied: float
    deltas: dict[str, float]  # each model's delta at inception
    pnl: dict[tuple[str, str], Array]


def cross_model(
    surface: Surface,
    heston: Heston,
    expiry: float,
    r: float,
    b: float,
    strike: float = 1.0,
    steps: int = 63,
    paths: int = 20_000,
    seed: int = 9,
) -> CrossModel:
    """A call sold at the surface's price, hedged daily in a Heston and a local-volatility world with each model's delta.

    Spot and strike are in units of today's spot; the surface's log-moneyness is relative
    to the forward.
    """
    q = r - b
    k = float(np.log(strike) - b * expiry)
    implied = float(surface.implied_vol(np.array([k]), expiry)[0])
    premium = float(greeks(1.0, strike, expiry, r, q, implied).price)
    times = np.linspace(0.0, expiry, steps + 1)
    lv = LocalVolGrid.build(surface, strike, expiry, r, b)
    worlds: dict[str, tuple[Array, Array | None]] = {}
    hw = heston_world(heston, expiry, b, steps, paths, seed)
    worlds["Heston"] = (hw.spot, hw.variance)
    s = local_vol_paths(1.0, expiry, r, b, surface, steps, paths, seed)
    worlds["local volatility"] = (np.column_stack([np.ones(paths), s]), None)
    v_max = float(np.quantile(hw.variance, 0.999)) * 1.2
    grid = HestonGrid.build(heston, strike, expiry, r, b, times[:-1], v_max=v_max)
    rows, pnls = [], {}
    deltas = {
        "Black-Scholes": float(greeks(1.0, strike, expiry, r, q, implied).delta),
        "local volatility": float(lv.at(0.0, np.array([1.0])).delta[0]),
    }
    v0 = grid.at(0, np.array([1.0]), np.array([heston.v0]))
    deltas["Heston"] = float(v0.delta[0])
    deltas["Heston, minimum variance"] = float(minimum_variance_delta(heston, np.array([1.0]), v0.delta, v0.vega_v)[0])
    for world, (spot, variance) in worlds.items():
        payoff = np.maximum(spot[:, -1] - strike, 0.0)
        stock = {
            "Black-Scholes delta (sticky strike)": greeks(
                spot[:, :-1], strike, (expiry - times[:-1])[None, :], r, q, implied
            ).delta,
            "local-volatility delta": np.column_stack([lv.at(float(t), spot[:, i]).delta for i, t in enumerate(times[:-1])]),
        }
        if variance is not None:
            hd = np.empty_like(spot[:, :-1])
            mv = np.empty_like(hd)
            for i in range(steps):
                val = grid.at(i, spot[:, i], variance[:, i])
                hd[:, i] = val.delta
                mv[:, i] = minimum_variance_delta(heston, spot[:, i], val.delta, val.vega_v)
            stock["Heston delta"] = hd
            stock["Heston minimum-variance delta"] = mv
        for name, held in stock.items():
            pnl = hedge(spot, times, r, q, premium, payoff, held)
            pnls[(world, name)] = pnl
            stats = {"mean": float(pnl.mean()), "std": float(pnl.std(ddof=1))}
            tail = np.sort(pnl)[: max(1, int(0.05 * pnl.size))]
            rows.append(
                {
                    "world": world,
                    "strategy": name,
                    **stats,
                    "std_pct_premium": 100 * stats["std"] / premium,
                    "es_05": float(tail.mean()),
                }
            )
    return CrossModel(pd.DataFrame(rows), premium, implied, deltas, pnls)
