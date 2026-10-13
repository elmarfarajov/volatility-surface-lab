"""The price of an exotic under each model of the same surface.

Black-Scholes at the at-the-money volatility, Heston calibrated on Day 5 and Dupire local
volatility from the Day 4 surface agree, more or less, on vanilla options - local volatility
exactly, by construction; Heston only to its calibration error. On path-dependent options
they need not agree at all: the smile's dynamics, which vanillas do not see, decide the price
of a barrier. The spread across models is a measure of model risk that no amount of
calibration to vanillas removes.

Every price here is in units of the spot, for a one-year maturity on the S&P 500, under the
forward measure with the market's carry; paths are monitored daily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..smile.interpolation import Surface
from ..stochvol.characteristic import Heston
from .monte_carlo import Estimate, black_scholes_paths, heston_paths, local_vol_paths, price

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Contract:
    name: str
    payoff: Callable[[Array], Array]  # paths in units of today's spot (columns: daily monitoring dates)


def contracts(strike: float = 1.0) -> list[Contract]:
    """The book: vanillas to check calibration, then exotics that depend on the path or the smile."""

    def vanilla(k: float, call: bool) -> Callable[[Array], Array]:
        return lambda p: np.maximum(p[:, -1] - k, 0.0) if call else np.maximum(k - p[:, -1], 0.0)

    def down_out_call(p: Array) -> Array:
        return np.where(p.min(axis=1) > 0.8, np.maximum(p[:, -1] - strike, 0.0), 0.0)

    def up_out_call(p: Array) -> Array:
        return np.where(p.max(axis=1) < 1.2, np.maximum(p[:, -1] - strike, 0.0), 0.0)

    def one_touch_down(p: Array) -> Array:
        return np.asarray((p.min(axis=1) <= 0.85).astype(np.float64))

    def asian(p: Array) -> Array:
        monthly = p[:, 20::21]  # about one fixing a month
        return np.maximum(monthly.mean(axis=1) - strike, 0.0)

    def digital(p: Array) -> Array:
        return np.asarray((p[:, -1] > 1.05).astype(np.float64))

    return [
        Contract("call at the money", vanilla(1.0, True)),
        Contract("put at 90%", vanilla(0.9, False)),
        Contract("digital call at 105%", digital),
        Contract("down-and-out call, barrier 80%", down_out_call),
        Contract("up-and-out call, barrier 120%", up_out_call),
        Contract("one-touch at 85%", one_touch_down),
        Contract("Asian call, monthly fixings", asian),
    ]


@dataclass(frozen=True)
class Book:
    models: tuple[str, ...]
    contracts: tuple[str, ...]
    prices: dict[tuple[str, str], Estimate]


def price_book(
    t: float,
    r: float,
    b: float,
    atm_vol: float,
    heston: Heston,
    surface: Surface,
    paths: int = 100_000,
    steps: int = 252,
    seed: int = 3,
) -> Book:
    """Each contract under Black-Scholes at ``atm_vol``, Heston and local volatility, on common random numbers where possible."""
    generators: dict[str, Callable[[], Array]] = {
        "Black-Scholes": lambda: black_scholes_paths(1.0, t, r, b, atm_vol, steps, paths, seed),
        "Heston": lambda: heston_paths(1.0, t, r, b, heston, steps, paths, seed),
        "local volatility": lambda: local_vol_paths(1.0, t, r, b, surface, steps, paths, seed),
    }
    book = contracts()
    prices: dict[tuple[str, str], Estimate] = {}
    for model, generate in generators.items():
        p = generate()
        for c in book:
            prices[(model, c.name)] = price(p, c.payoff, float(np.exp(-r * t)))
    return Book(tuple(generators), tuple(c.name for c in book), prices)
