"""Recombining lattices for European, Bermudan and American options, and how fast each converges.

A binomial tree replaces the log-normal distribution of the underlying over each
step ``dt`` by two outcomes, ``S u`` and ``S d``, with probability ``p`` of the up move.
The parameterisations differ in how they match the distribution:

======================  ==========================================  ====================================
lattice                 choice                                      European convergence
======================  ==========================================  ====================================
Cox-Ross-Rubinstein     u = e^{sigma sqrt dt}, d = 1/u              first order, oscillating with n
Jarrow-Rudd             equal probabilities, drift in the moves     first order, oscillating
Tian                    third moment matched                        first order, oscillating
Leisen-Reimer           Peizer-Pratt inversion, centred on strike   second order, smooth (odd n)
trinomial               three outcomes, dx = sigma sqrt(3 dt)       first order, smoother
======================  ==========================================  ====================================

The oscillation comes from where the strike falls between nodes at expiry, which
changes with every step added. Two remedies apply to any lattice:

* **BBS** (Broadie and Detemple, 1996) replaces the last step by the Black-Scholes
  value, which smooths the payoff's kink;
* **Richardson extrapolation** of BBS at ``n`` and ``2n`` steps (BBSR) cancels the
  leading error term: ``2 V(2n) - V(n)``.

Exercise is European (at expiry), American (at every node) or Bermudan (at the
nodes nearest a list of dates). Delta and gamma are read from the first two
levels of the tree, at no extra cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..analytic import greeks

Array = NDArray[np.float64]
Lattice = Literal["crr", "jarrow-rudd", "tian", "leisen-reimer", "trinomial"]
LATTICES: tuple[Lattice, ...] = ("crr", "jarrow-rudd", "tian", "leisen-reimer", "trinomial")


@dataclass(frozen=True)
class Contract:
    """A vanilla option on an underlying with a continuous yield."""

    spot: float
    strike: float
    maturity: float
    rate: float
    yield_: float
    vol: float
    is_call: bool = True

    def payoff(self, spot: Array) -> Array:
        sign = 1.0 if self.is_call else -1.0
        return np.asarray(np.maximum(sign * (spot - self.strike), 0.0), dtype=np.float64)

    def european(self) -> float:
        return float(greeks(self.spot, self.strike, self.maturity, self.rate, self.yield_, self.vol, self.is_call).price)


@dataclass(frozen=True)
class LatticeResult:
    price: float
    delta: float
    gamma: float
    steps: int


def _peizer_pratt(z: float, n: int) -> float:
    exponent = -((z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2) * (n + 1.0 / 6.0)
    return float(0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(exponent)))


def _binomial_moves(c: Contract, steps: int, lattice: Lattice) -> tuple[float, float, float]:
    """(up factor, down factor, up probability) for one step."""
    dt = c.maturity / steps
    growth = np.exp((c.rate - c.yield_) * dt)
    if lattice == "crr":
        up = float(np.exp(c.vol * np.sqrt(dt)))
        down = 1.0 / up
        p = (growth - down) / (up - down)
    elif lattice == "jarrow-rudd":
        drift = (c.rate - c.yield_ - 0.5 * c.vol**2) * dt
        up, down = float(np.exp(drift + c.vol * np.sqrt(dt))), float(np.exp(drift - c.vol * np.sqrt(dt)))
        p = (growth - down) / (up - down)  # the risk-neutral p, close to one half
    elif lattice == "tian":
        v = float(np.exp(c.vol**2 * dt))
        up = 0.5 * growth * v * (v + 1.0 + np.sqrt(v * v + 2.0 * v - 3.0))
        down = 0.5 * growth * v * (v + 1.0 - np.sqrt(v * v + 2.0 * v - 3.0))
        p = (growth - down) / (up - down)
    elif lattice == "leisen-reimer":
        total = c.vol * np.sqrt(c.maturity)
        d1 = (np.log(c.spot / c.strike) + (c.rate - c.yield_ + 0.5 * c.vol**2) * c.maturity) / total
        p = _peizer_pratt(d1 - total, steps)
        p_bar = _peizer_pratt(d1, steps)
        up = float(growth * p_bar / p)
        down = float((growth - p * up) / (1.0 - p))
    else:
        raise ValueError(f"no binomial parameterisation for {lattice!r}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"{lattice}: probability {p:.4f} outside (0, 1) - use more steps")
    return float(up), float(down), float(p)


def _exercise_steps(maturity: float, steps: int, exercise: str | Sequence[float]) -> set[int]:
    if exercise == "european":
        return set()
    if exercise == "american":
        return set(range(steps))
    if isinstance(exercise, str):
        raise ValueError(f"exercise must be 'european', 'american' or a list of dates, not {exercise!r}")
    # Bermudan: the step nearest each date, before expiry
    return {int(round(t / maturity * steps)) for t in exercise if 0 < t < maturity}


def price(
    contract: Contract,
    steps: int = 500,
    lattice: Lattice = "leisen-reimer",
    exercise: str | Sequence[float] = "european",
    smoothing: bool = False,
) -> LatticeResult:
    """The lattice price, delta and gamma. ``smoothing`` replaces the last step by Black-Scholes (BBS)."""
    c = contract
    if c.maturity <= 0:
        intrinsic = float(c.payoff(np.array([c.spot]))[0])
        return LatticeResult(intrinsic, float("nan"), float("nan"), 0)
    if lattice == "leisen-reimer" and steps % 2 == 0:
        steps += 1  # the Peizer-Pratt inversion is exact for odd n
    if lattice == "trinomial":
        return _trinomial(c, steps, exercise, smoothing)
    up, down, p = _binomial_moves(c, steps, lattice)
    dt = c.maturity / steps
    discount = float(np.exp(-c.rate * dt))
    log_up, log_down = np.log(up), np.log(down)
    exercisable = _exercise_steps(c.maturity, steps, exercise)

    def nodes(level: int) -> Array:
        j = np.arange(level + 1)
        return np.asarray(c.spot * np.exp(j * log_up + (level - j) * log_down), dtype=np.float64)

    if smoothing:
        last = steps - 1
        spots = nodes(last)
        values = np.asarray(greeks(spots, c.strike, dt, c.rate, c.yield_, c.vol, c.is_call).price, dtype=np.float64)
        if last in exercisable:
            values = np.maximum(values, c.payoff(spots))
        start = last - 1
    else:
        values = c.payoff(nodes(steps))
        start = steps - 1
    level_values: dict[int, Array] = {}
    for level in range(start, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if level in exercisable:
            values = np.maximum(values, c.payoff(nodes(level)))
        if level <= 2:
            level_values[level] = values
    return _greeks_from_levels(c, level_values, nodes, steps)


def _greeks_from_levels(c: Contract, levels: dict[int, Array], nodes: object, steps: int) -> LatticeResult:
    node = nodes  # a callable level -> node spots
    assert callable(node)
    value = float(levels[0][0])
    if 1 not in levels or 2 not in levels:
        return LatticeResult(value, float("nan"), float("nan"), steps)
    s1, v1 = node(1), levels[1]
    s2, v2 = node(2), levels[2]
    delta = float((v1[1] - v1[0]) / (s1[1] - s1[0]))
    up_delta = (v2[2] - v2[1]) / (s2[2] - s2[1])
    down_delta = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = float((up_delta - down_delta) / (0.5 * (s2[2] - s2[0])))
    return LatticeResult(value, delta, gamma, steps)


def _trinomial(c: Contract, steps: int, exercise: str | Sequence[float], smoothing: bool) -> LatticeResult:
    """A trinomial tree in log-spot with dx = sigma sqrt(3 dt) (Boyle's lambda = sqrt 3)."""
    dt = c.maturity / steps
    dx = c.vol * np.sqrt(3.0 * dt)
    nu = c.rate - c.yield_ - 0.5 * c.vol**2
    a = (c.vol**2 * dt + nu**2 * dt**2) / dx**2
    pu, pd = 0.5 * (a + nu * dt / dx), 0.5 * (a - nu * dt / dx)
    pm = 1.0 - pu - pd
    if min(pu, pm, pd) <= 0:
        raise ValueError("trinomial probabilities must be positive - use more steps")
    discount = float(np.exp(-c.rate * dt))
    exercisable = _exercise_steps(c.maturity, steps, exercise)

    def nodes(level: int) -> Array:
        j = np.arange(-level, level + 1)
        return np.asarray(c.spot * np.exp(j * dx), dtype=np.float64)

    if smoothing:
        last = steps - 1
        values = np.asarray(greeks(nodes(last), c.strike, dt, c.rate, c.yield_, c.vol, c.is_call).price, dtype=np.float64)
        if last in exercisable:
            values = np.maximum(values, c.payoff(nodes(last)))
        start = last - 1
    else:
        values = c.payoff(nodes(steps))
        start = steps - 1
    levels: dict[int, Array] = {}
    for level in range(start, -1, -1):
        values = discount * (pu * values[2:] + pm * values[1:-1] + pd * values[:-2])
        if level in exercisable:
            values = np.maximum(values, c.payoff(nodes(level)))
        if level <= 2:
            levels[level] = values
    # read delta and gamma from level 1 (three nodes)
    value = float(levels[0][0])
    s1, v1 = nodes(1), levels[1]
    delta = float((v1[2] - v1[0]) / (s1[2] - s1[0]))
    gamma = float(((v1[2] - v1[1]) / (s1[2] - s1[1]) - (v1[1] - v1[0]) / (s1[1] - s1[0])) / (0.5 * (s1[2] - s1[0])))
    return LatticeResult(value, delta, gamma, steps)


def bbsr(contract: Contract, steps: int = 500, lattice: Lattice = "crr", exercise: str | Sequence[float] = "american") -> float:
    """Broadie-Detemple's BBSR: the smoothed lattice at n and 2n steps, Richardson-extrapolated."""
    coarse = price(contract, steps, lattice, exercise, smoothing=True).price
    fine = price(contract, 2 * steps, lattice, exercise, smoothing=True).price
    return 2.0 * fine - coarse
