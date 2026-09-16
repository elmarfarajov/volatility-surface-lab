"""Recombining binomial trees for European and American options.

Two lattices are provided:

* Cox-Ross-Rubinstein (1979): the textbook tree, first-order convergent with the
  well-known odd/even oscillation.
* Leisen-Reimer (1996): node probabilities come from the Peizer-Pratt inversion of the
  normal CDF, centring the tree on the strike. Convergence for European options is
  second order and non-oscillating, so ~200 steps beat thousands of CRR steps.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal["leisen-reimer", "crr"]


def _peizer_pratt(z: float, n: int) -> float:
    exponent = -((z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2) * (n + 1.0 / 6.0)
    return 0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(exponent))


def binomial_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    is_call: bool = True,
    american: bool = False,
    steps: int = 501,
    method: Method = "leisen-reimer",
) -> float:
    if maturity <= 0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

    if method == "leisen-reimer":
        steps = steps if steps % 2 == 1 else steps + 1
    dt = maturity / steps
    growth = np.exp((rate - dividend_yield) * dt)
    discount = np.exp(-rate * dt)

    if method == "crr":
        up = np.exp(vol * np.sqrt(dt))
        down = 1.0 / up
        p = (growth - down) / (up - down)
    elif method == "leisen-reimer":
        sqrt_t = vol * np.sqrt(maturity)
        d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * maturity) / sqrt_t
        d2 = d1 - sqrt_t
        p = _peizer_pratt(d2, steps)
        p_bar = _peizer_pratt(d1, steps)
        up = growth * p_bar / p
        down = (growth - p * up) / (1.0 - p)
    else:
        raise ValueError(f"Unknown lattice method: {method}")

    if not 0.0 < p < 1.0:
        raise ValueError("Risk-neutral probability outside (0, 1); increase the number of steps")

    log_up, log_down = np.log(up), np.log(down)
    sign = 1.0 if is_call else -1.0

    j = np.arange(steps + 1)
    terminal = spot * np.exp(j * log_up + (steps - j) * log_down)
    values = np.maximum(sign * (terminal - strike), 0.0)

    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(i + 1)
            node_spot = spot * np.exp(j * log_up + (i - j) * log_down)
            values = np.maximum(values, sign * (node_spot - strike))
    return float(values[0])
