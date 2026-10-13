"""Closed forms for exotic options under Black-Scholes-Merton with cost of carry ``b``.

Every formula is written for a spot ``S``, strike ``X``, maturity ``T``, rate ``r``, carry ``b``
(``b = r - q`` for a stock paying a yield ``q``) and volatility ``sigma``, as in Haug's *Complete
Guide* (2007), and each is checked against QuantLib's engines in the tests:

* **digitals**: cash-or-nothing and asset-or-nothing;
* **barriers** (Reiner and Rubinstein, 1991): the eight single-barrier options, knocked in or out
  from above or below, with a rebate paid at expiry (knock-ins) or at the hit (knock-outs);
* **geometric Asians** (Kemna and Vorst, 1990): the continuous average, and the discrete one
  over ``n`` equally spaced fixings;
* **lookbacks**: floating strike (Goldman, Sosin and Gatto, 1979) and fixed strike (Conze and
  Viswanathan, 1991).

All monitoring here is continuous. Discrete monitoring is the subject of
:mod:`~volsurf.exotics.monte_carlo`, with the Broadie-Glasserman-Kou correction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import ndtr

BarrierType = Literal["down-in", "down-out", "up-in", "up-out"]


def _n(x: float) -> float:
    return float(ndtr(x))


def black_scholes(s: float, x: float, t: float, r: float, b: float, sigma: float, call: bool = True) -> float:
    d1 = (np.log(s / x) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if call:
        return float(s * np.exp((b - r) * t) * _n(d1) - x * np.exp(-r * t) * _n(d2))
    return float(x * np.exp(-r * t) * _n(-d2) - s * np.exp((b - r) * t) * _n(-d1))


def cash_or_nothing(
    s: float, x: float, t: float, r: float, b: float, sigma: float, cash: float = 1.0, call: bool = True
) -> float:
    d2 = (np.log(s / x) + (b - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return float(cash * np.exp(-r * t) * _n(d2 if call else -d2))


def asset_or_nothing(s: float, x: float, t: float, r: float, b: float, sigma: float, call: bool = True) -> float:
    d1 = (np.log(s / x) + (b + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return float(s * np.exp((b - r) * t) * _n(d1 if call else -d1))


def barrier(
    kind: BarrierType,
    s: float,
    x: float,
    h: float,
    t: float,
    r: float,
    b: float,
    sigma: float,
    call: bool = True,
    rebate: float = 0.0,
) -> float:
    """Reiner-Rubinstein's single barrier option, continuously monitored (Haug, 2007, section 4.17.1)."""
    down = kind.startswith("down")
    if (down and s <= h) or (not down and s >= h):
        raise ValueError("the barrier has already been reached")
    eta = 1.0 if down else -1.0
    phi = 1.0 if call else -1.0
    vol_t = sigma * np.sqrt(t)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = np.sqrt(mu**2 + 2.0 * r / sigma**2)
    carry = s * np.exp((b - r) * t)
    disc = np.exp(-r * t)
    x1 = np.log(s / x) / vol_t + (1 + mu) * vol_t
    x2 = np.log(s / h) / vol_t + (1 + mu) * vol_t
    y1 = np.log(h * h / (s * x)) / vol_t + (1 + mu) * vol_t
    y2 = np.log(h / s) / vol_t + (1 + mu) * vol_t
    z = np.log(h / s) / vol_t + lam * vol_t
    hs = h / s
    a_term = phi * carry * _n(phi * x1) - phi * x * disc * _n(phi * x1 - phi * vol_t)
    b_term = phi * carry * _n(phi * x2) - phi * x * disc * _n(phi * x2 - phi * vol_t)
    c_term = phi * carry * hs ** (2 * (mu + 1)) * _n(eta * y1) - phi * x * disc * hs ** (2 * mu) * _n(eta * y1 - eta * vol_t)
    d_term = phi * carry * hs ** (2 * (mu + 1)) * _n(eta * y2) - phi * x * disc * hs ** (2 * mu) * _n(eta * y2 - eta * vol_t)
    e_term = rebate * disc * (_n(eta * x2 - eta * vol_t) - hs ** (2 * mu) * _n(eta * y2 - eta * vol_t))
    f_term = rebate * (hs ** (mu + lam) * _n(eta * z) + hs ** (mu - lam) * _n(eta * z - 2 * eta * lam * vol_t))
    above = x > h
    table = {
        ("down-in", True): (c_term + e_term) if above else (a_term - b_term + d_term + e_term),
        ("up-in", True): (a_term + e_term) if above else (b_term - c_term + d_term + e_term),
        ("down-in", False): (b_term - c_term + d_term + e_term) if above else (a_term + e_term),
        ("up-in", False): (a_term - b_term + d_term + e_term) if above else (c_term + e_term),
        ("down-out", True): (a_term - c_term + f_term) if above else (b_term - d_term + f_term),
        ("up-out", True): f_term if above else (a_term - b_term + c_term - d_term + f_term),
        ("down-out", False): (a_term - b_term + c_term - d_term + f_term) if above else f_term,
        ("up-out", False): (b_term - d_term + f_term) if above else (a_term - c_term + f_term),
    }
    return float(table[(kind, call)])


def geometric_asian(
    s: float, x: float, t: float, r: float, b: float, sigma: float, call: bool = True, fixings: int | None = None
) -> float:
    """Geometric-average-price option: continuous (Kemna and Vorst) or over ``fixings`` equally spaced dates ending at ``t``.

    ``ln G`` is normal; with ``n`` fixings at ``iT/n`` its mean is ``ln S + (b - sigma^2/2) T (n + 1)/(2n)`` and its
    variance ``sigma^2 T (n + 1)(2n + 1)/(6 n^2)``; as ``n -> infinity`` these become ``(b - sigma^2/2) T/2`` and ``sigma^2 T/3``.
    """
    if fixings is None:
        mean = np.log(s) + (b - 0.5 * sigma**2) * t / 2.0
        var = sigma**2 * t / 3.0
    else:
        n = fixings
        mean = np.log(s) + (b - 0.5 * sigma**2) * t * (n + 1) / (2.0 * n)
        var = sigma**2 * t * (n + 1) * (2 * n + 1) / (6.0 * n * n)
    forward = np.exp(mean + 0.5 * var)
    sd = np.sqrt(var)
    d1 = (np.log(forward / x) + 0.5 * var) / sd
    d2 = d1 - sd
    disc = np.exp(-r * t)
    if call:
        return float(disc * (forward * _n(d1) - x * _n(d2)))
    return float(disc * (x * _n(-d2) - forward * _n(-d1)))


def floating_lookback(s: float, extreme: float, t: float, r: float, b: float, sigma: float, call: bool = True) -> float:
    """Floating-strike lookback: the call pays ``S_T - min S``, the put ``max S - S_T`` (Goldman, Sosin and Gatto, 1979).

    ``extreme`` is the minimum (call) or maximum (put) observed so far. Requires ``b != 0``.
    """
    vol_t = sigma * np.sqrt(t)
    k = 2.0 * b / sigma**2
    disc = np.exp(-r * t)
    carry = s * np.exp((b - r) * t)
    a1 = (np.log(s / extreme) + (b + 0.5 * sigma**2) * t) / vol_t
    a2 = a1 - vol_t
    if call:
        term = (s / extreme) ** (-k) * _n(-a1 + k * np.sqrt(t) * sigma) - np.exp(b * t) * _n(-a1)
        return float(carry * _n(a1) - extreme * disc * _n(a2) + s * disc * sigma**2 / (2 * b) * term)
    term = -((s / extreme) ** (-k)) * _n(a1 - k * np.sqrt(t) * sigma) + np.exp(b * t) * _n(a1)
    return float(extreme * disc * _n(-a2) - carry * _n(-a1) + s * disc * sigma**2 / (2 * b) * term)


def fixed_lookback(s: float, x: float, extreme: float, t: float, r: float, b: float, sigma: float, call: bool = True) -> float:
    """Fixed-strike lookback: the call pays ``(max S - X)^+``, the put ``(X - min S)^+`` (Conze and Viswanathan, 1991).

    ``extreme`` is the maximum (call) or minimum (put) observed so far. Requires ``b != 0``.
    """
    vol_t = sigma * np.sqrt(t)
    k = 2.0 * b / sigma**2
    disc = np.exp(-r * t)
    carry = s * np.exp((b - r) * t)
    if call:
        level = max(x, extreme)
        d1 = (np.log(s / level) + (b + 0.5 * sigma**2) * t) / vol_t
        d2 = d1 - vol_t
        term = -((s / level) ** (-k)) * _n(d1 - k * np.sqrt(t) * sigma) + np.exp(b * t) * _n(d1)
        intrinsic = disc * max(extreme - x, 0.0)
        return float(intrinsic + carry * _n(d1) - level * disc * _n(d2) + s * disc * sigma**2 / (2 * b) * term)
    level = min(x, extreme)
    d1 = (np.log(s / level) + (b + 0.5 * sigma**2) * t) / vol_t
    d2 = d1 - vol_t
    term = (s / level) ** (-k) * _n(-d1 + k * np.sqrt(t) * sigma) - np.exp(b * t) * _n(-d1)
    intrinsic = disc * max(x - extreme, 0.0)
    return float(intrinsic - carry * _n(-d1) + level * disc * _n(-d2) + s * disc * sigma**2 / (2 * b) * term)
