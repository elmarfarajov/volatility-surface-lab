"""Generalised Black-Scholes-Merton: the price of a European option and every Greek a desk uses.

With ``phi = +1`` for a call and ``-1`` for a put, total volatility ``s = sigma sqrt(T)``,
``Dq = exp(-qT)``, ``Dr = exp(-rT)`` and

    d1 = (ln(S/K) + (r - q + sigma^2/2) T) / s,    d2 = d1 - s,

the price is ``phi [S Dq N(phi d1) - K Dr N(phi d2)]``. Every Greek below is the
analytic derivative of that formula, written out once and checked in the tests
against arbitrary-precision differentiation (mpmath), QuantLib, and the published
examples in Haug (2007).

**Conventions**, stated once because desks differ:

* vega, vanna, volga, zomma and ultima are per unit of volatility (1.00 = 100 vol
  points), rho and psi per unit of rate - divide by 100 for "per point";
* the time Greeks - theta, charm, color, veta - are derivatives with respect to
  *calendar time passing* (``-d/dT``), per year; divide by 365 for "per day";
* rho moves the discount rate with the yield held fixed (a stock's dividend yield,
  a currency's foreign rate); psi moves the yield. For a futures option, whose
  forward does not depend on the rate, the rate sensitivity is ``rho + psi = -T V``,
  given as ``rho_forward_held``.

**At expiry or zero volatility** (``s = 0``) the option is its discounted intrinsic
value on the forward. Every Greek is then its limit: delta is a step, the
curvature Greeks are zero, and theta is the decay of the discounted intrinsic.
Exactly at the money the limits do not exist; delta is the average of its two
sides and the curvature Greeks are NaN, which is the honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

Array = NDArray[np.float64]
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _pdf(x: Array) -> Array:
    return np.asarray(np.exp(-0.5 * x * x) / _SQRT_2PI, dtype=np.float64)


def _cdf(x: Array) -> Array:
    return np.asarray(ndtr(x), dtype=np.float64)


@dataclass(frozen=True)
class Inputs:
    """The six inputs and the option type, broadcast to one shape."""

    spot: Array
    strike: Array
    maturity: Array
    rate: Array
    yield_: Array
    vol: Array
    phi: Array  # +1 call, -1 put

    @classmethod
    def of(
        cls,
        spot: ArrayLike,
        strike: ArrayLike,
        maturity: ArrayLike,
        rate: ArrayLike,
        yield_: ArrayLike,
        vol: ArrayLike,
        is_call: ArrayLike = True,
    ) -> Inputs:
        values = np.broadcast_arrays(
            *(np.asarray(v, dtype=np.float64) for v in (spot, strike, maturity, rate, yield_, vol)),
            np.asarray(is_call, dtype=bool),
        )
        *numbers, call = values
        if np.any(numbers[0] <= 0) or np.any(numbers[1] <= 0):
            raise ValueError("spot and strike must be positive")
        if np.any(numbers[2] < 0) or np.any(numbers[5] < 0):
            raise ValueError("maturity and volatility must not be negative")
        spot_, strike_, maturity_, rate_, yield__, vol_ = (np.array(v, dtype=np.float64) for v in numbers)
        phi = np.where(call, 1.0, -1.0).astype(np.float64)
        return cls(spot_, strike_, maturity_, rate_, yield__, vol_, phi)


@dataclass(frozen=True)
class Greeks:
    """A price and its sensitivities, each an array of the inputs' shape. See the module for units."""

    price: Array
    # first order
    delta: Array  # dV/dS
    vega: Array  # dV/dsigma
    theta: Array  # dV/dt, calendar time, per year
    rho: Array  # dV/dr, yield held
    psi: Array  # dV/dq
    dual_delta: Array  # dV/dK
    # second order
    gamma: Array  # d2V/dS2
    vanna: Array  # d2V/dS dsigma
    volga: Array  # d2V/dsigma2 (vomma)
    charm: Array  # d delta / dt
    veta: Array  # d vega / dt
    dual_gamma: Array  # d2V/dK2: the risk-neutral density, discounted
    # third order
    speed: Array  # d gamma / dS
    zomma: Array  # d gamma / dsigma
    color: Array  # d gamma / dt
    ultima: Array  # d volga / dsigma

    @property
    def rho_forward_held(self) -> Array:
        """The rate sensitivity when the forward does not move with the rate (a futures option): -T V."""
        return np.asarray(self.rho + self.psi, dtype=np.float64)

    def per_day(self, name: str, days_per_year: float = 365.0) -> Array:
        return np.asarray(getattr(self, name) / days_per_year, dtype=np.float64)

    def as_dict(self) -> dict[str, Array]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _intrinsic(x: Inputs) -> tuple[Array, Array, Array, Array]:
    """Discounted intrinsic value on the forward, its spot and calendar-time derivatives, and the exercise weight."""
    dq, dr = np.exp(-x.yield_ * x.maturity), np.exp(-x.rate * x.maturity)
    forward_leg, strike_leg = x.spot * dq, x.strike * dr
    exercise = x.phi * (forward_leg - strike_leg)
    money = np.sign(exercise)
    weight = np.where(money > 0, 1.0, np.where(money < 0, 0.0, 0.5))  # exactly at the money: half of each side
    # "+ 0.0" turns the negative zeros of an out-of-the-money put into plain zeros
    price = weight * exercise + 0.0
    delta = weight * x.phi * dq + 0.0
    # d/dt = -d/dT of phi (S e^{-qT} - K e^{-rT})
    theta = weight * x.phi * (x.yield_ * forward_leg - x.rate * strike_leg) + 0.0
    return price, delta, theta, weight


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    yield_: ArrayLike,
    vol: ArrayLike,
    is_call: ArrayLike = True,
) -> Greeks:
    """The price and every Greek of a European option under generalised Black-Scholes-Merton."""
    x = Inputs.of(spot, strike, maturity, rate, yield_, vol, is_call)
    S, K, T, r, q, sigma, phi = x.spot, x.strike, x.maturity, x.rate, x.yield_, x.vol, x.phi
    sqrt_t = np.sqrt(T)
    s = sigma * sqrt_t
    live = s > 0
    # the formulas below are evaluated on safe values everywhere and replaced by limits where s = 0
    s_ = np.where(live, s, 1.0)
    T_ = np.where(live, T, 1.0)
    sqrt_t_ = np.where(live, sqrt_t, 1.0)
    sigma_ = np.where(live, sigma, 1.0)
    b = r - q
    d1 = (np.log(S / K) + (b + 0.5 * sigma_ * sigma_) * T_) / s_
    d2 = d1 - s_
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    n1, n2 = _pdf(d1), _pdf(d2)
    nd1, nd2 = _cdf(phi * d1), _cdf(phi * d2)

    price = phi * (S * dq * nd1 - K * dr * nd2)
    delta = phi * dq * nd1
    vega = S * dq * n1 * sqrt_t_
    theta = -S * dq * n1 * sigma_ / (2.0 * sqrt_t_) + phi * (q * S * dq * nd1 - r * K * dr * nd2)
    rho = phi * T_ * K * dr * nd2
    psi = -phi * T_ * S * dq * nd1
    dual_delta = -phi * dr * nd2

    gamma = dq * n1 / (S * s_)
    vanna = -dq * n1 * d2 / sigma_
    volga = vega * d1 * d2 / sigma_
    dd1_dT = (2.0 * b * T_ - d2 * s_) / (2.0 * T_ * s_)  # d(d1)/dT
    charm = phi * q * dq * nd1 - dq * n1 * dd1_dT
    # veta and color as usually printed are derivatives in time to expiry; here time passes, so the sign turns
    veta = S * dq * n1 * sqrt_t_ * (q + b * d1 / s_ - (1.0 + d1 * d2) / (2.0 * T_))
    dual_gamma = dr * n2 / (K * s_)

    speed = -gamma / S * (d1 / s_ + 1.0)
    zomma = gamma * (d1 * d2 - 1.0) / sigma_
    color = dq * n1 / (2.0 * S * T_ * s_) * (2.0 * q * T_ + 1.0 + (2.0 * b * T_ - d2 * s_) * d1 / s_)
    ultima = -vega / (sigma_ * sigma_) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)

    if np.all(live):
        values = (price, delta, vega, theta, rho, psi, dual_delta, gamma, vanna, volga, charm, veta, dual_gamma)
        return Greeks(*values, speed, zomma, color, ultima)

    # s = 0: the discounted intrinsic value and its limits
    flat_price, flat_delta, flat_theta, weight = _intrinsic(x)
    flat = {
        "price": flat_price,
        "delta": flat_delta,
        "theta": flat_theta,
        "rho": weight * phi * T * K * dr,
        "psi": -weight * phi * T * S * dq,
        "dual_delta": -weight * phi * dr,
    }
    at_money = weight == 0.5
    curvature = np.where(at_money, np.nan, 0.0)
    out = {
        "price": price,
        "delta": delta,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "psi": psi,
        "dual_delta": dual_delta,
        "gamma": gamma,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
        "veta": veta,
        "dual_gamma": dual_gamma,
        "speed": speed,
        "zomma": zomma,
        "color": color,
        "ultima": ultima,
    }
    for name, value in out.items():
        limit = flat.get(name, curvature)
        out[name] = np.where(live, value, limit)
    return Greeks(**{name: np.asarray(value, dtype=np.float64) for name, value in out.items()})


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    yield_: ArrayLike,
    vol: ArrayLike,
    is_call: ArrayLike = True,
) -> Array:
    """The price alone: generalised Black-Scholes-Merton."""
    return greeks(spot, strike, maturity, rate, yield_, vol, is_call).price
