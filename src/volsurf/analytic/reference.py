"""An independent reference for every Greek: the price in arbitrary precision, differentiated numerically.

mpmath evaluates the Black-Scholes-Merton price to 50 significant digits and
differentiates it by high-order finite differences in that precision, so the
reference shares nothing with the analytic formulas but the definition of the
price itself. A sign convention or a missing term in an analytic Greek shows up
as a disagreement in the first digit; a correct formula agrees to fifteen.

Each Greek is defined here by the derivative it is - which variable, which order,
and whether time is measured to expiry (``T``) or as the calendar passes (``t = -T``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: name -> (variables differentiated, in order; sign applied: -1 per calendar-time derivative)
DEFINITIONS: dict[str, tuple[tuple[str, ...], int]] = {
    "price": ((), 1),
    "delta": (("S",), 1),
    "vega": (("sigma",), 1),
    "theta": (("T",), -1),
    "rho": (("r",), 1),
    "psi": (("q",), 1),
    "dual_delta": (("K",), 1),
    "gamma": (("S", "S"), 1),
    "vanna": (("S", "sigma"), 1),
    "volga": (("sigma", "sigma"), 1),
    "charm": (("S", "T"), -1),
    "veta": (("sigma", "T"), -1),
    "dual_gamma": (("K", "K"), 1),
    "speed": (("S", "S", "S"), 1),
    "zomma": (("S", "S", "sigma"), 1),
    "color": (("S", "S", "T"), -1),
    "ultima": (("sigma", "sigma", "sigma"), 1),
}
ORDER = ("S", "K", "T", "r", "q", "sigma")


def reference_greeks(
    spot: float, strike: float, maturity: float, rate: float, yield_: float, vol: float, is_call: bool, digits: int = 50
) -> dict[str, float]:
    """Every Greek in ``DEFINITIONS``, from the price differentiated in arbitrary precision."""
    import mpmath as mp

    mp.mp.dps = digits
    phi = 1 if is_call else -1

    def value(S: Any, K: Any, T: Any, r: Any, q: Any, sigma: Any) -> Any:
        s = sigma * mp.sqrt(T)
        d1 = (mp.log(S / K) + (r - q + sigma * sigma / 2) * T) / s
        d2 = d1 - s
        return phi * (S * mp.exp(-q * T) * mp.ncdf(phi * d1) - K * mp.exp(-r * T) * mp.ncdf(phi * d2))

    point = [mp.mpf(v) for v in (spot, strike, maturity, rate, yield_, vol)]
    out: dict[str, float] = {}
    for name, (variables, sign) in DEFINITIONS.items():
        orders = tuple(variables.count(v) for v in ORDER)
        derivative: Callable[..., Any] = mp.diff
        result = value(*point) if not variables else derivative(value, point, orders)
        out[name] = float(sign ** sum(1 for v in variables if v == "T") * result)
    return out
