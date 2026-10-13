"""What each model says the smile does when the underlying moves - and so what its delta is.

Fitted to the same smile today, local volatility and SABR disagree about tomorrow.

* **Local volatility** is a fixed function of the *absolute* strike and time. When the
  forward rises, the at-the-money point moves into a lower part of a downward-sloping
  local-volatility surface, so the at-the-money implied volatility falls - by about twice
  the smile's own slope, since implied volatility averages local volatility between spot
  and strike (Hagan, Kumar, Lesniewski and Woodward, 2002). The smile moves the opposite
  way to the market.
* **SABR** carries the smile with the forward: with ``beta = 1`` the at-the-money
  volatility depends on the forward only through the correlated move of ``alpha``.

The difference is a delta. Under a sticky-strike view the delta is Black's. Under SABR it
is Bartlett's (2006): the forward's move brings with it the expected move of ``alpha``,
``d alpha = rho nu / F^beta dF``, the regression of the move in alpha on the move in F. Under local volatility it is the
model's own finite difference.

The local-volatility smile after a move is computed exactly in the model: the forward
Dupire equation is solved again with the local variance held fixed in absolute strike,
``v'(k', t) = v(k' + ln(F'/F), t)``, on the surface's monthly expiries: between daily
weeklies, local volatility in the wings is noise (see :mod:`~volsurf.localvol.dupire`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..market.black import black_price
from ..smile.interpolation import Surface
from .dupire import implied_from_calls, local_variance, solve_forward
from .sabr import SABR, hagan_vol

Array = NDArray[np.float64]


def frozen_local_variance(surface: Surface, shift: float = 0.0) -> object:
    """Local variance as a function of ``(k', t)`` after the forward moves by ``ln(F'/F) = shift``."""

    def variance(k: Array, t: float) -> Array:
        return local_variance(surface, k + shift, max(t, 1e-6))

    return variance


def monthly(surface: Surface, first: float = 25 / 365, spacing: float = 1.25) -> Surface:
    """The surface on its monthly expiries: from ``first`` years, each at least ``spacing`` times the last."""
    keep: list[int] = []
    for i, t in enumerate(surface.maturities):
        if t >= first and (not keep or t >= surface.maturities[keep[-1]] * spacing):
            keep.append(i)
    return Surface(surface.maturities[keep], [surface.slices[i] for i in keep])


@dataclass(frozen=True)
class SmileMove:
    shifts: Array  # ln(F'/F)
    strikes: Array  # absolute strikes, as multiples of today's forward
    local_vol: Array  # implied vols after each move, one row per shift
    sabr: Array
    sticky_strike: Array


def smile_moves(surface: Surface, sabr: SABR, t: float, shifts: Array, strikes: Array, space_steps: int = 1600) -> SmileMove:
    """Implied volatilities at fixed absolute strikes after the forward moves, under each model."""
    lv_rows, sabr_rows = [], []
    for shift in shifts:
        k_new = np.log(strikes) - shift  # log-moneyness against the moved forward
        sol = solve_forward(
            frozen_local_variance(surface, float(shift)), np.array([t]), space_steps=space_steps, steps_per_year=800
        )
        calls = np.interp(k_new, sol.k, sol.calls[0])
        lv_rows.append(implied_from_calls(k_new, calls, t))
        sabr_rows.append(hagan_vol(float(np.exp(shift)), strikes, t, sabr, obloj=True))
    today = surface.implied_vol(np.log(strikes), t)
    return SmileMove(
        np.asarray(shifts, dtype=np.float64),
        np.asarray(strikes, dtype=np.float64),
        np.array(lv_rows),
        np.array(sabr_rows),
        np.tile(today, (len(shifts), 1)),
    )


def black_delta(f: float, k: float, t: float, vol: float) -> float:
    h = 1e-4 * f
    return float((black_price(f + h, k, t, vol) - black_price(f - h, k, t, vol)) / (2 * h))


def sabr_delta(f: float, k: float, t: float, p: SABR, bartlett: bool = True) -> float:
    """dC/dF under SABR: Hagan's (alpha held) or Bartlett's (alpha moving with F through rho)."""
    h = 1e-4 * f

    def call(fwd: float, alpha: float) -> float:
        q = SABR(alpha, p.beta, p.rho, p.nu)
        return float(black_price(fwd, k, t, hagan_vol(fwd, np.array([k]), t, q, obloj=True)[0]))

    delta = (call(f + h, p.alpha) - call(f - h, p.alpha)) / (2 * h)
    if not bartlett:
        return float(delta)
    da = 1e-4 * p.alpha
    vega_alpha = (call(f, p.alpha + da) - call(f, p.alpha - da)) / (2 * da)
    return float(delta + vega_alpha * p.rho * p.nu / f**p.beta)
