"""From a cleaned market to an arbitrage-free surface, in one call.

Each expiry date contributes one slice. Where SPX lists both an AM-settled monthly and a
PM-settled weekly on the same date, the monthly is used: it is the contract the VIX and
most listed volume reference, and the two differ by six and a half hours.

Quotes are weighted by the inverse square of their bid-ask width in volatility, floored
at 0.2 vol points, so a residual of one is one bid-ask width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..market.cleaning import Market, Slice
from .fit import SliceFit, fit_surface
from .interpolation import Surface
from .ssvi import SSVI, Quotes, SSVISlice, fit_essvi, fit_ssvi

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceFit:
    """Every model fitted to one market, slice for slice."""

    expiries: list[Slice]
    quotes: list[Quotes]
    ssvi: SSVI
    essvi: list[SSVISlice]
    svi: list[SliceFit]
    surface: Surface


def one_per_date(market: Market) -> list[Slice]:
    chosen: dict[str, Slice] = {}
    for s in market.slices:
        if s.expiry not in chosen or s.root == "SPX":
            chosen[s.expiry] = s
    return sorted(chosen.values(), key=lambda s: s.maturity)


def quotes_of(expiry: Slice, floor: float = 0.002) -> Quotes:
    q = expiry.quotes
    width = np.clip((q["iv_ask"] - q["iv_bid"]).to_numpy(float), floor, None)
    return Quotes(expiry.maturity, q["k"].to_numpy(float), q["iv_mid"].to_numpy(float), 1.0 / width**2)


def build_surface(market: Market) -> SurfaceFit:
    expiries = one_per_date(market)
    quotes = [quotes_of(e) for e in expiries]
    essvi = fit_essvi(quotes)
    svi = fit_surface(quotes, essvi)
    surface = Surface(
        np.array([e.maturity for e in expiries]),
        [f.slice for f in svi],
        np.array([e.forward for e in expiries]),
        market.chain.spot,
    )
    return SurfaceFit(expiries, quotes, fit_ssvi(quotes), essvi, svi, surface)


def inside_spread(expiry: Slice, total_variance: Array) -> float:
    """The share of quotes whose bid-ask range contains the model's implied volatility."""
    q = expiry.quotes
    iv = np.sqrt(np.maximum(total_variance, 0.0) / expiry.maturity)
    return float(np.mean((iv >= q["iv_bid"].to_numpy() - 1e-9) & (iv <= q["iv_ask"].to_numpy() + 1e-9)))
