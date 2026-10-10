"""Raw SVI fitted slice by slice with the no-arbitrage conditions as hard constraints.

The original fitter (``volsurf.svi``) added penalties for a negative density and for
crossing the previous slice, on a fixed grid of 81 strikes. A penalty trades arbitrage
against fit, and a grid says nothing between or beyond its points: on the S&P 500 chain
of 9 October 2026, 36 of its 48 slices had a negative density somewhere and 30 of 47
neighbouring pairs crossed.

Here the conditions are constraints, and they hold on the whole real line:

* **Lee's bound** on both wing slopes, ``b (1 +- rho) <= 2``, settles the far wings, where
  ``g`` tends to ``1/4 - b^2 (1 +- rho)^2 / 16``.
* **Durrleman's** ``g(k) >= 0`` and the **calendar** gap ``w(k) - w_prev(k) >= 0`` are
  imposed on a finite set of strikes - a semi-infinite programme. It is solved by the
  exchange method: fit with SLSQP on the current set, find the worst point on the whole
  line with :func:`~volsurf.smile.svi.butterfly_check` and
  :func:`~volsurf.smile.svi.calendar_check`, add it to the set and fit again, until the
  check passes. The wing slopes must also rise against the previous slice.

SLSQP meets an active constraint only to about 1e-9, so each constraint carries a small
margin. The fit starts from the arbitrage-free eSSVI slice of :mod:`volsurf.smile.ssvi` - itself a
raw SVI - and minimises the bid-ask-weighted squared error in implied volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .ssvi import Quotes, SSVISlice
from .svi import RawSVI, butterfly_check, calendar_check

Array = NDArray[np.float64]

# SLSQP meets an active constraint only to about 1e-9, so the constraints carry a margin that
# keeps the exact check on the whole line from failing by a rounding: g >= 1e-6 (g is of
# order 1/4) and w - w_prev >= 1e-7 times the ATM variance.
G_MARGIN = 1e-6
CALENDAR_MARGIN = 1e-7


@dataclass(frozen=True)
class SliceFit:
    """A fitted slice and how it was reached."""

    slice: RawSVI
    maturity: float
    rmse: float  # implied volatility, unweighted
    rounds: int  # exchange-method rounds
    butterfly_margin: float  # min g on the whole line
    calendar_margin: float  # min w - w_prev on the whole line (inf for the first slice)


def _grid(slice_: RawSVI, points: int = 61) -> Array:
    y = np.linspace(-np.arcsinh(30.0), np.arcsinh(30.0), points)
    return np.asarray(slice_.m + slice_.sigma * np.sinh(y), dtype=np.float64)


def fit_slice(quotes: Quotes, start: RawSVI, previous: RawSVI | None = None, max_rounds: int = 12) -> SliceFit:
    """Fit one raw SVI slice to ``quotes`` under butterfly, Lee and calendar constraints."""
    k, iv, weight, t = quotes.k, quotes.iv, quotes.weight / np.mean(quotes.weight), quotes.maturity
    scale = np.array(
        [
            max(abs(start.a), start.total_variance(0.0).item(), 1e-8),
            max(start.b, 1e-6),
            1.0,
            max(start.sigma, 1e-4),
            max(start.sigma, 1e-4),
        ]
    )

    def build(x: Array) -> RawSVI:
        p = x * scale
        return RawSVI(
            float(p[0]), float(max(p[1], 1e-12)), float(np.clip(p[2], -0.9999, 0.9999)), float(p[3]), float(max(p[4], 1e-8))
        )

    def objective(x: Array) -> float:
        s = build(x)
        model = np.sqrt(np.maximum(s.total_variance(k), 1e-14) / t)
        return float(np.mean(weight * (model - iv) ** 2)) / 1e-6

    butterfly_points = list(_grid(start))
    calendar_points = list(_grid(previous)) if previous is not None else []

    def constraints() -> list[dict[str, object]]:
        kb = np.array(butterfly_points)
        cons: list[dict[str, object]] = [
            {"type": "ineq", "fun": lambda x: 2.0 - build(x).b * (1.0 + build(x).rho)},
            {"type": "ineq", "fun": lambda x: 2.0 - build(x).b * (1.0 - build(x).rho)},
            {"type": "ineq", "fun": lambda x, kb=kb: build(x).g(kb) - G_MARGIN},
            {"type": "ineq", "fun": lambda x: np.atleast_1d(build(x).min_variance)},
        ]
        if previous is not None:
            kc = np.array(calendar_points)
            left, right = previous.wing_slopes
            cons += [
                {
                    "type": "ineq",
                    "fun": lambda x, kc=kc: (
                        (build(x).total_variance(kc) - previous.total_variance(kc)) / scale[0] - CALENDAR_MARGIN
                    ),
                },
                {"type": "ineq", "fun": lambda x: build(x).wing_slopes[0] - left},
                {"type": "ineq", "fun": lambda x: build(x).wing_slopes[1] - right},
            ]
        return cons

    bounds = [(None, None), (1e-10, None), (-0.9999, 0.9999), (None, None), (1e-6, None)]
    x = np.array([start.a, start.b, start.rho, start.m, start.sigma]) / scale
    scaled_bounds = [
        (None if lo is None else lo / sc, None if hi is None else hi / sc) for (lo, hi), sc in zip(bounds, scale, strict=True)
    ]
    best = start
    start_x = x.copy()
    start_objective = objective(start_x)
    last_good = start_x
    fallback: tuple[float, RawSVI] | None = None  # the best arbitrage-free candidate seen, however it fits
    rounds = 0
    for attempt in range(1, max_rounds + 1):
        rounds = attempt
        result = minimize(
            objective,
            last_good,
            method="SLSQP",
            bounds=scaled_bounds,
            constraints=constraints(),
            options={"maxiter": 400, "ftol": 1e-12},
        )
        candidate = build(result.x)
        butterfly = butterfly_check(candidate)
        calendar = calendar_check(previous, candidate) if previous is not None else None
        arbitrage_free = butterfly.passed and (calendar is None or calendar.passed)
        # SLSQP can end on "constraints incompatible" at a point that passes the checks but fits badly:
        # a result is accepted only if it fits at least as well as the arbitrage-free start
        value = objective(result.x)
        sound = bool(np.isfinite(value)) and value <= start_objective * (1 + 1e-9)
        if arbitrage_free and sound:
            best = candidate
            break
        if arbitrage_free and np.isfinite(value) and (fallback is None or value < fallback[0]):
            fallback = (value, candidate)
        # exchange step: add the worst points found on the whole line, and refit from the last sound point
        if not butterfly.passed and np.isfinite(butterfly.at):
            butterfly_points.extend([butterfly.at - 1e-3, butterfly.at, butterfly.at + 1e-3])
        if calendar is not None and not calendar.passed and np.isfinite(calendar.at):
            calendar_points.extend([calendar.at - 1e-3, calendar.at, calendar.at + 1e-3])
        butterfly_points.extend(_grid(candidate, 31))
        if sound:
            last_good = result.x
    else:
        rounds = max_rounds
        # keep the arbitrage-free start - unless it crosses the previous slice, which a later fit cannot repair
        start_ok = butterfly_check(start).passed and (previous is None or calendar_check(previous, start).passed)
        if start_ok:
            best = start
        elif fallback is not None:
            best = fallback[1]
        else:
            raise RuntimeError(f"no arbitrage-free SVI slice found at T = {t:.4f}")
    model = best.implied_vol(k, t)
    final_b = butterfly_check(best)
    final_c = calendar_check(previous, best) if previous is not None else None
    return SliceFit(
        best,
        t,
        float(np.sqrt(np.mean((model - iv) ** 2))),
        rounds,
        final_b.worst,
        float("inf") if final_c is None else final_c.worst,
    )


def fit_surface(quotes: list[Quotes], starts: list[SSVISlice]) -> list[SliceFit]:
    """Fit every slice in order of maturity, each constrained against the one before."""
    order = np.argsort([q.maturity for q in quotes])
    fits: list[SliceFit] = []
    for i in order:
        previous = fits[-1].slice if fits else None
        fits.append(fit_slice(quotes[i], starts[i].raw(), previous))
    return fits
