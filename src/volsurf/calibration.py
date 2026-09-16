"""Heston calibration to an implied volatility surface.

Objective: vega-scaled price residuals,

    r_i = sqrt(w_i) * (C_model,i - C_market,i) / Vega_i  ~  sqrt(w_i) * (sigma_model,i - sigma_market,i),

which match implied-volatility errors to first order while needing no implied-vol
inversion inside the optimiser loop. Weights w_i are inverse bid-ask widths in vol
space, so tight, liquid quotes dominate.

Optimisation (default ``method="multistart"``):
1. Market-informed starting points. The short-dated ATM variance anchors v0, the
   long-dated ATM variance anchors theta, and a small design over (kappa, sigma, rho)
   covers the directions the surface does not pin down directly.
2. Each start runs trust-region-reflective least squares on a strike subsample.
3. The best candidate is polished on the full quote set.

``method="global"`` replaces step 1-2 with differential evolution, which is slower but
assumption-free. Reported fit quality always uses exact implied volatilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from .black_scholes import black76_vega
from .heston import HestonParams, heston_price_cos
from .implied_vol import implied_vol_black76
from .market_data import MarketSnapshot

PARAM_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
DEFAULT_BOUNDS = ((1e-4, 1.0), (0.05, 15.0), (1e-3, 1.0), (0.05, 4.0), (-0.99, 0.5))
_FAST_COS = {"truncation": 14.0, "tail_tolerance": 1e-11}
_START_DESIGN = ((1.5, 0.6, -0.7), (4.0, 1.2, -0.8), (0.8, 0.35, -0.4), (6.0, 2.0, -0.6))


@dataclass
class _ExpiryBlock:
    maturity: float
    forward: float
    discount: float
    strikes: np.ndarray
    is_call: np.ndarray
    price: np.ndarray
    vega: np.ndarray
    weight: np.ndarray


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_vol: float
    weighted_rmse_vol: float
    max_abs_error_vol: float
    n_quotes: int
    n_evaluations: int
    elapsed_seconds: float
    method: str
    residuals: pd.DataFrame

    def summary(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {name: float(getattr(self.params, name)) for name in PARAM_NAMES}
        out.update(
            feller_ratio=self.params.feller_ratio,
            rmse_vol_pts=100 * self.rmse_vol,
            weighted_rmse_vol_pts=100 * self.weighted_rmse_vol,
            max_abs_error_vol_pts=100 * self.max_abs_error_vol,
            n_quotes=self.n_quotes,
            evaluations=self.n_evaluations,
            seconds=self.elapsed_seconds,
            method=self.method,
        )
        return out

    def error_by_expiry(self) -> pd.DataFrame:
        g = self.residuals.groupby("expiry", sort=False)
        return pd.DataFrame(
            {
                "maturity": g["maturity"].first(),
                "rmse_vol_pts": g["error"].apply(lambda e: 100 * float(np.sqrt(np.nanmean(e**2)))),
                "bias_vol_pts": g["error"].apply(lambda e: 100 * float(np.nanmean(e))),
                "n_quotes": g["error"].size(),
            }
        ).reset_index()


def _subsample(quotes: pd.DataFrame, max_quotes: int) -> pd.DataFrame:
    if len(quotes) <= max_quotes:
        return quotes
    idx = np.unique(np.round(np.linspace(0, len(quotes) - 1, max_quotes)).astype(int))
    return quotes.iloc[idx]


def _blocks(snapshot: MarketSnapshot, max_quotes_per_expiry: int) -> list[_ExpiryBlock]:
    blocks = []
    for e in snapshot.expiries:
        q = _subsample(e.quotes.sort_values("strike"), max_quotes_per_expiry)
        strikes = q["strike"].to_numpy(float)
        iv = q["iv_mid"].to_numpy(float)
        vega_floor = 1e-4 * e.forward * np.sqrt(e.maturity) * e.discount
        vega = np.maximum(black76_vega(e.forward, strikes, e.maturity, e.discount, iv), vega_floor)
        weight = q["weight"].to_numpy(float) if "weight" in q else np.ones(len(q))
        blocks.append(
            _ExpiryBlock(
                e.maturity,
                e.forward,
                e.discount,
                strikes,
                (q["option_type"] == "C").to_numpy(),
                q["mid"].to_numpy(float),
                vega,
                weight,
            )
        )
    mean_weight = np.concatenate([b.weight for b in blocks]).mean()
    for b in blocks:
        b.weight = b.weight / mean_weight
    return blocks


def _residuals(x: np.ndarray, blocks: list[_ExpiryBlock]) -> np.ndarray:
    try:
        params = HestonParams.from_array(x)
    except ValueError:
        return np.concatenate([np.ones(b.strikes.size) for b in blocks])
    out = []
    for b in blocks:
        model = heston_price_cos(b.forward, b.strikes, b.maturity, b.discount, params, b.is_call, **_FAST_COS)
        out.append(np.sqrt(b.weight) * (model - b.price) / b.vega)
    r = np.concatenate(out)
    return np.where(np.isfinite(r), r, 1.0)


def market_informed_starts(snapshot: MarketSnapshot, bounds=DEFAULT_BOUNDS) -> list[np.ndarray]:
    short_var = snapshot.expiries[0].atm_vol() ** 2
    long_var = snapshot.expiries[-1].atm_vol() ** 2
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    starts = []
    for kappa, sigma, rho in _START_DESIGN:
        x = np.array([short_var, kappa, long_var, sigma, rho])
        starts.append(np.clip(x, lower + 1e-8, upper - 1e-8))
    return starts


def calibrate_heston(
    snapshot: MarketSnapshot,
    method: Literal["multistart", "global"] = "multistart",
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
    max_quotes_per_expiry: int = 30,
    search_quotes_per_expiry: int = 12,
    global_iterations: int = 25,
    population: int = 10,
    seed: int = 42,
) -> CalibrationResult:
    start_time = time.perf_counter()
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    search_blocks = _blocks(snapshot, search_quotes_per_expiry)
    full_blocks = _blocks(snapshot, max_quotes_per_expiry)
    evaluations = 0

    def fit(x0: np.ndarray, blocks: list[_ExpiryBlock], max_nfev: int):
        nonlocal evaluations

        def fun(x: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            return _residuals(x, blocks)

        return least_squares(
            fun,
            np.clip(x0, lower + 1e-10, upper - 1e-10),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )

    if method == "global":

        def objective(x: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            r = _residuals(x, search_blocks)
            return float(r @ r)

        de = differential_evolution(
            objective,
            bounds,
            maxiter=global_iterations,
            popsize=population,
            seed=seed,
            tol=1e-8,
            polish=False,
            init="sobol",
        )
        best_x = de.x
    elif method == "multistart":
        candidates = [fit(x0, search_blocks, 120) for x0 in market_informed_starts(snapshot, bounds)]
        best_x = min(candidates, key=lambda r: r.cost).x
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    polished = fit(best_x, full_blocks, 300)
    params = HestonParams.from_array(polished.x)
    residuals = _model_residuals(snapshot, params)
    err = residuals["error"].to_numpy()
    finite = np.isfinite(err)
    w = residuals["weight"].to_numpy()[finite]

    return CalibrationResult(
        params=params,
        rmse_vol=float(np.sqrt(np.mean(err[finite] ** 2))),
        weighted_rmse_vol=float(np.sqrt(np.sum(w * err[finite] ** 2) / np.sum(w))),
        max_abs_error_vol=float(np.max(np.abs(err[finite]))),
        n_quotes=int(finite.sum()),
        n_evaluations=evaluations,
        elapsed_seconds=time.perf_counter() - start_time,
        method=method,
        residuals=residuals,
    )


def _model_residuals(snapshot: MarketSnapshot, params: HestonParams) -> pd.DataFrame:
    rows = []
    for e in snapshot.expiries:
        q = e.quotes
        strikes = q["strike"].to_numpy(float)
        is_call = (q["option_type"] == "C").to_numpy()
        model_price = heston_price_cos(e.forward, strikes, e.maturity, e.discount, params, is_call)
        rows.append(
            pd.DataFrame(
                {
                    "expiry": e.expiry,
                    "maturity": e.maturity,
                    "strike": strikes,
                    "k": q["k"].to_numpy(float),
                    "iv_market": q["iv_mid"].to_numpy(float),
                    "iv_bid": q["iv_bid"].to_numpy(float),
                    "iv_ask": q["iv_ask"].to_numpy(float),
                    "iv_heston": implied_vol_black76(model_price, e.forward, strikes, e.maturity, e.discount, is_call),
                    "weight": q["weight"].to_numpy(float),
                }
            )
        )
    residuals = pd.concat(rows, ignore_index=True)
    residuals["error"] = residuals["iv_heston"] - residuals["iv_market"]
    return residuals
