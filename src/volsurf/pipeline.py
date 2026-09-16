"""End-to-end analysis: snapshot -> SVI surface -> local vol -> Heston calibration -> hedging lab."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .calibration import CalibrationResult, calibrate_heston
from .hedging import HedgingExperiment, run_black_scholes_world, run_heston_world
from .market_data import MarketSnapshot
from .surface import VolSurface
from .svi import SVIFit, fit_svi_slice


@dataclass
class AnalysisConfig:
    calibration_method: str = "multistart"
    max_quotes_per_expiry: int = 30
    run_hedging: bool = True
    hedge_maturity: float = 0.25
    heston_hedge_paths: int = 4_000
    heston_rebalances: tuple[int, ...] = (3, 9, 21, 63)
    bs_hedge_paths: int = 20_000
    bs_rebalances: tuple[int, ...] = (4, 16, 64, 256)
    seed: int = 7


@dataclass
class AnalysisResult:
    snapshot: MarketSnapshot
    svi_fits: list[SVIFit]
    surface: VolSurface
    calibration: CalibrationResult
    hedging_bs: HedgingExperiment | None
    hedging_heston: HedgingExperiment | None
    timings: dict[str, float] = field(default_factory=dict)


def fit_surface(snapshot: MarketSnapshot) -> tuple[list[SVIFit], VolSurface]:
    fits: list[SVIFit] = []
    previous = None
    for expiry in snapshot.expiries:
        q = expiry.quotes
        fit = fit_svi_slice(q["k"], q["iv_mid"], expiry.maturity, q["weight"], previous=previous)
        fits.append(fit)
        previous = fit.slice
    surface = VolSurface(
        [f.slice for f in fits],
        [e.forward for e in snapshot.expiries],
        [e.discount for e in snapshot.expiries],
        snapshot.spot,
    )
    return fits, surface


def _hedging_market(snapshot: MarketSnapshot, maturity: float) -> tuple[float, float]:
    ts = snapshot.term_structure()
    row = ts.iloc[int(np.argmin(np.abs(ts["maturity"] - maturity)))]
    rate = float(row["implied_rate"])
    return rate, float(rate - row["implied_carry"])


def run_analysis(
    snapshot: MarketSnapshot,
    config: AnalysisConfig | None = None,
    log: Callable[[str], None] = print,
) -> AnalysisResult:
    config = config or AnalysisConfig()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    fits, surface = fit_surface(snapshot)
    timings["svi_surface"] = time.perf_counter() - t
    log(
        f"  SVI surface: {len(fits)} slices, worst slice RMSE {100 * max(f.rmse_vol for f in fits):.2f} vol pts "
        f"({timings['svi_surface']:.1f}s)"
    )

    t = time.perf_counter()
    calibration = calibrate_heston(snapshot, method=config.calibration_method, max_quotes_per_expiry=config.max_quotes_per_expiry)
    timings["heston_calibration"] = time.perf_counter() - t
    p = calibration.params
    log(
        f"  Heston: v0={p.v0:.4f} kappa={p.kappa:.3f} theta={p.theta:.4f} sigma={p.sigma:.3f} rho={p.rho:.3f} | "
        f"RMSE {100 * calibration.rmse_vol:.2f} vol pts ({timings['heston_calibration']:.1f}s)"
    )

    hedging_bs = hedging_heston = None
    if config.run_hedging:
        rate, dividend_yield = _hedging_market(snapshot, config.hedge_maturity)
        atm_vol = snapshot.expiries[
            int(np.argmin([abs(e.maturity - config.hedge_maturity) for e in snapshot.expiries]))
        ].atm_vol()
        t = time.perf_counter()
        hedging_bs = run_black_scholes_world(
            spot=100.0,
            strike=100.0,
            maturity=config.hedge_maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            vol=atm_vol,
            rebalances=config.bs_rebalances,
            n_paths=config.bs_hedge_paths,
            seed=config.seed,
        )
        hedging_heston = run_heston_world(
            calibration.params,
            spot=100.0,
            strike=100.0,
            maturity=config.hedge_maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            rebalances=config.heston_rebalances,
            n_paths=config.heston_hedge_paths,
            seed=config.seed + 1,
        )
        timings["hedging_lab"] = time.perf_counter() - t
        log(
            f"  Hedging lab: {config.heston_hedge_paths:,} Heston paths x {max(config.heston_rebalances)} steps "
            f"({timings['hedging_lab']:.1f}s)"
        )

    return AnalysisResult(snapshot, fits, surface, calibration, hedging_bs, hedging_heston, timings)
