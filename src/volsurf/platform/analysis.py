"""One analysis over every engine: quotes to surface, models, exotics, hedging and a book's risk.

The steps are the days of the lab, in order, on one source:

1. **Market** (Day 3): parity forwards and discount factors, quotes kept or dropped for stated reasons.
2. **Surface** (Day 4): raw SVI under hard no-arbitrage constraints, joined by price interpolation.
3. **Heston** (Day 5): multistart calibration to bid-ask-weighted implied volatilities.
4. **SABR and local volatility** (Day 6): SABR per expiry; Dupire from the surface.
5. **Exotics** (Day 7): a book of barriers, digitals and an Asian under Black-Scholes, Heston and local volatility.
6. **Hedging** (Day 8): a three-month call hedged in Heston and local-volatility worlds with each model's delta.
7. **Book risk** (Day 8): a dealer book's stress ladder, and - with the local history - its historical VaR.

Every number the report quotes is collected in :attr:`Analysis.results`, a flat mapping that
the manifest stores and :mod:`volsurf.platform.manifest` compares on a rerun.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..exotics.model_risk import Book as ExoticBook
from ..exotics.model_risk import price_book
from ..localvol.dynamics import monthly
from ..localvol.sabr import SABRFit
from ..localvol.sabr import calibrate as calibrate_sabr
from ..market.cleaning import Market, Slice, prepare
from ..risk import book as bk
from ..risk import history
from ..risk.models import CrossModel, cross_model
from ..smile.build import SurfaceFit, build_surface, one_per_date
from ..smile.interpolation import Surface
from ..stochvol.calibration import Calibration, Expiry, multistart
from .sources import Source

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    quick: bool = False  # fewer starts and paths: minutes become seconds, for tests and first looks
    heston_starts: int = 8  # a power of two keeps the Sobol design balanced
    exotics: bool = True
    exotic_paths: int = 50_000
    hedging: bool = True
    hedge_paths: int = 20_000
    seed: int = 7

    @classmethod
    def fast(cls, **overrides: object) -> Config:
        base: dict[str, object] = {"quick": True, "heston_starts": 2, "exotic_paths": 10_000, "hedge_paths": 2_000}
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Analysis:
    source: Source
    config: Config
    market: Market
    fit: SurfaceFit
    monthly: Surface
    expiries: list[Expiry]  # those Heston and SABR are fitted to
    heston: Calibration
    sabr: list[tuple[float, SABRFit]]
    exotics: ExoticBook | None
    hedging: CrossModel | None
    book: bk.Book
    ladder: pd.DataFrame
    historical: bk.HistoricalRisk | None
    dynamics: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)
    results: dict[str, float] = field(default_factory=dict)


def model_expiries(market: Market, lo: float = 25 / 365, hi: float = 2.2, spacing: float = 1.25, per: int = 15) -> list[Expiry]:
    """Expiries for the models: a month to two years, each at least 1.25 times the last, 15 quotes each across the smile."""
    chosen: list[Expiry] = []
    for e in one_per_date(market):
        if not lo <= e.maturity <= hi or (chosen and e.maturity < chosen[-1].maturity * spacing):
            continue
        q = e.quotes
        z = q["k"] / np.sqrt(e.maturity)
        q = q[(z > -2.5) & (z < 1.0)]
        q = q.iloc[np.unique(np.linspace(0, len(q) - 1, min(len(q), per)).astype(int))]
        width = np.clip((q["iv_ask"] - q["iv_bid"]).to_numpy(float), 0.002, None)
        chosen.append(Expiry(e.maturity, q["k"].to_numpy(float), q["iv_mid"].to_numpy(float), width))
    return chosen


def _nearest(fit: SurfaceFit, years: float) -> Slice:
    return min(fit.expiries, key=lambda e: abs(e.maturity - years))


def _rates(fit: SurfaceFit, e: Slice) -> tuple[float, float]:
    spot = float(fit.surface.spot or e.forward)
    return float(-np.log(e.discount) / e.maturity), float(np.log(e.forward / spot) / e.maturity)


def run(source: Source, config: Config | None = None, log: Callable[[str], None] = print) -> Analysis:
    config = config or Config()
    timings: dict[str, float] = {}

    def step(name: str, started: float) -> float:
        timings[name] = time.perf_counter() - started
        return timings[name]

    t = time.perf_counter()
    market = prepare(source.chain)
    fit = build_surface(market)
    surface = monthly(fit.surface)
    log(f"  surface: {len(fit.expiries)} expiries, {sum(len(q.k) for q in fit.quotes):,} quotes ({step('surface', t):.1f}s)")

    t = time.perf_counter()
    expiries = model_expiries(market)
    heston = multistart(expiries, config.heston_starts, seed=config.seed)[0]
    h = heston.heston
    log(
        f"  Heston: v0 {h.v0:.4f} kappa {h.kappa:.2f} theta {h.theta:.4f} sigma {h.sigma:.2f} rho {h.rho:.2f} "
        f"({step('heston', t):.1f}s)"
    )

    t = time.perf_counter()
    sabr = []
    for e in expiries:
        weights = 1.0 / e.width**2
        sabr.append((e.maturity, calibrate_sabr(1.0, np.exp(e.k), e.maturity, e.iv, weights)))
    log(f"  SABR: {len(sabr)} expiries ({step('sabr', t):.1f}s)")

    exotics = None
    if config.exotics:
        t = time.perf_counter()
        r, b = _rates(fit, _nearest(fit, 1.0))
        atm = float(surface.implied_vol(np.array([0.0]), 1.0)[0])
        steps = 126 if config.quick else 252
        exotics = price_book(1.0, r, b, atm, h, surface, paths=config.exotic_paths, steps=steps, seed=config.seed)
        log(f"  exotics: {len(exotics.contracts)} contracts x {len(exotics.models)} models ({step('exotics', t):.1f}s)")

    hedging = None
    if config.hedging:
        t = time.perf_counter()
        near = _nearest(fit, 0.25)
        r, b = _rates(fit, near)
        hedging = cross_model(
            surface, h, near.maturity, r, b, steps=21 if config.quick else 63, paths=config.hedge_paths, seed=config.seed
        )
        log(f"  hedging: two worlds, {config.hedge_paths:,} paths ({step('hedging', t):.1f}s)")

    t = time.perf_counter()
    book = bk.build(fit)
    dynamics = {"sticky moneyness": 0.0, "sticky strike": 1.0, "local volatility": 2.0}
    historical = None
    if source.kind == "snapshot" and history.available():
        past = history.load()
        sv = history.spot_vol(past)
        e30 = min(fit.expiries, key=lambda e: abs(e.maturity * 365 - 30))
        iv = fit.surface.implied_vol(np.array([-0.01, 0.01]), e30.maturity)
        dynamics["market"] = float(sv.beta / ((iv[1] - iv[0]) / 0.02))
        historical = bk.historical(book, past, 1)
    ladder = bk.stress_ladder(book, np.round(np.linspace(-0.2, 0.1, 13), 3), dynamics)
    log(f"  book: stress under {len(dynamics)} smile dynamics{', historical VaR' if historical else ''} ({step('book', t):.1f}s)")

    analysis = Analysis(
        source, config, market, fit, surface, expiries, heston, sabr, exotics, hedging, book, ladder, historical, dynamics
    )
    analysis.timings = timings
    analysis.results = collect(analysis)
    return analysis


def collect(a: Analysis) -> dict[str, float]:
    """Every headline number, flat and named, for the report, the manifest and a rerun's comparison."""
    out: dict[str, float] = {
        "spot": float(a.source.chain.spot),
        "expiries": float(len(a.fit.expiries)),
        "quotes": float(sum(len(q.k) for q in a.fit.quotes)),
        "surface.median_rmse_vol": float(np.median([f.rmse for f in a.fit.svi])),
        "surface.worst_rmse_vol": float(max(f.rmse for f in a.fit.svi)),
        "surface.min_butterfly_margin": float(min(f.butterfly_margin for f in a.fit.svi)),
        "surface.min_calendar_margin": float(min(f.calendar_margin for f in a.fit.svi)),
    }
    h = a.heston.heston
    out.update({f"heston.{k}": float(getattr(h, k)) for k in ("v0", "kappa", "theta", "sigma", "rho")})
    out["heston.rmse_vol"] = float(a.heston.rmse)
    out["sabr.median_nu"] = float(np.median([f.params.nu for _, f in a.sabr]))
    out["sabr.median_rho"] = float(np.median([f.params.rho for _, f in a.sabr]))
    if a.exotics is not None:
        for (model, contract), estimate in a.exotics.prices.items():
            out[f"exotics.{model}.{contract}"] = float(estimate.value)
    if a.hedging is not None:
        for name, value in a.hedging.deltas.items():
            out[f"hedging.delta.{name}"] = float(value)
        for row in a.hedging.table.itertuples():
            out[f"hedging.std_pct.{row.world}.{row.strategy}"] = float(row.std_pct_premium)
    for name in a.ladder.columns:
        out[f"book.stress_-10%.{name}"] = float(a.ladder.loc[-0.1, name])
    if a.historical is not None:
        out["book.var99_1d"] = a.historical.var()
        out["book.var99_1d_greeks"] = a.historical.var(which="greeks")
    return out
