"""Validation suite: every engine checked against published values or an independent method."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .black_scholes import black76_price, bsm_greeks, bsm_price
from .calibration import calibrate_heston
from .hedging import run_black_scholes_world
from .heston import HestonParams, heston_cos_sensitivities, heston_price_cos, heston_price_integration
from .implied_vol import implied_vol_black76
from .lattice import binomial_price
from .market_data import build_snapshot, synthetic_chain
from .monte_carlo import lsm_american_price, mc_european_gbm, mc_european_heston
from .svi import SVISlice

FANG_OOSTERLEE_PARAMS = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
FANG_OOSTERLEE_PRICE = 5.785155450


@dataclass
class Check:
    area: str
    check: str
    reference: str
    result: str
    error: str
    passed: bool
    seconds: float


def _timed(fn: Callable[[], tuple[str, str, str, bool]]) -> tuple[tuple[str, str, str, bool], float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def run_validation(fast: bool = False) -> pd.DataFrame:
    checks: list[Check] = []

    def add(area: str, name: str, fn: Callable[[], tuple[str, str, str, bool]]) -> None:
        (reference, result, error, passed), seconds = _timed(fn)
        checks.append(Check(area, name, reference, result, error, passed, seconds))

    def bs_hull():
        call = float(bsm_price(100, 100, 1, 0.05, 0, 0.2, True))
        return "10.4506 (Hull)", f"{call:.4f}", f"{abs(call - 10.4506):.1e}", abs(call - 10.4506) < 1e-4

    def parity():
        rng = np.random.default_rng(0)
        s, k = rng.uniform(50, 150, 5000), rng.uniform(50, 150, 5000)
        t, r, q, v = (
            rng.uniform(0.01, 5, 5000),
            rng.uniform(-0.01, 0.08, 5000),
            rng.uniform(0, 0.05, 5000),
            rng.uniform(0.05, 1, 5000),
        )
        gap = np.max(
            np.abs(
                bsm_price(s, k, t, r, q, v, True) - bsm_price(s, k, t, r, q, v, False) - (s * np.exp(-q * t) - k * np.exp(-r * t))
            )
        )
        return "0", f"{gap:.1e}", f"{gap:.1e}", gap < 1e-10

    def greeks_fd():
        s, k, t, r, q, v = 105.0, 100.0, 0.7, 0.03, 0.01, 0.27
        g = bsm_greeks(s, k, t, r, q, v, True)
        h = 1e-4
        fd = {
            "delta": (bsm_price(s + h, k, t, r, q, v) - bsm_price(s - h, k, t, r, q, v)) / (2 * h),
            "vega": (bsm_price(s, k, t, r, q, v + h) - bsm_price(s, k, t, r, q, v - h)) / (2 * h),
            "rho": (bsm_price(s, k, t, r + h, q, v) - bsm_price(s, k, t, r - h, q, v)) / (2 * h),
        }
        worst = max(abs(float(getattr(g, name)) - float(val)) for name, val in fd.items())
        return "finite differences", "delta, vega, rho", f"{worst:.1e}", worst < 1e-5

    def iv_roundtrip():
        rng = np.random.default_rng(1)
        n = 20_000
        f, t, v = 100.0, rng.uniform(1 / 365, 5, n), rng.uniform(0.03, 2.0, n)
        k = f * np.exp(rng.uniform(-1.0, 1.0, n) * v * np.sqrt(t) * 2.5)
        is_call = rng.random(n) < 0.5
        price = black76_price(f, k, t, 0.97, v, is_call)
        otm = black76_price(f, k, t, 0.97, v, k >= f) / (0.97 * f)
        ok = otm > 1e-9
        start = time.perf_counter()
        iv = implied_vol_black76(price[ok], f, k[ok], t[ok], 0.97, is_call[ok])
        elapsed = time.perf_counter() - start
        worst = float(np.nanmax(np.abs(iv - v[ok])))
        return (
            f"{ok.sum():,} random quotes",
            f"{elapsed * 1000:.0f} ms total",
            f"{worst:.1e}",
            worst < 1e-6 and not np.isnan(iv).any(),
        )

    def lr_tree():
        ref = float(bsm_price(100, 100, 1, 0.05, 0.02, 0.25, True))
        lr = binomial_price(100, 100, 1, 0.05, 0.02, 0.25, True, steps=201)
        crr = binomial_price(100, 100, 1, 0.05, 0.02, 0.25, True, steps=2001, method="crr")
        return (
            f"BS {ref:.6f}",
            f"LR(201) {lr:.6f}; CRR(2001) {crr:.6f}",
            f"LR {abs(lr - ref):.1e}; CRR {abs(crr - ref):.1e}",
            abs(lr - ref) < abs(crr - ref),
        )

    def american_tree():
        value = binomial_price(36, 40, 1, 0.06, 0, 0.2, False, True, steps=2001)
        return (
            "4.478 (finite difference, Longstaff-Schwartz 2001)",
            f"{value:.4f}",
            f"{abs(value - 4.478):.1e}",
            abs(value - 4.478) < 0.01,
        )

    def lsm():
        res = lsm_american_price(36, 40, 1, 0.06, 0, 0.2, False, 50, 50_000 if fast else 200_000, seed=11)
        return (
            "4.472 +- 0.010 (Longstaff-Schwartz 2001)",
            f"{res.price:.4f} +- {res.std_error:.4f}",
            f"{abs(res.price - 4.472):.1e}",
            abs(res.price - 4.472) < 0.04,
        )

    def cos_reference():
        value = float(heston_price_cos(100, [100.0], 1.0, 1.0, FANG_OOSTERLEE_PARAMS)[0])
        return (
            f"{FANG_OOSTERLEE_PRICE:.9f} (Fang-Oosterlee 2008)",
            f"{value:.9f}",
            f"{abs(value - FANG_OOSTERLEE_PRICE):.1e}",
            abs(value - FANG_OOSTERLEE_PRICE) < 1e-6,
        )

    def integration_reference():
        value = heston_price_integration(100, 100, 1.0, 1.0, FANG_OOSTERLEE_PARAMS)
        return (
            f"{FANG_OOSTERLEE_PRICE:.9f} (Fang-Oosterlee 2008)",
            f"{value:.9f}",
            f"{abs(value - FANG_OOSTERLEE_PRICE):.1e}",
            abs(value - FANG_OOSTERLEE_PRICE) < 1e-6,
        )

    def cos_vs_integration():
        rng = np.random.default_rng(3)
        worst, t_cos, t_int = 0.0, 0.0, 0.0
        n_sets = 10 if fast else 40
        for _ in range(n_sets):
            p = HestonParams(
                rng.uniform(0.005, 0.3),
                rng.uniform(0.3, 8),
                rng.uniform(0.01, 0.3),
                rng.uniform(0.1, 1.5),
                rng.uniform(-0.95, 0.3),
            )
            t = float(rng.choice([0.05, 0.25, 1.0, 3.0]))
            strikes = 100 * np.exp(np.linspace(-0.5, 0.35, 15) * max(np.sqrt(t), 0.3))
            s = time.perf_counter()
            cos = heston_price_cos(100, strikes, t, 0.96, p, False)
            t_cos += time.perf_counter() - s
            s = time.perf_counter()
            ref = np.array([heston_price_integration(100, k, t, 0.96, p, False) for k in strikes])
            t_int += time.perf_counter() - s
            worst = max(worst, float(np.max(np.abs(cos - ref))))
        return (
            f"Gil-Pelaez quadrature, {n_sets} random models x 15 strikes",
            f"COS {t_int / t_cos:.0f}x faster",
            f"{worst:.1e}",
            worst < 2e-5,
        )

    def heston_mc():
        res = mc_european_heston(
            100, 100, 1.0, 0.0, 0.0, FANG_OOSTERLEE_PARAMS, True, n_steps=100, n_paths=40_000 if fast else 200_000, seed=4
        )
        gap = abs(res.price - FANG_OOSTERLEE_PRICE)
        return (
            f"{FANG_OOSTERLEE_PRICE:.6f}",
            f"QE {res.price:.4f} +- {res.std_error:.4f}",
            f"{gap:.1e}",
            gap < 4 * res.std_error + 0.02,
        )

    def cv_efficiency():
        plain = mc_european_gbm(100, 100, 1, 0.05, 0.02, 0.25, True, 100_000, seed=5, control_variate=False)
        cv = mc_european_gbm(100, 100, 1, 0.05, 0.02, 0.25, True, 100_000, seed=5, control_variate=True)
        ref = float(bsm_price(100, 100, 1, 0.05, 0.02, 0.25, True))
        return (
            f"BS {ref:.4f}",
            f"{cv.price:.4f} +- {cv.std_error:.4f}",
            f"variance reduced {(plain.std_error / cv.std_error) ** 2:.1f}x",
            cv.contains(ref),
        )

    def heston_greeks():
        s, v, k, t, r, q = 100.0, 0.04, 105.0, 0.5, 0.03, 0.01
        p = HestonParams(v, 2.0, 0.05, 0.6, -0.7)
        sens = heston_cos_sensitivities(
            np.array([s]),
            np.array([v]),
            k,
            t,
            r,
            q,
            p.kappa,
            p.theta,
            p.sigma,
            p.rho,
            True,
            truncation=20.0,
            tail_tolerance=1e-14,
        )

        def price(spot, var):
            return heston_price_integration(
                spot * np.exp((r - q) * t), k, t, np.exp(-r * t), HestonParams(var, p.kappa, p.theta, p.sigma, p.rho)
            )

        fd_delta = (price(s + 1e-3, v) - price(s - 1e-3, v)) / 2e-3
        fd_vega = (price(s, v + 1e-6) - price(s, v - 1e-6)) / 2e-6
        err = max(abs(sens.delta[0] - fd_delta), abs(sens.dprice_dv[0] - fd_vega) / fd_vega)
        return (
            "finite differences of quadrature prices",
            f"delta {sens.delta[0]:.6f}, dC/dv {sens.dprice_dv[0]:.3f}",
            f"{err:.1e}",
            err < 1e-5,
        )

    def svi_vogt():
        g = (
            SVISlice(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153, maturity=1.0)
            .durrleman_g(np.linspace(-1.5, 1.5, 601))
            .min()
        )
        return "arbitrage present (Gatheral-Jacquier 2014)", f"min g(k) = {g:.4f}", "detected" if g < 0 else "missed", g < 0

    def calibration_recovery():
        truth = HestonParams(0.035, 2.2, 0.045, 0.75, -0.72)
        snap = build_snapshot(synthetic_chain(truth, iv_noise=0.0), 5000.0, datetime.now(timezone.utc), "SYNTH", "synthetic")
        res = calibrate_heston(snap)
        rel = np.abs(res.params.as_array() - truth.as_array()) / np.abs(truth.as_array())
        return (
            "synthetic surface from known parameters",
            f"RMSE {100 * res.rmse_vol:.3f} vol pts in {res.elapsed_seconds:.1f}s",
            f"max param error {100 * rel.max():.1f}%",
            rel.max() < 0.05,
        )

    def derman_kamal():
        exp = run_black_scholes_world(n_paths=10_000 if fast else 40_000, rebalances=(16, 64, 256))
        s = exp.summary()
        ratios = [row.std_pnl / exp.notes[f"derman_kamal_std_{row.rebalances}"] for row in s.itertuples()]
        worst = max(abs(x - 1) for x in ratios)
        return (
            "sqrt(pi/4) vega sigma / sqrt(N)",
            "ratios " + ", ".join(f"{x:.3f}" for x in ratios),
            f"{100 * worst:.1f}%",
            worst < 0.1,
        )

    add("Black-Scholes", "Call price", bs_hull)
    add("Black-Scholes", "Put-call parity, 5,000 random contracts", parity)
    add("Black-Scholes", "Analytic Greeks", greeks_fd)
    add("Implied vol", "Round trip, safeguarded Newton", iv_roundtrip)
    add("Lattice", "Leisen-Reimer vs CRR convergence", lr_tree)
    add("Lattice", "American put S=36, K=40", american_tree)
    add("Monte Carlo", "Longstaff-Schwartz American put", lsm)
    add("Monte Carlo", "Control variate on GBM", cv_efficiency)
    add("Heston", "COS method reference price", cos_reference)
    add("Heston", "Gil-Pelaez reference price", integration_reference)
    add("Heston", "COS vs quadrature, random models", cos_vs_integration)
    add("Heston", "Andersen QE Monte Carlo", heston_mc)
    add("Heston", "COS delta and variance sensitivity", heston_greeks)
    add("SVI", "Butterfly-arbitrage detection", svi_vogt)
    add("Calibration", "Parameter recovery", calibration_recovery)
    add("Hedging", "Discrete-hedging error vs theory", derman_kamal)

    return pd.DataFrame([c.__dict__ for c in checks])


def to_markdown(results: pd.DataFrame) -> str:
    results = results.assign(**{c: results[c].astype(str).str.replace("|", "/") for c in ("reference", "result", "error")})
    lines = [
        "| Area | Check | Reference | Result | Error | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in results.itertuples():
        status = "PASS" if row.passed else "FAIL"
        lines.append(f"| {row.area} | {row.check} | {row.reference} | {row.result} | {row.error} | {status} |")
    return "\n".join(lines)
