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
    cache: dict[str, object] = {}

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
        # the reference used to be Longstaff-Schwartz's 4.478, which is the Bermudan price (numerics.references)
        value = binomial_price(36, 40, 1, 0.06, 0, 0.2, False, True, steps=2001)
        reference = 4.486674
        return (
            f"{reference} (Andersen-Lake-Offengenden 2016)",
            f"{value:.4f}",
            f"{abs(value - reference):.1e}",
            abs(value - reference) < 1e-3,
        )

    def lsm():
        # 50 exercise dates a year: a Bermudan option, whose price is 4.4778 (Crank-Nicolson, numerics.references)
        res = lsm_american_price(36, 40, 1, 0.06, 0, 0.2, False, 50, 50_000 if fast else 200_000, seed=11)
        reference = 4.477792
        return (
            f"{reference} (Bermudan, 50 dates)",
            f"{res.price:.4f} +- {res.std_error:.4f}",
            f"{abs(res.price - reference):.1e}",
            abs(res.price - reference) < 4 * res.std_error + 0.005,
        )

    def bbsr_reference():
        from .numerics.lattice import Contract, bbsr
        from .numerics.references import LONGSTAFF_SCHWARTZ_TABLE

        rows = LONGSTAFF_SCHWARTZ_TABLE[::4] if fast else LONGSTAFF_SCHWARTZ_TABLE
        worst = max(
            abs(bbsr(Contract(r.spot, r.strike, r.maturity, r.rate, 0.0, r.vol, False), 1000, "crr") - r.american) for r in rows
        )
        return f"{len(rows)} American puts, Andersen-Lake-Offengenden", "BBSR, 1,000 / 2,000 steps", f"{worst:.1e}", worst < 2e-4

    def crank_nicolson_american():
        from .numerics.lattice import Contract
        from .numerics.pde import solve

        r0 = 4.486674
        value = solve(Contract(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False), 1600, 1600, exercise="american").price
        return (
            f"{r0} (Andersen-Lake-Offengenden)",
            "Crank-Nicolson + Rannacher, 1,600 x 1,600",
            f"{abs(value - r0):.1e}",
            abs(value - r0) < 5e-4,
        )

    def longstaff_schwartz_table():
        from .numerics.references import LONGSTAFF_SCHWARTZ_TABLE

        bermudan = sum(abs(r.longstaff_schwartz - r.bermudan_50) < 5e-4 for r in LONGSTAFF_SCHWARTZ_TABLE)
        american = sum(abs(r.longstaff_schwartz - r.american) < 5e-4 for r in LONGSTAFF_SCHWARTZ_TABLE)
        return (
            "Longstaff-Schwartz (2001) Table 1, finite differences",
            f"{bermudan}/20 match the Bermudan price, {american}/20 the American",
            "a Bermudan column",
            bermudan >= 15,
        )

    def lsm_bounds():
        from .numerics.lattice import Contract
        from .numerics.montecarlo import longstaff_schwartz

        res = longstaff_schwartz(
            Contract(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False),
            [k / 50 for k in range(1, 51)],
            40_000 if fast else 100_000,
            40_000 if fast else 100_000,
            dual_paths=1_000 if fast else 4_000,
            inner_paths=100 if fast else 200,
            seed=3,
        )
        truth = 4.477792
        inside = res.lower.price - 3 * res.lower.std_error <= truth <= res.upper.price + 3 * res.upper.std_error
        return (
            f"{truth} (Bermudan, 50 dates)",
            f"[{res.lower.price:.4f}, {res.upper.price:.4f}]",
            f"gap {res.gap:.4f}",
            inside and res.gap < 0.02,
        )

    def sobol():
        from .numerics.lattice import Contract
        from .numerics.montecarlo import european

        c = Contract(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)
        pseudo, quasi = european(c, 2**18, "pseudo", seed=1), european(c, 2**18, "sobol", seed=1)
        return (
            "Black-Scholes",
            f"std error {quasi.std_error:.1e} vs {pseudo.std_error:.1e}",
            f"{(pseudo.std_error / quasi.std_error) ** 2:.0f}x fewer samples",
            abs(quasi.price - c.european()) < 4 * quasi.std_error,
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

    def haug():
        from .analytic import greeks
        from .analytic.published import HAUG_2007

        misses = [
            e.title
            for e in HAUG_2007
            if round(float(getattr(greeks(e.spot, e.strike, e.maturity, e.rate, e.yield_, e.vol, e.is_call), e.quantity)), 4)
            != e.published
        ]
        return (
            f"{len(HAUG_2007)} worked examples (Haug 2007)",
            f"{len(HAUG_2007) - len(misses)} reproduced",
            ", ".join(misses) or "none",
            not misses,
        )

    def mpmath_greeks():
        from .analytic import greeks
        from .analytic.reference import DEFINITIONS, reference_greeks

        rng = np.random.default_rng(5)
        worst = 0.0
        for _ in range(4 if fast else 12):
            contract = (*rng.uniform([50, 50, 0.05, -0.02, 0.0, 0.08], [150, 150, 3, 0.1, 0.06, 0.8]), bool(rng.random() < 0.5))
            ours, ref = greeks(*contract).as_dict(), reference_greeks(*contract)
            worst = max(worst, max(abs(float(ours[n]) - ref[n]) / max(abs(ref[n]), 1e-12) for n in DEFINITIONS))
        return "mpmath, 50 digits", f"{len(DEFINITIONS)} Greeks to third order", f"{worst:.1e}", worst < 1e-11

    def pde():
        from .analytic import greeks

        rng = np.random.default_rng(6)
        s, k = rng.uniform(5, 500, 20_000), rng.uniform(5, 500, 20_000)
        t, r, q, v = (
            rng.uniform(0.01, 10, 20_000),
            rng.uniform(-0.03, 0.15, 20_000),
            rng.uniform(-0.02, 0.1, 20_000),
            rng.uniform(0.02, 2, 20_000),
        )
        g = greeks(s, k, t, r, q, v, True)
        terms = np.stack([g.theta, (r - q) * s * g.delta, 0.5 * v**2 * s**2 * g.gamma, -r * g.price])
        worst = float(np.max(np.abs(terms.sum(axis=0)) / np.maximum(np.abs(terms).max(axis=0), 1e-300)))
        return "dV/dt + (r-q)S V_S + sigma^2 S^2 V_SS / 2 - rV = 0", "20,000 random contracts", f"{worst:.1e}", worst < 1e-10

    def black_tails():
        import mpmath as mp

        from .market.black import log_otm_value

        rng = np.random.default_rng(12)
        n = 300 if fast else 1500
        x, s = -(10 ** rng.uniform(-6, 1.2, n)), 10 ** rng.uniform(-3, 1.2, n)
        worst = 0.0
        for xi, si, lb in zip(x, s, log_otm_value(x, s), strict=True):
            with mp.workdps(60):
                xm, sm = mp.mpf(float(xi)), mp.mpf(float(si))  # divide in 60 digits, not in double precision
                exact = mp.log(mp.ncdf(xm / sm + sm / 2) * mp.exp(xm / 2) - mp.ncdf(xm / sm - sm / 2) * mp.exp(-xm / 2))
            worst = max(worst, abs(lb - float(exact)) / max(1.0, abs(float(exact))))
        return "mpmath, 60 digits", f"{n:,} prices down to 1e-2000 (ln b)", f"{worst:.1e} (relative, ln b)", worst < 1e-15

    def iv_conditioning():
        import mpmath as mp

        from .market.black import log_vega
        from .market.implied import invert_normalised

        rng = np.random.default_rng(13)
        n = 600 if fast else 3000
        x, s = rng.uniform(-6, 6, n), 10 ** rng.uniform(-3, 1, n)
        call = rng.random(n) < 0.5
        beta = np.empty(n)
        for i, (xi, si, ci) in enumerate(zip(x, s, call, strict=True)):
            with mp.workdps(50):
                xs, sm = mp.mpf(float(xi if ci else -xi)), mp.mpf(float(si))
                beta[i] = float(mp.ncdf(xs / sm + sm / 2) * mp.exp(xs / 2) - mp.ncdf(xs / sm - sm / 2) * mp.exp(-xs / 2))
        eps = np.finfo(float).eps
        with np.errstate(divide="ignore", over="ignore"):
            kappa = eps * np.maximum(beta, 1e-300) / (s * np.exp(log_vega(-np.abs(x), s))) + eps
        ok = (beta > 0) & (kappa < 1e-3)
        result = invert_normalised(beta[ok], x[ok], call[ok])
        ratio = float(np.max(np.abs(result.total_vol - s[ok]) / s[ok] / kappa[ok]))
        return (
            "50-digit prices; error / condition number",
            f"{ok.sum():,} quotes, at most {result.iterations.max()} iterations",
            f"{ratio:.1f} x conditioning",
            ratio < 8.0 and not np.isnan(result.total_vol).any(),
        )

    def iv_tiny_prices():
        from .market.black import log_otm_value
        from .market.implied import invert_normalised

        xx, ss = np.meshgrid(-np.linspace(0.05, 4.0, 80), np.geomspace(0.003, 0.2, 80))
        beta = np.exp(log_otm_value(xx, ss))
        tiny = (beta > 1e-300) & (beta < 1e-10)
        worst = float(np.max(np.abs(invert_normalised(beta[tiny], xx[tiny]).total_vol - ss[tiny]) / ss[tiny]))
        return "exact total volatility", f"{tiny.sum():,} prices, 1e-300..1e-10 (v1.0: 400% off)", f"{worst:.1e}", worst < 1e-12

    def parity_recovery():
        from datetime import datetime, timezone

        from .market.chains import Chain
        from .market.cleaning import prepare

        params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
        raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=(44, 107, 261, 534))
        quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
        chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
        market = prepare(chain)
        forward_error = max(abs(s.forward / (5000.0 * np.exp(0.027 * s.maturity)) - 1) for s in market.slices)
        rate_error = max(abs(float(market.curve.rate(s.maturity)) - 0.04) for s in market.slices)
        return (
            "F = S e^{(r-q)T}, r = 4%",
            f"{len(market.slices)} expiries, {market.funnel()['kept']:,} quotes kept",
            f"F {forward_error:.1e} (rel); r {rate_error * 1e4:.1f} bp",
            forward_error < 1e-4 and rate_error < 5e-4,
        )

    def _synthetic_surface():
        from datetime import datetime, timezone

        from .market.chains import Chain
        from .market.cleaning import prepare
        from .smile.build import build_surface

        if "surface" not in cache:
            params = HestonParams(v0=0.03, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
            days = (23, 72, 170, 352) if fast else (9, 23, 44, 72, 107, 170, 261, 352, 534, 716)
            raw = synthetic_chain(params, spot=5000.0, rate=0.04, dividend_yield=0.013, maturities_days=days)
            quotes = raw.assign(root="SYN", is_call=raw["option_type"].eq("C"), last_trade=pd.Timestamp("2026-10-09", tz="UTC"))
            chain = Chain("SYN", 5000.0, datetime(2026, 10, 9, 20, tzinfo=timezone.utc), quotes.assign(vendor_iv=np.nan))
            cache["surface"] = build_surface(prepare(chain))
        return cache["surface"]

    def vogt_repair():
        from .smile.fit import fit_slice
        from .smile.ssvi import Quotes, fit_essvi
        from .smile.svi import RawSVI, butterfly_check

        vogt = RawSVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
        k = np.linspace(-1.5, 1.5, 31)
        quotes = Quotes(1.0, k, vogt.implied_vol(k, 1.0), np.ones(k.size))
        repaired = fit_slice(quotes, fit_essvi([quotes])[0].raw()).slice
        before, after = butterfly_check(vogt), butterfly_check(repaired)
        move = float(np.max(np.abs(repaired.implied_vol(k, 1.0) - quotes.iv)))
        return (
            "min g -0.0329 (Gatheral-Jacquier 2014)",
            f"found {before.worst:.4f}; repaired min g {after.worst:+.1e}",
            f"{100 * move:.2f} vol pts moved",
            abs(before.worst + 0.0329) < 5e-4 and after.passed,
        )

    def ssvi_theorem():
        from .smile.ssvi import SSVI
        from .smile.svi import butterfly_check, calendar_check

        rng = np.random.default_rng(14)
        failures, n = 0, 40 if fast else 200
        for _ in range(n):
            rho = rng.uniform(-0.95, 0.95)
            surface = SSVI(
                rho,
                rng.uniform(0.05, 1.0) * 2.0 / (1.0 + abs(rho)),
                rng.uniform(0.01, 0.5),
                np.arange(1.0, 6.0),
                np.cumsum(rng.uniform(1e-4, 0.05, 5)),
            )
            raw = [s.raw() for s in surface.slices()]
            failures += sum(not butterfly_check(r).passed for r in raw)
            failures += sum(not calendar_check(a, b).passed for a, b in zip(raw[:-1], raw[1:], strict=True))
        return (
            "no arbitrage if eta(1+|rho|) <= 2, gamma <= 1/2",
            f"{n} random surfaces, whole-line checks",
            f"{failures} failures",
            failures == 0,
        )

    def constrained_fit():
        fit = _synthetic_surface()
        rmse = max(f.rmse for f in fit.svi)
        bfly = min(f.butterfly_margin for f in fit.svi)
        cal = min(f.calendar_margin for f in fit.svi)
        return (
            "Heston chain; g >= 0 and w rising, whole line",
            f"{len(fit.svi)} slices; min g {bfly:.1e}, min gap {cal:.1e}",
            f"worst RMSE {100 * rmse:.2f} vol pts",
            bfly >= 0 and cal >= 0 and rmse < 0.01,
        )

    def densities():
        from .smile.density import moments

        stats = [moments(f.slice) for f in _synthetic_surface().svi]
        mass = max(abs(m.mass - 1) for m in stats)
        mean = max(abs(m.forward - 1) for m in stats)
        bl = max(m.breeden_litzenberger for m in stats)
        return (
            "mass 1, E[S_T] = F, Breeden-Litzenberger",
            f"{len(stats)} slices",
            f"{mass:.0e}; {mean:.0e}; BL {bl:.0e}",
            mass < 1e-8 and mean < 1e-8 and bl < 1e-4,
        )

    def price_interpolation():
        surface = _synthetic_surface().surface
        k = np.linspace(-0.8, 0.5, 651)
        h = k[1] - k[0]
        worst, monotone, previous = np.inf, np.inf, None
        for t in np.linspace(0.01, surface.maturities[-1] * 1.5, 60 if fast else 200):
            c = surface.call(k, t)
            worst = min(
                worst, float(np.min(np.exp(-k[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2 * h))))
            )
            if previous is not None:
                monotone = min(monotone, float(np.min(c - previous)))
            previous = c
        return (
            "density >= 0, calls rising in t",
            "between and beyond expiries",
            f"min density {worst:.1e}; min dC {monotone:.1e}",
            worst > -1e-9 and monotone > -1e-12,
        )

    add("Analytic", "Published examples", haug)
    add("Analytic", "Seventeen Greeks vs arbitrary precision", mpmath_greeks)
    add("Analytic", "Black-Scholes PDE from the Greeks", pde)
    add("Black-Scholes", "Call price", bs_hull)
    add("Black-Scholes", "Put-call parity, 5,000 random contracts", parity)
    add("Black-Scholes", "Analytic Greeks", greeks_fd)
    add("Implied vol", "Round trip, 20,000 random quotes", iv_roundtrip)
    add("Implied vol", "Normalised Black in the far tails", black_tails)
    add("Implied vol", "Inversion vs its conditioning", iv_conditioning)
    add("Implied vol", "Prices from 1e-300 to 1e-10", iv_tiny_prices)
    add("Market", "Forward and rate from parity", parity_recovery)
    add("Lattice", "Leisen-Reimer vs CRR convergence", lr_tree)
    add("Lattice", "American put S=36, K=40", american_tree)
    add("Lattice", "BBSR on the American references", bbsr_reference)
    add("Finite differences", "American put, Crank-Nicolson", crank_nicolson_american)
    add("References", "What Longstaff-Schwartz's column is", longstaff_schwartz_table)
    add("Monte Carlo", "Longstaff-Schwartz, in-sample (Bermudan)", lsm)
    add("Monte Carlo", "Lower and dual upper bound", lsm_bounds)
    add("Monte Carlo", "Scrambled Sobol' vs pseudo-random", sobol)
    add("Monte Carlo", "Control variate on GBM", cv_efficiency)
    add("Heston", "COS method reference price", cos_reference)
    add("Heston", "Gil-Pelaez reference price", integration_reference)
    add("Heston", "COS vs quadrature, random models", cos_vs_integration)
    add("Heston", "Andersen QE Monte Carlo", heston_mc)
    add("Heston", "COS delta and variance sensitivity", heston_greeks)
    add("SVI", "Butterfly-arbitrage detection", svi_vogt)
    add("Surface", "Vogt's slice found and repaired", vogt_repair)
    add("Surface", "SSVI theorem, checked on the whole line", ssvi_theorem)
    add("Surface", "Constrained SVI on a Heston chain", constrained_fit)
    add("Surface", "Risk-neutral densities", densities)
    add("Surface", "Price interpolation in time", price_interpolation)
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
