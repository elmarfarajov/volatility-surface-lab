"""Day 5 charts: Heston priced to the last digit, calibrated to the S&P 500, and simulated across the Feller boundary.

The pricing and simulation charts need only the code, mpmath and QuantLib; the
calibration charts use the local S&P 500 snapshot and are skipped where there is none.
"""

from __future__ import annotations

from functools import cache, lru_cache

import mpmath as mp
import numpy as np
import QuantLib as ql
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..heston import HestonParams, heston_price_cos, heston_price_integration
from ..market import chains
from ..market.cleaning import prepare
from ..smile.build import one_per_date
from ..stochvol.calibration import Calibration, Expiry, Identifiability, identifiability, model_vols, multistart, profile
from ..stochvol.characteristic import Heston, cf, moment_strip
from ..stochvol.pricing import _envelope, _nodes, carr_madan_fft, optimal_alpha, otm_price
from ..stochvol.simulation import exact_variance_cdf, otm_estimate, qe_variance, simulate
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

Array = NDArray[np.float64]
CASES = (
    ("Fang and Oosterlee's case, one year", Heston(0.0175, 1.5768, 0.0398, 0.5751, -0.5711), 1.0),
    ("Feller violated, one week", Heston(0.04, 1.0, 0.04, 1.0, -0.7), 7 / 365),
    ("long-dated, ten years", Heston(0.04, 0.5, 0.06, 0.8, -0.8), 10.0),
)
FELLER_HOLDS = Heston(0.04, 3.0, 0.04, 0.3, -0.7)
FELLER_FAILS = Heston(0.04, 1.0, 0.04, 1.0, -0.7)


def _mp_cf(u: mp.mpc, t: float, h: Heston) -> mp.mpc:
    iu = 1j * u
    beta = h.kappa - h.rho * h.sigma * iu
    d = mp.sqrt(beta**2 + h.sigma**2 * (iu + u * u))
    g = (beta - d) / (beta + d)
    e = mp.exp(-d * t)
    a = h.kappa * h.theta / h.sigma**2 * ((beta - d) * t - 2 * mp.log((1 - g * e) / (1 - g)))
    return mp.exp(a + (beta - d) / h.sigma**2 * (1 - e) / (1 - g * e) * h.v0)


@cache
def reference(k: float, t: float, h: Heston) -> float:
    """The out-of-the-money price over the forward from Lewis's integral at 40 digits."""
    with mp.workdps(40):
        km = mp.mpf(k)
        integral = mp.quad(
            lambda u: mp.re(mp.exp(-1j * u * km) * _mp_cf(u - 0.5j, t, h)) / (u * u + 0.25), [0, 1, 10, 50, 200, mp.inf]
        )
        call = 1 - mp.exp(km / 2) / mp.pi * integral
        return float(call if k >= 0 else call - (1 - mp.exp(km)))


def _quantlib(k: Array, t: float, h: Heston) -> Array:
    today = ql.Date(10, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    process = ql.HestonProcess(flat, flat, ql.QuoteHandle(ql.SimpleQuote(1.0)), h.v0, h.kappa, h.theta, h.sigma, h.rho)
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-14, 100000)
    expiry = today + int(round(t * 365))
    out = []
    for kk in k:
        side = ql.Option.Call if kk >= 0 else ql.Option.Put
        option = ql.VanillaOption(ql.PlainVanillaPayoff(side, float(np.exp(kk))), ql.EuropeanExercise(expiry))
        option.setPricingEngine(engine)
        out.append(option.NPV())
    return np.array(out)


def engine_accuracy() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Heston prices to the last digit, far into the wings",
        "Relative error of out-of-the-money prices against Lewis's integral evaluated to 40 digits. "
        "Engines that price one side and "
        "take the other from put-call parity lose their digits as the option goes out of the money; the contour integral prices "
        "each side directly and stays at rounding.",
    )
    for index, (label, h, t) in enumerate(CASES):
        ax = fig.add_axes((0.055 + index * 0.322, 0.15, 0.27, 0.6))
        spread = 4.0 * np.sqrt(max(h.theta * t, h.v0 * t))
        k = np.linspace(-1.5, 1.2, 19) * spread
        exact = np.array([reference(float(x), t, h) for x in k])
        params = HestonParams(h.v0, h.kappa, h.theta, h.sigma, h.rho)
        strikes, calls = np.exp(k), k >= 0
        fft_k, fft_c = carr_madan_fft(t, h)
        fft_call = np.interp(k, fft_k, fft_c)
        fft = np.where(calls, fft_call, fft_call - (1 - np.exp(k)))
        engines = (
            ("v1.0 COS (put + parity)", heston_price_cos(1.0, strikes, t, 1.0, params, calls), PALETTE["red"], 1.1),
            (
                "v1.0 Gil-Pelaez",
                np.array([heston_price_integration(1.0, s, t, 1.0, params, c) for s, c in zip(strikes, calls, strict=True)]),
                PALETTE["slate"],
                1.1,
            ),
            ("Carr-Madan FFT (interpolated)", fft, PALETTE["violet"], 1.1),
            ("QuantLib analytic", _quantlib(k, t, h), PALETTE["green"], 1.1),
            ("contour, out of the money", otm_price(k, t, h), PALETTE["blue"], 2.0),
        )
        for name, values, colour, width in engines:
            error = np.abs(values - exact) / exact
            ax.semilogy(k, np.clip(error, 1e-17, 10.0), "o-", ms=2.5, lw=width, color=colour, label=name)
        ax.axhline(np.finfo(float).eps, color=PALETTE["muted"], lw=0.7, ls=":")
        ax.set_ylim(1e-17, 30)
        tidy(ax, title=label, xlabel="log-moneyness ln(K/F)", ylabel="relative error" if index == 0 else None)
        ax.text(0.03, 0.04, f"smallest price {exact.min():.0e}", transform=ax.transAxes, fontsize=7.5)
        if index == 0:
            ax.legend(fontsize=7, loc="upper center")
    caption(
        fig, "Puts below the forward, calls above. Floored at 1e-17 for display; an error near 1 is a price of the wrong size."
    )
    return fig


def contour_damping() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Where the contour runs, and why the quadrature is graded",
        "The integrand for one put at the money, Fang and Oosterlee's case, for several dampings alpha. Near a pole - of the "
        "denominator as alpha nears -1, of the characteristic function at the edge of the moment strip - it peaks sharply at "
        "the origin; uniform panels then miss it, panels graded towards the origin do not.",
    )
    _, h, t = CASES[0]
    k = -0.08
    exact = reference(k, t, h)
    p_minus, _ = moment_strip(t, h)
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    u = np.linspace(0.0, 2.0, 801)
    for index, alpha in enumerate((-1.02, -1.2, -3.0, p_minus - 1.0 + 0.15)):
        env = _envelope(u, alpha, t, h) * np.exp(-alpha * k)
        ax.semilogy(u, env, color=SERIES[index], lw=1.5, label=f"alpha = {alpha:.2f}")
    tidy(ax, title="size of the integrand near the origin", xlabel="u", ylabel="|integrand|")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    alphas = np.linspace(-1.02, p_minus - 1.0 + 0.02, 40)
    for graded, colour, label in (
        (False, PALETTE["red"], "uniform panels"),
        (True, PALETTE["blue"], "graded towards the origin"),
    ):
        errors = []
        for alpha in alphas:
            top = 400.0
            nodes, weights = _nodes(top, 2.0) if graded else _uniform(top, 2.0)
            z = nodes - 1j * (alpha + 1.0)
            kernel = cf(z, t, h) / (alpha * alpha + alpha - nodes**2 + 1j * (2 * alpha + 1) * nodes)
            value = np.exp(-alpha * k) / np.pi * float((np.exp(-1j * k * nodes) * kernel).real @ weights)
            errors.append(abs(value - exact) / exact)
        ax2.semilogy(alphas, np.clip(errors, 1e-17, 10), "o-", ms=2.5, color=colour, lw=1.5, label=label)
    chosen = optimal_alpha(k, t, h)
    ax2.axvline(chosen, color=PALETTE["accent"], lw=1.4, label=f"chosen damping {chosen:.2f}")
    ax2.axvspan(p_minus - 1.0, p_minus - 1.0 + 0.1 * (-1.0 - (p_minus - 1.0)), color=PALETTE["band"])
    ax2.axvspan(-1.0 - 0.1 * (-1.0 - (p_minus - 1.0)), -1.0, color=PALETTE["band"])
    tidy(
        ax2, title="error of the put against 40 digits, by damping", xlabel="alpha (put side: below -1)", ylabel="relative error"
    )
    ax2.legend(fontsize=8, loc="upper center")
    caption(
        fig,
        f"Shaded: the 10% of the admissible interval at each end that the damping avoids - at the very edge (p = {p_minus:.3f}) "
        "even graded panels fail. The left panel's last damping sits 0.15 inside the edge.",
    )
    return fig


def _uniform(top: float, width: float) -> tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, top, int(np.ceil(top / width)) + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def simulation_bias() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Simulating Heston across the Feller boundary",
        "Monte Carlo error of the at-the-money option, in standard errors, against the number of "
        "time steps in a year. Euler with "
        "full truncation converges slowly, and when Feller's condition fails it is still ten standard errors out at 64 steps; "
        "Andersen's QE is within two standard errors from four steps a year.",
    )
    steps = (1, 2, 4, 8, 16, 32, 64)
    for index, (label, h) in enumerate(
        (("Feller holds: 2 kappa theta / sigma^2 = 2.67", FELLER_HOLDS), ("Feller fails: 0.08", FELLER_FAILS))
    ):
        ax = fig.add_axes((0.06 + index * 0.5, 0.15, 0.4, 0.6))
        k = np.array([0.0])
        exact = otm_price(k, 1.0, h)[0]
        for scheme, colour in (("euler", PALETTE["red"]), ("qe", PALETTE["blue"])):
            bias, error = [], []
            for n in steps:
                est = otm_estimate(simulate(h, 1.0, n, 200_000, scheme, seed=11), k)
                bias.append((est.value[0] - exact) / est.stderr[0])
                error.append(est.value[0] - exact)
            ax.plot(
                steps,
                bias,
                "o-",
                ms=4,
                lw=1.6,
                color=colour,
                label="Euler, full truncation" if scheme == "euler" else "QE (Andersen)",
            )
        ax.axhspan(-2, 2, color=PALETTE["band"], label="within two standard errors")
        ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
        ax.set_xscale("log", base=2)
        tidy(ax, title=label, xlabel="time steps in one year", ylabel="(MC - exact) / standard error" if index == 0 else None)
        ax.legend(fontsize=8)
    caption(
        fig,
        "200,000 paths per point; exact prices from the contour integral. "
        "The ATM call over the forward, v0 = theta = 0.04, rho = -0.7.",
    )
    return fig


def qe_transition() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "QE against the exact variance transition near zero",
        "One step of a quarter-year from v = 0.01 when Feller's condition fails: the exact transition is a scaled non-central "
        "chi-squared law with no atom, but with most of its mass within a hair of zero. QE replaces that with a mass at zero "
        "and an exponential tail, matching the first two moments; Euler with full truncation is a normal that goes negative.",
    )
    h, v, dt = FELLER_FAILS, 0.01, 0.25
    rng = np.random.default_rng(5)
    n = 400_000
    qe, *_ = qe_variance(np.full(n, v), dt, h, rng.standard_normal(n), rng.random(n))
    euler = v + h.kappa * (h.theta - v) * dt + h.sigma * np.sqrt(v * dt) * rng.standard_normal(n)
    x = np.linspace(0.0, 0.08, 400)
    exact = exact_variance_cdf(x, v, dt, h)
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax.plot(x, exact, color=PALETTE["ink"], lw=2.0, label="exact (non-central chi-squared)")
    ax.plot(x, np.searchsorted(np.sort(qe), x, side="right") / n, color=PALETTE["blue"], lw=1.4, label="QE")
    ax.plot(
        x,
        np.searchsorted(np.sort(euler), x, side="right") / n,
        color=PALETTE["red"],
        lw=1.2,
        label="Euler step (truncated at 0 for the next)",
    )
    tidy(ax, title="cumulative distribution of the next variance", xlabel="v after one step", ylabel="probability")
    ax.legend(fontsize=8, loc="lower right")
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    mean_exact = h.theta + (v - h.theta) * np.exp(-h.kappa * dt)
    tiny = np.geomspace(1e-14, 0.1, 400)
    ax2.semilogx(tiny, exact_variance_cdf(tiny, v, dt, h), color=PALETTE["ink"], lw=2.0, label="exact")
    ax2.semilogx(tiny, np.searchsorted(np.sort(qe), tiny, side="right") / n, color=PALETTE["blue"], lw=1.4, label="QE")
    ax2.semilogx(tiny, np.searchsorted(np.sort(euler), tiny, side="right") / n, color=PALETTE["red"], lw=1.2, label="Euler")
    tidy(ax2, title="the same, on a log scale: the mass near zero", xlabel="v after one step", ylabel="probability")
    ax2.legend(fontsize=8, loc="lower right")
    gap = float(np.max(np.abs(np.searchsorted(np.sort(qe), x[1:], side="right") / n - exact[1:])))
    caption(
        fig,
        f"Largest gap between the QE and exact distribution functions above zero: {gap:.3f}; Euler "
        f"puts {np.mean(euler < 0):.0%} of paths below zero. "
        f"Exact mean {mean_exact:.5f}, QE {qe.mean():.5f}.",
    )
    return fig


def moment_explosion() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "The moment strip, and the smile wings it implies",
        "E[S_T^p] is finite only for p inside a strip that narrows with maturity (Andersen and Piterbarg, 2007). Through Lee's "
        "moment formula its edges fix the slopes of the implied variance wings - so the model's wings are not a free choice.",
    )
    ts = np.geomspace(0.05, 10.0, 40)
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    for index, (label, h) in enumerate(
        (("Feller holds", FELLER_HOLDS), ("Feller fails", FELLER_FAILS), ("Fang-Oosterlee", CASES[0][1]))
    ):
        edges = np.array([moment_strip(t, h) for t in ts])
        colour = SERIES[index]
        ax.plot(ts, edges[:, 1], color=colour, lw=1.6, label=f"{label}: upper edge")
        ax.plot(ts, edges[:, 0], color=colour, lw=1.6, ls="--")

        def slope(p: Array) -> Array:
            return np.asarray(2.0 - 4.0 * (np.sqrt(p * p + p) - p), dtype=np.float64)

        ax2.plot(ts, slope(edges[:, 1] - 1.0), color=colour, lw=1.6, label=f"{label}: call wing")
        ax2.plot(ts, slope(-edges[:, 0]), color=colour, lw=1.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    tidy(ax, title="edges of the moment strip (dashed: lower)", xlabel="years to expiry", ylabel="p")
    ax.legend(fontsize=7.5)
    ax2.axhline(2.0, color=PALETTE["accent"], lw=1.2, label="Lee's bound, 2")
    ax2.set_xscale("log")
    tidy(ax2, title="wing slopes of total variance (dashed: put wing)", xlabel="years to expiry", ylabel="slope")
    ax2.legend(fontsize=7.5)
    caption(fig, "Lee (2004): slope = 2 - 4 (sqrt(p^2 + p) - p) with p = p_plus - 1 for calls and p = -p_minus for puts.")
    return fig


# ---------------------------------------------------------------- calibration to the S&P 500


@lru_cache(maxsize=1)
def _spx_expiries() -> tuple[str, list[Expiry]]:
    market = prepare(chains.load("^SPX"))
    chosen: list[Expiry] = []
    for e in one_per_date(market):
        if not 25 / 365 <= e.maturity <= 2.2 or (chosen and e.maturity < chosen[-1].maturity * 1.25):
            continue
        q = e.quotes
        z = q["k"] / np.sqrt(e.maturity)
        q = q[(z > -2.5) & (z < 1.0)]
        q = q.iloc[np.unique(np.linspace(0, len(q) - 1, min(len(q), 15)).astype(int))]
        width = np.clip((q["iv_ask"] - q["iv_bid"]).to_numpy(float), 0.002, None)
        chosen.append(Expiry(e.maturity, q["k"].to_numpy(float), q["iv_mid"].to_numpy(float), width))
    return market.chain.as_of.strftime("%d %B %Y"), chosen


@lru_cache(maxsize=1)
def _spx_calibration() -> tuple[list[Calibration], Identifiability, list[Calibration], Array]:
    _, expiries = _spx_expiries()
    fits = multistart(expiries, 6)
    kappas = np.geomspace(0.3, 10.0, 6)
    return fits, identifiability(expiries, fits[0].heston), profile(expiries, fits[0].heston, kappas), kappas


def heston_calibration() -> Figure:
    day, expiries = _spx_expiries()
    fits, _, _, _ = _spx_calibration()
    best = fits[0]
    h = best.heston
    fig = new_figure(14.0, 8.8)
    title_block(
        fig,
        "Heston calibrated to the S&P 500",
        f"{day}: {len(expiries)} expiries from a month to two years, weighted by bid-ask width. Five parameters cannot follow "
        "a short-dated skew this steep and a long-dated one this flat at once.",
    )
    picks = np.unique(np.linspace(0, len(expiries) - 1, 6).astype(int))
    vols = model_vols(h, expiries)
    for index, i in enumerate(picks):
        e = expiries[i]
        row, col = divmod(index, 3)
        ax = fig.add_axes((0.055 + col * 0.322, 0.53 - row * 0.41, 0.27, 0.3))
        ax.vlines(e.k, 100 * (e.iv - e.width / 2), 100 * (e.iv + e.width / 2), color=PALETTE["slate"], lw=1.4)
        ax.plot(e.k, 100 * e.iv, "o", ms=3.5, color=PALETTE["slate"], label="market mid, bid to ask")
        grid = np.linspace(e.k.min(), e.k.max(), 120)
        model = model_vols(h, [Expiry(e.maturity, grid, np.zeros_like(grid), np.ones_like(grid))])[0]
        ax.plot(grid, 100 * model, color=PALETTE["blue"], lw=1.6, label="Heston")
        rmse = 100 * float(np.sqrt(np.mean((vols[i] - e.iv) ** 2)))
        tidy(
            ax,
            title=f"{e.maturity * 365:.0f} days: error {rmse:.2f} vol pts",
            xlabel="ln(K/F)" if row else None,
            ylabel="implied vol (%)" if col == 0 else None,
        )
        if index == 0:
            ax.legend(fontsize=7.5)
    caption(
        fig,
        f"v0 = {h.v0:.4f}, kappa = {h.kappa:.2f}, theta = {h.theta:.4f}, sigma = {h.sigma:.2f}, rho = {h.rho:.2f}; Feller ratio "
        f"{h.feller:.2f}. Root-mean-square error {100 * best.rmse:.2f} vol pts, {best.weighted:.1f} bid-ask widths.",
    )
    return fig


def heston_identifiability() -> Figure:
    fits, ident, prof, kappas = _spx_calibration()
    weighted = np.array([p.weighted for p in prof])
    near = np.argsort(weighted)[:2]  # the two best profile points
    lo, hi = sorted(near)
    rise = 100 * (weighted[near].max() / weighted[near].min() - 1)
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "What the S&P 500 surface pins down, and what it does not",
        "Every start of a scrambled Sobol' design reached the same minimum and the Jacobian is well conditioned. The weakest "
        f"direction is mean reversion against long-run variance: kappa from {kappas[lo]:.1f} to "
        f"{kappas[hi]:.1f} changes the error "
        f"by {rise:.0f}% while theta moves from {prof[lo].heston.theta:.3f} to {prof[hi].heston.theta:.3f} to compensate.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.27, 0.6))
    ax.plot(kappas, weighted, "o-", ms=4, color=PALETTE["blue"], lw=1.6)
    ax.axvline(fits[0].heston.kappa, color=PALETTE["accent"], lw=1.2, label=f"best kappa {fits[0].heston.kappa:.2f}")
    ax.set_xscale("log")
    tidy(ax, title="error with kappa held fixed", xlabel="kappa", ylabel="RMS residual (bid-ask widths)")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.40, 0.15, 0.25, 0.6))
    ax2.plot(kappas, [p.heston.theta for p in prof], "o-", ms=4, color=PALETTE["green"], label="theta")
    ax2.plot(kappas, [p.heston.v0 for p in prof], "o-", ms=4, color=PALETTE["violet"], label="v0")
    ax2b = ax2.twinx()
    ax2b.plot(kappas, [p.heston.sigma for p in prof], "s--", ms=3, color=PALETTE["slate"], label="sigma (right)")
    ax2.set_xscale("log")
    tidy(ax2, title="how the others compensate", xlabel="kappa", ylabel="variance")
    ax2.legend(fontsize=8, loc="upper right")
    ax2b.legend(fontsize=8, loc="center right")
    ax2b.grid(False)
    ax3 = fig.add_axes((0.74, 0.15, 0.23, 0.6))
    names = ["ln v0", "ln kappa", "ln theta", "ln sigma", "rho"]
    biggest = float(np.max(np.abs(ident.weakest)))
    colours = [PALETTE["accent"] if abs(v) == biggest else PALETTE["slate"] for v in ident.weakest]
    ax3.barh(names, ident.weakest, color=colours)
    ax3.invert_yaxis()
    ax3.axvline(0.0, color=PALETTE["ink"], lw=0.7)
    tidy(ax3, title="the least determined direction", xlabel="component")
    spread = max(abs(f.heston.kappa - fits[0].heston.kappa) for f in fits)
    caption(
        fig,
        f"{len(fits)} starts, largest distance in kappa between their minima {spread:.1e}. Singular "
        f"values of the weighted Jacobian "
        f"in log-parameters: {', '.join(f'{v:.0f}' for v in ident.singular_values)} (condition number {ident.condition:.0f}).",
    )
    return fig
