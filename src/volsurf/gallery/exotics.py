"""Day 7 charts: exotic options in closed form, by Monte Carlo with corrections, and across the models of one surface.

The barrier, Asian and lookback charts need only the code and QuantLib; the digital and
model-risk charts are drawn from the local S&P 500 snapshot and skipped where there is none.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import QuantLib as ql
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..exotics import closed_form as cf
from ..exotics.model_risk import Book, price_book
from ..exotics.monte_carlo import arithmetic_asian, barrier_mc, bgk_barrier, black_scholes_paths
from ..localvol.dynamics import monthly
from ..market import chains
from ..market.black import black_price
from ..market.cleaning import prepare
from ..smile.build import SurfaceFit, build_surface
from ..stochvol.characteristic import Heston
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

Array = NDArray[np.float64]
S, T, R, B, SIGMA = 100.0, 1.0, 0.05, 0.03, 0.25


@lru_cache(maxsize=1)
def _fit() -> SurfaceFit:
    return build_surface(prepare(chains.load("^SPX")))


@lru_cache(maxsize=1)
def _day() -> str:
    return chains.load("^SPX").as_of.strftime("%d %B %Y")


def barrier_monitoring() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Barriers: watched continuously or once a day",
        "A down-and-out and an up-and-out call (strike 100, one year, volatility 25%). Monitored on "
        "a grid, the barrier is crossed "
        "less often and the knock-out is worth more. Broadie, Glasserman and Kou's shifted barrier recovers the discrete price "
        "except when monitoring is coarse and the barrier near; the Brownian bridge recovers the continuous price from any grid.",
    )
    grids = (4, 12, 26, 52, 126, 252)
    for index, (kind, h, label) in enumerate(
        (("down-out", 85.0, "down-and-out call, barrier 85"), ("up-out", 120.0, "up-and-out call, barrier 120"))
    ):
        ax = fig.add_axes((0.06 + index * 0.5, 0.15, 0.4, 0.6))
        continuous = cf.barrier(kind, S, 100.0, h, T, R, B, SIGMA, True)  # type: ignore[arg-type]
        discrete, bridge, corrected = [], [], []
        for m in grids:
            discrete.append(barrier_mc(kind, S, 100.0, h, T, R, B, SIGMA, True, m, 200_000, False, seed=5))
            bridge.append(barrier_mc(kind, S, 100.0, h, T, R, B, SIGMA, True, m, 200_000, True, seed=5))
            corrected.append(bgk_barrier(kind, S, 100.0, h, T, R, B, SIGMA, True, m))
        ax.axhline(continuous, color=PALETTE["ink"], lw=1.2, label=f"continuous, closed form {continuous:.3f}")
        ax.errorbar(
            grids,
            [d.value for d in discrete],
            yerr=[2 * d.stderr for d in discrete],
            fmt="o-",
            ms=4,
            color=PALETTE["red"],
            capsize=2,
            label="discrete monitoring, Monte Carlo",
        )
        ax.plot(grids, corrected, "s--", ms=4, color=PALETTE["accent"], label="closed form, shifted barrier (BGK)")
        ax.errorbar(
            grids,
            [d.value for d in bridge],
            yerr=[2 * d.stderr for d in bridge],
            fmt="o-",
            ms=4,
            color=PALETTE["blue"],
            capsize=2,
            label="Brownian bridge on that grid",
        )
        ax.set_xscale("log")
        tidy(ax, title=label, xlabel="monitoring dates in the year", ylabel="price" if index == 0 else None)
        ax.legend(fontsize=7.5)
    caption(
        fig,
        "200,000 antithetic paths per point; bars are two standard errors. Rate 5%, carry 3%. BGK: "
        "barrier moved away from spot by exp(0.5826 sigma sqrt(dt)).",
    )
    return fig


def _ql_process(r: float, q: float, sigma: float) -> ql.BlackScholesMertonProcess:
    today = ql.Date(14, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)),
    )


def asians() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Arithmetic Asians: a control variate, and a reference that is not",
        "The geometric average has a closed form and moves almost one-for-one with the arithmetic one, so it removes nearly all "
        "of the Monte Carlo variance. With it, two independent Monte Carlo engines agree to their noise - and QuantLib's "
        "Choi engine, converged in its parameters, is right at four fixings and 0.4% low at fifty-two.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    for index, sigma in enumerate((0.1, 0.25, 0.5)):
        ratios = [arithmetic_asian(S, k, 364 / 365, R, B, sigma, 52, True, 100_000, seed=9).variance_ratio for k in strikes]
        ax.semilogy(strikes, ratios, "o-", ms=4, color=SERIES[index], lw=1.6, label=f"volatility {sigma:.0%}")
    tidy(
        ax,
        title="variance removed by the geometric control (52 weekly fixings)",
        xlabel="strike",
        ylabel="plain variance / controlled variance",
    )
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    process = _ql_process(R, R - B, SIGMA)
    today = ql.Date(14, 10, 2026)
    counts = (4, 7, 13, 26, 52)
    choi_gap, tw_gap, errors = [], [], []
    for n in counts:
        step = 364 // n
        dates = [today + step * (i + 1) for i in range(n)]
        option = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic,
            0.0,
            0,
            dates,
            ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
            ql.EuropeanExercise(today + step * n),
        )
        option.setPricingEngine(ql.ChoiAsianEngine(process, 60, 150))
        choi = option.NPV()
        option.setPricingEngine(ql.TurnbullWakemanAsianEngine(process))
        tw = option.NPV()
        mc = arithmetic_asian(S, 100.0, step * n / 365, R, B, SIGMA, n, True, 400_000, seed=11).controlled
        choi_gap.append((choi - mc.value) / mc.value)
        tw_gap.append((tw - mc.value) / mc.value)
        errors.append(2 * mc.stderr / mc.value)
    ax2.plot(counts, 100 * np.array(choi_gap), "o-", ms=4, color=PALETTE["red"], lw=1.6, label="QuantLib Choi engine (60, 150)")
    ax2.plot(counts, 100 * np.array(tw_gap), "s-", ms=4, color=PALETTE["violet"], lw=1.4, label="Turnbull-Wakeman approximation")
    ax2.fill_between(
        counts, -100 * np.array(errors), 100 * np.array(errors), color=PALETTE["band"], label="Monte Carlo, two standard errors"
    )
    ax2.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    ax2.set_xscale("log")
    tidy(
        ax2, title="against Monte Carlo with the control variate", xlabel="fixings in the year", ylabel="difference (% of price)"
    )
    ax2.legend(fontsize=8)
    caption(
        fig,
        "At-the-money call, one year, rate 5%, carry 3%, volatility 25%. QuantLib's own Monte Carlo "
        "engine with a control variate agrees with this one to two standard errors.",
    )
    return fig


def lookbacks() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Lookbacks: the extreme seen once a day is not the extreme",
        "A floating-strike lookback call pays the final price less the lowest seen. Monitored on a grid, the lowest is missed by "
        "about sigma sqrt(dt) times 0.58, so the discrete option is worth less, and converges to Goldman, Sosin and Gatto's "
        "closed form only as the square root of the grid spacing.",
    )
    exact_call = cf.floating_lookback(S, S, T, R, B, SIGMA, True)
    exact_put = cf.floating_lookback(S, S, T, R, B, SIGMA, False)
    grids = np.array([4, 12, 52, 252, 1000])
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    gaps_call, gaps_put = [], []
    for m in grids:
        paths = black_scholes_paths(S, T, R, B, SIGMA, int(m), 100_000, seed=13)
        low = np.minimum(paths.min(axis=1), S)
        high = np.maximum(paths.max(axis=1), S)
        disc = np.exp(-R * T)
        gaps_call.append(exact_call - disc * np.mean(paths[:, -1] - low))
        gaps_put.append(exact_put - disc * np.mean(high - paths[:, -1]))
    ax.loglog(grids, gaps_call, "o-", ms=4, color=PALETTE["blue"], lw=1.6, label="floating-strike call")
    ax.loglog(grids, gaps_put, "o-", ms=4, color=PALETTE["green"], lw=1.6, label="floating-strike put")
    reference = gaps_call[2] * np.sqrt(grids[2] / grids)
    ax.loglog(grids, reference, "--", color=PALETTE["muted"], lw=1.0, label="slope -1/2")
    tidy(ax, title="continuous minus discrete price", xlabel="monitoring dates in the year", ylabel="price gap")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    vols = np.linspace(0.1, 0.6, 11)
    ax2.plot(
        100 * vols,
        [cf.floating_lookback(S, S, T, R, B, v, True) for v in vols],
        color=PALETTE["blue"],
        lw=1.6,
        label="floating-strike call",
    )
    ax2.plot(
        100 * vols,
        [cf.fixed_lookback(S, 100.0, S, T, R, B, v, True) for v in vols],
        color=PALETTE["violet"],
        lw=1.6,
        label="fixed-strike call, strike 100",
    )
    ax2.plot(
        100 * vols,
        [cf.black_scholes(S, 100.0, T, R, B, v, True) for v in vols],
        color=PALETTE["slate"],
        lw=1.2,
        ls="--",
        label="vanilla call, strike 100",
    )
    tidy(ax2, title="closed forms against volatility", xlabel="volatility (%)", ylabel="price")
    ax2.legend(fontsize=8)
    caption(
        fig,
        "Spot 100, one year, rate 5%, carry 3%, volatility 25% on the left; 100,000 antithetic "
        "paths. Closed forms agree with QuantLib's to 1e-12.",
    )
    return fig


def digital_smile() -> Figure:
    fit = _fit()
    surface = monthly(fit.surface)
    e = fit.expiries[int(np.argmin([abs(e.maturity - 1.0) for e in fit.expiries]))]
    t = e.maturity
    r = float(-np.log(e.discount) / t)
    fwd = float(e.forward / fit.surface.spot) if fit.surface.spot else 1.0
    strikes = np.linspace(0.8, 1.2, 41) * fwd
    k = np.log(strikes / fwd)
    vols = surface.implied_vol(k, t)
    atm = float(surface.implied_vol(np.array([0.0]), t)[0])
    disc = np.exp(-r * t)
    flat = np.array([cf.cash_or_nothing(1.0, x, t, r, np.log(fwd) / t, atm, 1.0, True) for x in strikes])
    own = np.array([cf.cash_or_nothing(1.0, x, t, r, np.log(fwd) / t, v, 1.0, True) for x, v in zip(strikes, vols, strict=True)])
    h = 1e-4
    up = black_price(fwd, strikes * (1 + h), t, surface.implied_vol(np.log(strikes * (1 + h) / fwd), t), disc)
    down = black_price(fwd, strikes * (1 - h), t, surface.implied_vol(np.log(strikes * (1 - h) / fwd), t), disc)
    smile = -(up - down) / (2 * h * strikes)
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "A digital is a call spread, and the smile prices it",
        f"S&P 500, one year, {_day()}. The digital call pays one if the index ends above the "
        f"strike; it is minus the derivative of "
        "the call price in strike, which includes the smile's slope. Black-Scholes at the "
        "at-the-money volatility, or even at the "
        "strike's own volatility, misses the skew term: eight cents on the dollar at 105% of the forward, eleven at 95%.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax.plot(strikes / fwd, flat, color=PALETTE["slate"], lw=1.4, ls="--", label="Black-Scholes, at-the-money volatility")
    ax.plot(strikes / fwd, own, color=PALETTE["violet"], lw=1.4, label="Black-Scholes, the strike's own volatility")
    ax.plot(strikes / fwd, smile, color=PALETTE["blue"], lw=2.0, label="from the smile: -dC/dK")
    tidy(ax, title="digital call price", xlabel="strike / forward", ylabel="price per unit paid")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    ax2.plot(strikes / fwd, 100 * (smile - own), color=PALETTE["accent"], lw=1.8, label="smile minus own-volatility price")
    ax2.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    tidy(ax2, title="the skew term, -vega x dsigma/dK", xlabel="strike / forward", ylabel="cents per unit paid")
    ax2.legend(fontsize=8)
    caption(fig, "Smile from the Day 4 arbitrage-free surface on its monthly expiries; strikes relative to the one-year forward.")
    return fig


@lru_cache(maxsize=1)
def _book() -> tuple[Book, float, float]:
    fit = _fit()
    surface = monthly(fit.surface)
    i = int(np.argmin([abs(e.maturity - 1.0) for e in fit.expiries]))
    e = fit.expiries[i]
    r = float(-np.log(e.discount) / e.maturity)
    b = float(np.log(e.forward / fit.surface.spot) / e.maturity) if fit.surface.spot else r
    atm = float(surface.implied_vol(np.array([0.0]), 1.0)[0])
    heston = Heston(0.0130, 4.42, 0.0530, 1.98, -0.59)  # Day 5's calibration to the same snapshot
    return price_book(1.0, r, b, atm, heston, surface, paths=100_000), r, b


def exotic_model_risk() -> Figure:
    book, r, b = _book()
    fig = new_figure(14.0, 6.0)
    title_block(
        fig,
        "One surface, three models, three prices",
        f"S&P 500, one year, {_day()}. Local volatility reproduces the vanillas by construction and "
        f"Heston nearly; on path-dependent "
        "options they part. The up-and-out call is worth a third more under Heston than under local volatility: the smile's "
        "dynamics, which vanillas do not see, price the barrier.",
    )
    ax = fig.add_axes((0.24, 0.15, 0.72, 0.6))
    reference = "local volatility"
    names = list(book.contracts)
    y = np.arange(len(names))
    colours = {"Black-Scholes": PALETTE["slate"], "Heston": PALETTE["green"]}
    for offset, model in enumerate(colours):
        ratio = [100 * (book.prices[(model, n)].value / book.prices[(reference, n)].value - 1) for n in names]
        # two standard errors of the ratio, from both models' noise
        err = [
            200 * np.hypot(book.prices[(model, n)].stderr, book.prices[(reference, n)].stderr) / book.prices[(reference, n)].value
            for n in names
        ]
        ax.barh(y + (offset - 0.5) * 0.36, ratio, height=0.34, xerr=err, color=colours[model], label=model, error_kw={"lw": 0.8})
    ax.axvline(0.0, color=PALETTE["ink"], lw=0.8)
    ax.set_yticks(y, [f"{n}  ({book.prices[(reference, n)].value:.4f})" for n in names])
    ax.invert_yaxis()
    tidy(
        ax,
        title="price relative to local volatility (local-volatility price in brackets, per unit of spot)",
        xlabel="difference (%)",
    )
    ax.legend(fontsize=8, loc="lower right")
    caption(
        fig,
        f"100,000 paths per model, daily monitoring, rate {r:.2%}, carry {b:.2%}. Black-Scholes at "
        f"the one-year at-the-money volatility; "
        "Heston as calibrated on Day 5; local volatility from the monthly Day 4 surface. Bars: two "
        "standard errors of the difference.",
    )
    return fig
