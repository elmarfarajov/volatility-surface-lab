"""Day 3 charts: an exact Black function, its inverse, and what real option chains look like once cleaned.

The first three charts need nothing but the code and arbitrary-precision arithmetic.
The others are drawn from a local snapshot of the S&P 500 index (SPX), SPY and
Apple chains (``volsurf chains fetch``); the gallery skips them where none exists.
"""

from __future__ import annotations

import time
from functools import lru_cache

import mpmath as mp
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.special import ndtr

from ..market import chains
from ..market.black import log_otm_value, normalised_black
from ..market.cleaning import Market, prepare, strike_arbitrage
from ..market.implied import implied_vol, invert_normalised
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

Array = NDArray[np.float64]
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------- references


def _mp_otm(x: float, s: float, digits: int = 60) -> float:
    """The out-of-the-money normalised price ``b(-|x|, s)`` in arbitrary precision."""
    with mp.workdps(digits):
        xm, sm = -abs(mp.mpf(x)), mp.mpf(s)
        value = mp.ncdf(xm / sm + sm / 2) * mp.exp(xm / 2) - mp.ncdf(xm / sm - sm / 2) * mp.exp(-xm / 2)
        return float(value)


def _textbook(x: Array, s: Array) -> Array:
    """``e^{x/2} N(d1) - e^{-x/2} N(d2)`` evaluated literally, as most pricers do."""
    d1 = x / s + 0.5 * s
    return np.asarray(np.exp(0.5 * x) * ndtr(d1) - np.exp(-0.5 * x) * ndtr(d1 - s), dtype=np.float64)


def legacy_implied_vol(beta: Array, x: Array, max_iter: int = 100, tol: float = 1e-12) -> Array:
    """The lab's v1.0 inversion (until Day 3), reproduced to draw the comparison: Newton from the
    Manaster-Koehler point with a bisection bracket, stopped by a tolerance on the normalised price."""
    shape = beta.shape
    beta, x = beta.ravel(), x.ravel()
    m = np.exp(-x)  # K/F, out-of-the-money call for x <= 0
    target = beta * np.exp(-0.5 * x)  # undiscounted price over F: b sqrt(K/F)
    sigma = np.clip(np.sqrt(2.0 * np.abs(x)), 0.05, 2.0)
    lo, hi = np.full_like(beta, 1e-8), np.full_like(beta, 10.0)
    active = np.ones(beta.shape, dtype=bool)
    for _ in range(max_iter):
        if not active.any():
            break
        s = sigma[active]
        d1 = x[active] / s + 0.5 * s
        diff = ndtr(d1) - m[active] * ndtr(d1 - s) - target[active]
        vega = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        lo_a, hi_a = np.where(diff < 0, s, lo[active]), np.where(diff > 0, s, hi[active])
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s - diff / vega
        step = np.where(np.isfinite(newton) & (newton > lo_a) & (newton < hi_a), newton, 0.5 * (lo_a + hi_a))
        done = (np.abs(diff) <= tol * np.maximum(target[active], 1e-300) + 1e-15) | (hi_a - lo_a < 1e-14)
        idx = np.flatnonzero(active)
        lo[idx], hi[idx] = lo_a, hi_a
        sigma[idx] = np.where(done, s, step)
        active[idx[done]] = False
    return sigma.reshape(shape)


@lru_cache(maxsize=1)
def _grid() -> tuple[Array, Array, Array]:
    """Log-moneyness ``k = ln(K/F)``, total volatility, and the exact OTM price on a 121 x 100 grid."""
    k = np.linspace(-4.0, 4.0, 121)
    s = np.geomspace(1e-3, 3.0, 100)
    kk, ss = np.meshgrid(k, s)
    beta = np.array([_mp_otm(a, b) for a, b in zip(kk.ravel(), ss.ravel(), strict=True)]).reshape(kk.shape)
    return kk, ss, beta


def _condition(beta: Array, x: Array, s: Array) -> Array:
    """The relative error in ``s`` that one rounding of the price forces: ``eps beta / (s b'(s))``."""
    log_vega = -0.5 * ((x / s) ** 2 + 0.25 * s * s) - 0.5 * np.log(2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(EPS * beta / (s * np.exp(log_vega)) + EPS, dtype=np.float64)


# ---------------------------------------------------------------- analytic charts


def black_cancellation() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "The textbook Black formula loses digits in the wings",
        "Relative error of an out-of-the-money call against 60-digit arithmetic. e^{x/2}N(d1) - e^{-x/2}N(d2) subtracts two "
        "nearly equal numbers, and the smaller the volatility the more digits go - seven of sixteen at total volatility 0.02. "
        "The Mills-ratio form computes the difference itself, by quadrature where it is small, and "
        "keeps all but the last one or two.",
    )
    k = np.linspace(0.0, 1.6, 241)[1:]
    for index, s in enumerate((0.02, 0.1, 0.4)):
        ax = fig.add_axes((0.055 + index * 0.322, 0.15, 0.27, 0.6))
        x = -k
        exact = np.array([_mp_otm(a, s) for a in x])
        live = exact > 1e-300
        ours = normalised_black(x, s)
        literal = _textbook(x, np.full_like(x, s))
        for values, label, colour, width in (
            (literal, "textbook", PALETTE["red"], 1.3),
            (ours, "Mills ratio", PALETTE["blue"], 1.8),
        ):
            error = np.abs(values[live] - exact[live]) / exact[live]
            ax.semilogy(k[live], np.clip(error, 1e-17, 10.0), color=colour, lw=width, label=label)
        ax.axhline(EPS, color=PALETTE["muted"], lw=0.8, ls=":")
        ax.text(k[2], EPS * 2.2, "machine epsilon", fontsize=7.5, color=PALETTE["muted"])
        ax.set_ylim(1e-17, 30.0)
        last = k[live][-1]
        tidy(ax, title=f"total volatility {s:g}", xlabel="log-moneyness ln(K/F)", ylabel="relative error" if index == 0 else None)
        ax.text(
            0.98,
            0.04,
            f"price falls to {exact[live][-1]:.0e} at ln(K/F) = {last:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
        )
        if index == 0:
            ax.legend(loc="center left", fontsize=8)
    caption(
        fig,
        "Reference: mpmath, 60 significant digits. Curves end where the price underflows double "
        "precision; floored at 1e-17 for display.",
    )
    return fig


def iv_accuracy() -> Figure:
    fig = new_figure(14.0, 6.4)
    title_block(
        fig,
        "Implied volatility to the last digit, from 1e-300 to the forward",
        "Relative error of the recovered volatility for exact out-of-the-money prices on a grid of strike and total volatility. "
        "The v1.0 Newton iteration stopped on a tolerance in price and gave up accuracy as soon as prices became small; the "
        "Jaeckel-style inversion stays within a few roundings of the problem's own conditioning everywhere.",
    )
    kk, ss, beta = _grid()
    x = -np.abs(kk)
    live = beta > 1e-300
    truth = ss
    legacy = legacy_implied_vol(np.where(live, beta, 1.0), x)
    ours = invert_normalised(np.where(live, beta, 1.0), x).total_vol
    levels = np.array([-16, -14, -12, -10, -8, -6, -4, -2, 0, 1])
    colours = ["#0b3d91", "#2f6fdb", "#6fa8f5", "#b7d4fb", "#f6e7c1", "#f2b880", "#e07b54", "#c23b3b", "#7a1f1f"]
    cmap, norm = ListedColormap(colours), BoundaryNorm(levels, len(colours))
    for index, (values, label) in enumerate(((legacy, "v1.0: Newton on the price"), (ours, "now: Householder on ln b"))):
        ax = fig.add_axes((0.05 + index * 0.3, 0.15, 0.25, 0.6))
        error = np.where(live, np.abs(values - truth) / truth, np.nan)
        image = ax.pcolormesh(kk, ss, np.log10(np.clip(error, 1e-17, 9.0)), cmap=cmap, norm=norm, shading="auto")
        contours = ax.contour(
            kk, ss, np.log10(np.where(live, beta, 1e-300)), levels=[-200, -50, -5], colors=PALETTE["ink"], linewidths=0.6
        )
        ax.clabel(contours, fmt=lambda v: f"price 1e{int(v)}", fontsize=6.5)
        ax.set_yscale("log")
        tidy(ax, title=label, xlabel="log-moneyness ln(K/F)", ylabel="total volatility" if index == 0 else None)
        ax.grid(False)
    bar = fig.add_axes((0.605, 0.15, 0.008, 0.6))
    fig.colorbar(image, cax=bar)
    bar.set_title("log10 error", fontsize=8, loc="left")
    ax = fig.add_axes((0.71, 0.15, 0.27, 0.6))
    decades = np.arange(-300, 1, 20)
    centre, worst_legacy, worst_ours, condition = [], [], [], []
    kappa = _condition(beta, x, ss)
    for lo in decades[:-1]:
        band = live & (np.log10(np.where(live, beta, 1e-300)) >= lo) & (np.log10(np.where(live, beta, 1e-300)) < lo + 20)
        if band.any():
            centre.append(lo + 10)
            worst_legacy.append(np.max(np.abs(legacy[band] - truth[band]) / truth[band]))
            worst_ours.append(np.max(np.abs(ours[band] - truth[band]) / truth[band]))
            condition.append(np.max(kappa[band]))
    ax.semilogy(centre, np.clip(worst_legacy, 1e-17, None), "o-", color=PALETTE["red"], ms=3.5, lw=1.3, label="v1.0")
    ax.semilogy(centre, np.clip(worst_ours, 1e-17, None), "o-", color=PALETTE["blue"], ms=3.5, lw=1.8, label="now")
    ax.semilogy(centre, condition, color=PALETTE["muted"], lw=1.0, ls="--", label="conditioning (one ulp of price)")
    tidy(ax, title="worst error per band of prices", xlabel="log10 of the normalised price", ylabel="relative error")
    ax.legend(fontsize=8, loc="center left")
    n = int(live.sum())
    bad_legacy = int(np.sum(live & (np.abs(legacy - truth) / truth > 1e-8)))
    caption(
        fig,
        f"{n:,} representable quotes; v1.0 is worse than 1e-8 on {bad_legacy:,} of them, the new inversion on "
        f"{int(np.sum(live & (np.abs(ours - truth) / truth > 1e-8)))}. Reference prices from mpmath at 60 digits.",
    )
    return fig


def inversion_iterations() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "How many steps the inversion takes",
        "Third-order Householder iterations from asymptotic starting points, on ln b below half the price bound and on "
        "ln(b_max - b) above it. The objectives are concave, so a bracket tightens at every step and bisection catches the rest.",
    )
    kk, ss, beta = _grid()
    x = -np.abs(kk)
    live = beta > 1e-300
    result = invert_normalised(np.where(live, beta, 1.0), x)
    ax = fig.add_axes((0.05, 0.15, 0.4, 0.6))
    counts = np.where(live, result.iterations, np.nan)
    top = int(np.nanmax(counts))
    image = ax.pcolormesh(kk, ss, counts, cmap="Blues", vmin=0, vmax=max(top, 1), shading="auto")
    ax.contour(kk, ss, np.where(live, beta / np.exp(0.5 * x), 0.0), levels=[0.5], colors=PALETTE["accent"], linewidths=1.4)
    ax.set_yscale("log")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="iterations", pad=0.01)
    tidy(
        ax,
        title="iterations over the grid (orange: the switch of objective)",
        xlabel="log-moneyness ln(K/F)",
        ylabel="total volatility",
    )
    rng = np.random.default_rng(3)
    n = 200_000
    xs = -np.abs(rng.uniform(-3.0, 3.0, n))
    sv = 10 ** rng.uniform(-2.5, 0.5, n)
    prices = np.exp(log_otm_value(xs, sv))
    ok = prices > 1e-300
    start = time.perf_counter()
    sample = invert_normalised(prices[ok], xs[ok])
    elapsed = time.perf_counter() - start
    ax = fig.add_axes((0.56, 0.15, 0.41, 0.6))
    bins = np.arange(0, int(np.max(sample.iterations)) + 2) - 0.5
    for branch, label, colour in (
        (0, "below half the bound: ln b", PALETTE["blue"]),
        (1, "above: ln(b_max - b)", PALETTE["green"]),
    ):
        ax.hist(sample.iterations[sample.branch == branch], bins=bins.tolist(), color=colour, alpha=0.8, label=label)
    tidy(ax, title=f"{int(ok.sum()):,} random quotes", xlabel="iterations", ylabel="quotes")
    ax.legend(fontsize=8)
    ax.text(
        0.98,
        0.62,
        f"mean {sample.iterations.mean():.2f}, at most {int(np.max(sample.iterations))}\n"
        f"{elapsed / ok.sum() * 1e6:.2f} microseconds a quote\n(vectorised NumPy)",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
    )
    caption(
        fig, "Jaeckel's rational-cubic starting points reach full precision in two steps; these asymptotic ones take a few more."
    )
    return fig


# ---------------------------------------------------------------- market charts


@lru_cache(maxsize=4)
def _market(ticker: str) -> Market:
    if ticker == "^SPX":
        return prepare(chains.load("^SPX"))
    return prepare(chains.load(ticker), american=True, curve=_market("^SPX").curve)


def _day() -> str:
    return _market("^SPX").chain.as_of.strftime("%d %B %Y")


def _slice_near(market: Market, days: float, root: str | None = None) -> int:
    t = np.array([s.maturity for s in market.slices])
    allowed = np.array([root is None or s.root == root for s in market.slices])
    return int(np.argmin(np.where(allowed, np.abs(t - days / 365.0), np.inf)))


def parity_forward() -> Figure:
    spx = _market("^SPX")
    fig = new_figure(14.0, 9.6)
    title_block(
        fig,
        "Forwards and discount factors from put-call parity",
        f"S&P 500 index options at the close of {_day()}. C - P = D (F - K) for every strike: weighted, trimmed least squares "
        "gives each expiry's forward and discount factor; one Nelson-Siegel curve through them gives the short expiries the "
        "long ones' precision.",
    )
    chosen = max(spx.slices, key=lambda s: s.free_fit.pairs)
    fit = chosen.free_fit
    ax = fig.add_axes((0.06, 0.53, 0.4, 0.3))
    k, mid = fit.strikes, fit.synthetic_mid
    quoted = fit.quoted
    ax.plot(k[fit.used], mid[fit.used], "o", ms=2.6, color=PALETTE["blue"], label=f"{fit.pairs} pairs used")
    out = quoted & ~fit.used
    ax.plot(k[out], mid[out], "x", ms=4, color=PALETTE["red"], label=f"{int(out.sum())} trimmed")
    line = np.linspace(float(np.min(k)), float(np.max(k)), 2)
    ax.plot(line, fit.discount * (fit.forward - line), color=PALETTE["ink"], lw=1.0, label="fitted D (F - K)")
    tidy(ax, title=f"synthetic forward, {chosen.expiry} {chosen.root}", xlabel="strike", ylabel="C - P (index points)")
    ax.legend(fontsize=8)
    ax = fig.add_axes((0.55, 0.53, 0.42, 0.3))
    residual = fit.residuals
    half = 0.5 * (fit.synthetic_ask - fit.synthetic_bid)
    ax.errorbar(
        k[fit.used], residual[fit.used], yerr=half[fit.used], fmt="o", ms=2.4, color=PALETTE["blue"], elinewidth=0.6, label="used"
    )
    ax.plot(k[out], residual[out], "x", ms=4, color=PALETTE["red"], label="trimmed")
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.8)
    lim = 4.0 * float(np.median(half[fit.used]))
    ax.set_ylim(-lim, lim)
    tidy(ax, title="residuals with each pair's bid-ask half-width", xlabel="strike", ylabel="index points")
    ax.text(
        0.02,
        0.04,
        f"F = {fit.forward:,.2f} +- {fit.forward_se:.2f}   D = {fit.discount:.5f} +- {fit.discount_se:.5f}   "
        f"line inside {fit.inside:.0%} of the boxes",
        transform=ax.transAxes,
        fontsize=8,
    )
    ts = spx.term_structure()
    ax = fig.add_axes((0.06, 0.1, 0.4, 0.3))
    reliable = (ts["pairs"] >= 10) & (ts["inside"] >= 0.8)
    shown = reliable & (ts["free_rate_se"] < 0.05)
    ax.errorbar(
        ts["maturity"][shown],
        100 * ts["free_rate"][shown],
        yerr=100 * ts["free_rate_se"][shown],
        fmt="o",
        ms=3,
        color=PALETTE["blue"],
        elinewidth=0.7,
        label="one expiry, +- 1 se",
    )
    grid = np.geomspace(ts["maturity"].min(), ts["maturity"].max(), 200)
    ax.plot(grid, 100 * spx.curve.rate(grid), color=PALETTE["accent"], lw=2.0, label="Nelson-Siegel curve")
    ax.set_xscale("log")
    ax.set_ylim(2.0, 8.0)
    tidy(ax, title="the rate the options imply", xlabel="years to expiry", ylabel="zero rate (%)")
    ax.legend(fontsize=8)
    ax = fig.add_axes((0.55, 0.1, 0.42, 0.3))
    short = (ts["maturity"] < 0.25) & reliable
    implied_spot = float(np.exp(np.polyfit(ts["maturity"][short], np.log(ts["forward"][short]), 1)[1]))
    close = spx.chain.spot
    curve_rate = spx.curve.rate(ts["maturity"])
    carry_close = curve_rate - np.log(ts["forward"] / close) / ts["maturity"]
    carry_implied = curve_rate - np.log(ts["forward"] / implied_spot) / ts["maturity"]
    ax.plot(
        ts["maturity"][reliable],
        100 * carry_close[reliable],
        "o",
        ms=3,
        color=PALETTE["muted"],
        label=f"against the 16:00 print, {close:,.2f}",
    )
    ax.plot(
        ts["maturity"][reliable],
        100 * carry_implied[reliable],
        "o",
        ms=3,
        color=PALETTE["green"],
        label=f"against the options' own spot, {implied_spot:,.2f}",
    )
    ax.set_xscale("log")
    ax.set_ylim(-5.0, 4.0)
    tidy(ax, title="the dividend yield the forwards imply", xlabel="years to expiry", ylabel="carry r - ln(F/S)/T (%)")
    ax.legend(fontsize=8, loc="lower right")
    caption(
        fig,
        f"{len(spx.slices)} expiries (SPX AM and SPXW PM apart); lower panels: the "
        f"{int(reliable.sum())} with 10+ pairs and the line inside 80% "
        "of boxes. Options quote to 16:15, after the 16:00 print; the implied spot is the short forwards' intercept.",
    )
    return fig


def american_parity() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Parity is an equality for European options only",
        f"C - P minus the European forward D (F - K), as a share of spot, about three months out on {_day()}. The index "
        "options sit on the line. SPY and Apple options are American: early exercise lifts in-the-money puts above their "
        "European value, so C - P bends below the line, and in-the-money calls - exercised before dividends - lift it above.",
    )
    for index, ticker in enumerate(("^SPX", "SPY", "AAPL")):
        market = _market(ticker)
        s = market.slices[_slice_near(market, 98, "SPX" if ticker == "^SPX" else None)]
        fit = s.fit
        k = fit.strikes
        m = k / fit.forward
        quoted = fit.quoted & (np.abs(np.log(m)) < 0.45)
        scale = 1e4 / market.chain.spot
        residual = fit.residuals * scale
        ax = fig.add_axes((0.05 + index * 0.325, 0.15, 0.27, 0.6))
        ax.fill_between(
            m[quoted],
            (fit.synthetic_bid[quoted] - fit.discount * (fit.forward - k[quoted])) * scale,
            (fit.synthetic_ask[quoted] - fit.discount * (fit.forward - k[quoted])) * scale,
            color=PALETTE["band"],
            label="bid-ask range",
        )
        ax.plot(m[quoted], residual[quoted], "o", ms=2.6, color=SERIES[index], label="mid")
        ax.axhline(0.0, color=PALETTE["ink"], lw=0.8)
        ax.axvline(1.0, color=PALETTE["muted"], lw=0.6, ls=":")
        ax.set_ylim(-160.0, 40.0)
        name = {"^SPX": "SPX (European)", "SPY": "SPY (American)", "AAPL": "Apple (American)"}[ticker]
        tidy(ax, title=f"{name}, {s.expiry}", xlabel="strike / forward", ylabel="basis points of spot" if index == 0 else None)
        deep = quoted & (m > 1.15)
        if deep.any():
            ax.text(
                0.97,
                0.06,
                f"puts beyond 115%: {np.median(residual[deep]):+.1f} bp",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
            )
        if index == 0:
            ax.legend(fontsize=8, loc="lower left")
    caption(
        fig,
        "Discount factors from the SPX curve; American forwards fitted within 5% of spot, where the "
        "early-exercise premium is small.",
    )
    return fig


def cleaning_funnel() -> Figure:
    fig = new_figure(14.0, 6.0)
    title_block(
        fig,
        "From a raw chain to clean quotes, one stated reason at a time",
        f"Every quote listed at the close of {_day()}, and the rule that removed it. Strike arbitrage is tested at the bid "
        "and the ask - a price pattern that can actually be traded - removing, one at a time, the "
        "quote whose loss leaves the least arbitrage.",
    )
    ax = fig.add_axes((0.17, 0.15, 0.35, 0.6))
    rules = [
        "no bid",
        "crossed or locked",
        "no open interest",
        "wide spread",
        "in the money",
        "outside price bounds",
        "strike arbitrage",
        "no implied volatility",
        "kept",
    ]
    for offset, (ticker, colour) in enumerate((("^SPX", SERIES[0]), ("SPY", SERIES[1]), ("AAPL", SERIES[2]))):
        funnel = _market(ticker).funnel()
        values = [100.0 * funnel.get(rule, 0) / funnel["raw"] for rule in rules]
        y = np.arange(len(rules)) + (offset - 1) * 0.27
        label = {"^SPX": "SPX", "SPY": "SPY", "AAPL": "Apple"}[ticker] + f" ({funnel['raw']:,} quotes)"
        ax.barh(y, values, height=0.25, color=colour, label=label)
    ax.set_yticks(np.arange(len(rules)), rules)
    ax.invert_yaxis()
    tidy(ax, title="share of the raw chain each rule removes", xlabel="% of quotes")
    ax.legend(fontsize=8, loc="upper right")
    spx = _market("^SPX")
    raw = spx.chain.quotes
    example = None
    for s in spx.slices:
        group = raw[(raw["expiry"] == s.expiry) & (raw["root"] == s.root) & (raw["bid"] > 0) & (raw["ask"] > raw["bid"])]
        otm = group[np.where(group["is_call"], group["strike"] >= s.forward, group["strike"] < s.forward)].sort_values("strike")
        k = otm["strike"].to_numpy(float)
        shift = np.where(otm["is_call"], 0.0, s.discount * (s.forward - k))
        bid, ask = otm["bid"].to_numpy(float) + shift, otm["ask"].to_numpy(float) + shift
        removed = strike_arbitrage(k, bid, ask, s.discount)
        if removed.size == 0:
            continue
        mids = 0.5 * (bid + ask)
        kept = np.setdiff1d(np.arange(k.size), removed)
        if kept.size < 2:
            continue
        interior = (k[removed] > k[kept].min()) & (k[removed] < k[kept].max())  # with kept neighbours on both sides
        if not interior.any():
            continue
        removed = removed[interior]
        gaps = np.abs(mids[removed] - np.interp(k[removed], k[kept], mids[kept]))
        j = int(np.argmax(gaps))
        if example is None or gaps[j] > example[-1]:
            example = (s, k, bid, ask, removed, int(removed[j]), float(gaps[j]))
    ax = fig.add_axes((0.6, 0.15, 0.37, 0.6))
    if example is not None:
        s, k, bid, ask, removed, worst, gap = example
        near = np.abs(k - k[worst]) <= 12 * float(np.median(np.diff(k)))
        mids = 0.5 * (bid + ask)
        gone = np.zeros(k.size, dtype=bool)
        gone[removed] = True
        ax.vlines(
            k[near & ~gone],
            bid[near & ~gone],
            ask[near & ~gone],
            color=PALETTE["blue"],
            lw=3,
            label="kept: call-equivalent bid to ask",
        )
        ax.vlines(
            k[near & gone], bid[near & gone], ask[near & gone], color=PALETTE["accent"], lw=3, label="removed as strike arbitrage"
        )
        kind = "put" if k[worst] < s.forward else "call"
        tidy(
            ax,
            title=f"the worst quote in the chain: {s.expiry} {s.root}",
            xlabel="strike",
            ylabel="index points (puts through parity)",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.02,
            0.05,
            f"the {k[worst]:.0f} {kind} sits {gap:.1f} points off its neighbours' line:\ntrading it "
            f"against them is riskless at the quotes",
            transform=ax.transAxes,
            fontsize=8,
        )
    caption(
        fig,
        "Rules in the order applied; each bar is the share of the raw chain that rule removed. 'In "
        "the money' keeps the out-of-the-money twin of every strike.",
    )
    return fig


def market_smiles() -> Figure:
    spx = _market("^SPX")
    fig = new_figure(14.0, 8.6)
    title_block(
        fig,
        "The S&P 500 smile across the term structure",
        f"Implied volatility at the bid, mid and ask of every cleaned out-of-the-money quote, {_day()}, against standardised "
        "moneyness ln(K/F)/sqrt(T). Grey dots are Yahoo Finance's own implied volatilities for the same quotes.",
    )
    targets = (7, 30, 91, 182, 365, 730)
    for index, days in enumerate(targets):
        s = spx.slices[_slice_near(spx, days)]
        q = s.quotes
        z = q["k"] / np.sqrt(s.maturity)
        keep = (z > -4.0) & (z < 2.5)
        row, col = divmod(index, 3)
        ax = fig.add_axes((0.055 + col * 0.322, 0.53 - row * 0.41, 0.27, 0.3))
        ax.vlines(z[keep], 100 * q["iv_bid"][keep], 100 * q["iv_ask"][keep], color="#b9cbe8", lw=2.0)
        ax.plot(z[keep], 100 * q["vendor_iv"][keep], ".", ms=2.5, color=PALETTE["muted"])
        ax.plot(z[keep], 100 * q["iv_mid"][keep], "-", lw=1.4, color=SERIES[index % len(SERIES)])
        atm = float(np.interp(0.0, z, q["iv_mid"]))
        tidy(
            ax,
            title=f"{s.expiry} ({s.maturity * 365:.0f} days), ATM {100 * atm:.1f}%",
            xlabel="ln(K/F)/sqrt(T)" if row else None,
            ylabel="implied vol (%)" if col == 0 else None,
        )
    caption(
        fig,
        "Bars: bid to ask; line: mid. Parity forwards make puts and calls meet at the money; "
        "Yahoo's zero-rate figures jump there. No smoothing.",
    )
    return fig


def vendor_iv() -> Figure:
    spx = _market("^SPX")
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Yahoo Finance's implied volatilities assume rates and dividends are zero",
        "Its figures match the mid price inverted with the forward set to spot and no discounting; against volatilities "
        "with the market's own forward and discount factor they are off by more the longer the expiry - too low for "
        "puts, too high for calls.",
    )
    frames = []
    for s in spx.slices:
        q = s.quotes
        k, call = q["strike"].to_numpy(float), q["is_call"].to_numpy(bool)
        zero = implied_vol(q["mid"].to_numpy(float), spx.chain.spot, k, s.maturity, 1.0, call)
        frames.append(pd.DataFrame({"T": s.maturity, "call": call, "vendor": q["vendor_iv"], "ours": q["iv_mid"], "zero": zero}))
    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["vendor"]) & (df["vendor"] > 0.001)]
    ax = fig.add_axes((0.06, 0.15, 0.42, 0.6))
    for flag, label, colour in ((False, "puts", PALETTE["blue"]), (True, "calls", PALETTE["green"])):
        part = df[df["call"] == flag]
        ax.plot(part["T"], 100 * (part["vendor"] - part["ours"]), ".", ms=2, alpha=0.35, color=colour)
        grouped = part.groupby(pd.cut(part["T"], np.geomspace(0.005, 6.0, 18)), observed=True)
        centres = grouped["T"].median()
        ax.plot(
            centres,
            100 * grouped.apply(lambda g: (g["vendor"] - g["ours"]).median()),
            "-o",
            ms=3.5,
            lw=1.8,
            color=colour,
            label=f"{label}, median",
        )
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-8.0, 8.0)
    tidy(ax, title="Yahoo minus implied vol from parity forwards", xlabel="years to expiry", ylabel="volatility points")
    ax.legend(fontsize=8)
    ax = fig.add_axes((0.57, 0.15, 0.4, 0.6))
    bins = np.linspace(-3.0, 3.0, 121)
    for column, label, colour in (
        ("ours", "against parity forwards and discounting", PALETTE["slate"]),
        ("zero", "against F = S, D = 1", PALETTE["accent"]),
    ):
        gap = 100 * (df["vendor"] - df[column])
        share = float(np.mean(np.abs(gap) < 0.1))
        ax.hist(gap.clip(-3.0, 3.0), bins=bins.tolist(), color=colour, alpha=0.75, label=f"{label}: {share:.0%} within 0.1 pt")
    tidy(
        ax,
        title=f"{len(df):,} quotes: which assumption reproduces Yahoo",
        xlabel="Yahoo minus recomputed (volatility points)",
        ylabel="quotes",
    )
    ax.legend(fontsize=8, loc="upper left")
    caption(
        fig,
        "Same quotes and inversion for every series; only the forward and the discount factor "
        "differ. Histogram clipped at +-3 points.",
    )
    return fig


def snapshot_available() -> bool:
    return chains.latest("SPX") is not None and chains.latest("SPY") is not None and chains.latest("AAPL") is not None


__all__ = [
    "american_parity",
    "black_cancellation",
    "cleaning_funnel",
    "inversion_iterations",
    "iv_accuracy",
    "legacy_implied_vol",
    "market_smiles",
    "parity_forward",
    "snapshot_available",
    "vendor_iv",
]
