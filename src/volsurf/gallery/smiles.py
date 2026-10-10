"""Day 4 charts: smiles and surfaces free of static arbitrage, and what the original fitter left behind.

Vogt's slice needs nothing but the code; the others are drawn from the local S&P 500
snapshot (``volsurf chains fetch``) and are skipped where none exists.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..market import chains
from ..market.cleaning import prepare
from ..smile.build import SurfaceFit, build_surface, inside_spread
from ..smile.density import density, moments
from ..smile.fit import fit_slice
from ..smile.ssvi import Quotes, fit_essvi
from ..smile.svi import RawSVI, butterfly_check, calendar_check
from ..surface import VolSurface
from ..svi import SVISlice, fit_svi_slice
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

Array = NDArray[np.float64]
VOGT = RawSVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)  # Gatheral and Jacquier (2014), section 7, T = 1


@lru_cache(maxsize=1)
def _fit() -> SurfaceFit:
    return build_surface(prepare(chains.load("^SPX")))


@lru_cache(maxsize=1)
def _v1() -> list[SVISlice]:
    """The original fitter on the same slices, each against the one before, as it was used."""
    fitted: list[SVISlice] = []
    for e, q in zip(_fit().expiries, _fit().quotes, strict=True):
        weights = np.sqrt(q.weight)
        fitted.append(fit_svi_slice(q.k, q.iv, e.maturity, weights, previous=fitted[-1] if fitted else None).slice)
    return fitted


def _as_raw(s: SVISlice) -> RawSVI:
    return RawSVI(s.a, s.b, float(np.clip(s.rho, -0.9999, 0.9999)), s.m, max(s.s, 1e-8))


@lru_cache(maxsize=1)
def _day() -> str:
    return chains.load("^SPX").as_of.strftime("%d %B %Y")


def _g(w: Array, k: Array) -> Array:
    """Durrleman's g from a sampled total variance, by central differences."""
    h = k[1] - k[0]
    w1 = np.gradient(w, h)
    w2 = np.gradient(w1, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray((1 - k * w1 / (2 * w)) ** 2 - 0.25 * w1**2 * (1 / w + 0.25) + 0.5 * w2, dtype=np.float64)


def vogt_slice() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "A smile that fits and still admits arbitrage, and its repair",
        "Axel Vogt's raw SVI slice (Gatheral and Jacquier, 2014, section 7) looks like any equity smile, yet its density is "
        "negative for strikes well above the forward. Refitted to its own implied volatilities with the conditions as hard "
        "constraints on the whole line, it moves by at most a volatility point and a half, and its density is a density.",
    )
    k = np.linspace(-1.5, 1.5, 601)
    quotes = Quotes(1.0, k[::10], VOGT.implied_vol(k[::10], 1.0), np.ones(k[::10].size))
    repaired = fit_slice(quotes, fit_essvi([quotes])[0].raw()).slice
    panels: tuple[tuple[str, Callable[[RawSVI], Array], str], ...] = (
        ("implied volatility", lambda s: 100 * s.implied_vol(k, 1.0), "volatility (%)"),
        ("Durrleman's g", lambda s: s.g(k), "g(k)"),
        ("density of ln(S_T / F)", lambda s: density(s, k), "density"),
    )
    for index, (title, values, label) in enumerate(panels):
        ax = fig.add_axes((0.055 + index * 0.322, 0.15, 0.27, 0.6))
        ax.plot(k, values(VOGT), color=PALETTE["red"], lw=1.6, label="Vogt's slice")
        ax.plot(k, values(repaired), color=PALETTE["blue"], lw=1.6, label="refitted under constraints")
        if index:
            ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
            bad = values(VOGT) < 0
            ax.fill_between(k, 0, values(VOGT), where=bad, color=PALETTE["red"], alpha=0.15)
        tidy(ax, title=title, xlabel="log-moneyness ln(K/F)", ylabel=label)
        if index == 0:
            ax.legend(fontsize=8)
    worst = butterfly_check(VOGT)
    negative = k[VOGT.g(k) < 0]
    gap = float(np.max(np.abs(repaired.implied_vol(k, 1.0) - VOGT.implied_vol(k, 1.0))))
    caption(
        fig,
        f"Vogt: a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153; g < 0 for k in "
        f"[{float(np.min(negative)):.2f}, {float(np.max(negative)):.2f}], "
        f"min g = {worst.worst:.4f} at k = {worst.at:.2f}. "
        f"Refit: min g {butterfly_check(repaired).worst:+.1e} on the whole line, largest volatility "
        f"change {100 * gap:.2f} points.",
    )
    return fig


def v1_arbitrage() -> Figure:
    fit = _fit()
    v1 = [_as_raw(s) for s in _v1()]
    t = np.array([e.maturity for e in fit.expiries])
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "The original surface had arbitrage in most of its slices and between them",
        f"S&P 500 options, {_day()}. The v1.0 fitter penalised a negative density and crossing slices on a grid of 81 strikes; "
        "checked on the whole line, most of its slices fail. The constrained fit and price interpolation pass everywhere.",
    )
    ax = fig.add_axes((0.055, 0.15, 0.27, 0.6))
    g_v1 = np.array([butterfly_check(s).worst for s in v1])
    g_new = np.array([f.butterfly_margin for f in fit.svi])
    ax.plot(t, g_v1, "o", ms=4, color=PALETTE["red"], label=f"v1.0: {int(np.sum(g_v1 < -1e-10))} of {len(t)} negative")
    ax.plot(t, g_new, "o", ms=4, color=PALETTE["blue"], label=f"constrained: {int(np.sum(g_new < -1e-10))} negative")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    tidy(ax, title="smallest g on the whole line, per slice", xlabel="years to expiry", ylabel="min g")
    ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_axes((0.385, 0.15, 0.27, 0.6))
    c_v1 = np.array([calendar_check(a, b).worst for a, b in zip(v1[:-1], v1[1:], strict=True)])
    c_new = np.array([f.calendar_margin for f in fit.svi[1:]])
    ax.plot(t[1:], c_v1, "o", ms=4, color=PALETTE["red"], label=f"v1.0: {int(np.sum(c_v1 < -1e-12))} of {len(c_v1)} pairs cross")
    ax.plot(t[1:], c_new, "o", ms=4, color=PALETTE["blue"], label=f"constrained: {int(np.sum(c_new < -1e-12))} cross")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    tidy(ax, title="smallest w(k) - w_prev(k), per pair", xlabel="years to expiry", ylabel="total variance gap")
    ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_axes((0.715, 0.15, 0.26, 0.6))
    surface = VolSurface(
        _v1(), np.array([e.forward for e in fit.expiries]), np.array([e.discount for e in fit.expiries]), fit.surface.spot or 1.0
    )
    kk = np.linspace(-1.5, 1.0, 2501)
    tt = np.geomspace(t[0] * 1.05, t[-1] * 0.97, 200)
    worst = [float(np.min(_g(surface.total_variance(kk, x), kk)[5:-5])) for x in tt]
    t_bad = float(tt[int(np.argmin(worst))])
    g_v1 = _g(surface.total_variance(kk, t_bad), kk)
    h = kk[1] - kk[0]
    c = fit.surface.call(kk, t_bad)
    w_new = fit.surface.total_variance(kk, t_bad)
    # Durrleman's g of the price-interpolated smile, from the density: g = p sqrt(2 pi w) exp(d_-^2 / 2)
    dens = np.exp(-kk[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2 * h))
    root = np.sqrt(w_new[1:-1])
    g_new = dens * np.sqrt(2 * np.pi) * root * np.exp(0.5 * (-kk[1:-1] / root - 0.5 * root) ** 2)
    ax.plot(kk[5:-5], g_v1[5:-5], color=PALETTE["red"], lw=1.4, label="v1.0: variance interpolation")
    ax.plot(kk[1:-1][4:-4], g_new[4:-4], color=PALETTE["blue"], lw=1.6, label="price interpolation")
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    ax.set_yscale("symlog", linthresh=1e-2)
    tidy(ax, title=f"the worst intermediate smile, T = {t_bad:.3f}", xlabel="ln(K/F)", ylabel="g(k)")
    ax.legend(fontsize=8, loc="lower left")
    caption(
        fig,
        f"Right: Durrleman's g of the smile between expiries where v1.0 is worst on |k| <= 1.5 (min "
        f"g = {min(worst):.2f}); the v1.0 "
        "surface interpolates total variance by PCHIP after a running maximum. Symmetric-log axes: linear near zero.",
    )
    return fig


def smile_fits() -> Figure:
    fit = _fit()
    v1 = _v1()
    fig = new_figure(14.0, 8.6)
    title_block(
        fig,
        "Fitting the S&P 500 smile without arbitrage",
        f"{_day()}. Bid-ask bands of the cleaned quotes with three fits: the v1.0 penalised SVI, "
        f"eSSVI (three parameters a slice, "
        "arbitrage-free by its conditions) and raw SVI constrained on the whole line, started from eSSVI.",
    )
    targets = (7, 30, 91, 182, 365, 730)
    t = np.array([e.maturity for e in fit.expiries])
    for index, days in enumerate(targets):
        i = int(np.argmin(np.abs(t - days / 365.0)))
        e, q = fit.expiries[i], fit.expiries[i].quotes
        row, col = divmod(index, 3)
        ax = fig.add_axes((0.055 + col * 0.322, 0.53 - row * 0.41, 0.27, 0.3))
        k = q["k"].to_numpy()
        ax.vlines(k, 100 * q["iv_bid"], 100 * q["iv_ask"], color="#b9cbe8", lw=2.0)
        grid = np.linspace(k.min(), k.max(), 400)
        series = (
            ("v1.0 SVI", v1[i].total_variance(grid), PALETTE["red"], 1.0),
            ("eSSVI", fit.essvi[i].total_variance(grid), PALETTE["green"], 1.2),
            ("constrained SVI", fit.svi[i].slice.total_variance(grid), PALETTE["blue"], 1.6),
        )
        for label, w, colour, width in series:
            ax.plot(grid, 100 * np.sqrt(np.maximum(w, 0) / e.maturity), color=colour, lw=width, label=label)
        share = inside_spread(e, fit.svi[i].slice.total_variance(k))
        tidy(
            ax,
            title=f"{e.expiry} ({e.maturity * 365:.0f} days): {share:.0%} inside",
            xlabel="ln(K/F)" if row else None,
            ylabel="implied vol (%)" if col == 0 else None,
        )
        if index == 0:
            ax.legend(fontsize=7.5)
    caption(fig, "'Inside': share of quotes whose bid-ask range contains the constrained fit. Bars: bid to ask.")
    return fig


def fit_quality() -> Figure:
    fit = _fit()
    v1 = _v1()
    t = np.array([e.maturity for e in fit.expiries])
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "What the no-arbitrage conditions cost in fit",
        "Root-mean-square error and the share of quotes inside the bid-ask range, slice by slice. SSVI's three-parameter slice "
        "cannot follow the S&P 500 smile - fitted to one expiry with no constraint at all it misses by as much - so its "
        "conditions cost nothing; the shape does. Raw SVI constrained on the whole line fits almost as closely as the v1.0 fit, "
        "and is free of arbitrage.",
    )
    models = (
        ("SSVI (one surface)", [s.total_variance for s in fit.ssvi.slices()], PALETTE["slate"]),
        ("eSSVI", [s.total_variance for s in fit.essvi], PALETTE["green"]),
        ("v1.0 SVI (arbitrage)", [s.total_variance for s in v1], PALETTE["red"]),
        ("constrained SVI", [f.slice.total_variance for f in fit.svi], PALETTE["blue"]),
    )
    ax1 = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    for label, curves, colour in models:
        rmse, inside = [], []
        for e, w in zip(fit.expiries, curves, strict=True):
            k = e.quotes["k"].to_numpy()
            iv = np.sqrt(np.maximum(w(k), 0) / e.maturity)
            rmse.append(100 * np.sqrt(np.mean((iv - e.quotes["iv_mid"].to_numpy()) ** 2)))
            inside.append(100 * inside_spread(e, w(k)))
        ax1.plot(t, rmse, "o-", ms=3, lw=1.0, color=colour, label=f"{label}: median {np.median(rmse):.2f}")
        ax2.plot(t, inside, "o-", ms=3, lw=1.0, color=colour, label=f"{label}: median {np.median(inside):.0f}%")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
    ax1.set_yscale("log")
    tidy(ax1, title="root-mean-square error", xlabel="years to expiry", ylabel="volatility points")
    tidy(ax2, title="quotes inside the bid-ask range", xlabel="years to expiry", ylabel="%")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8, loc="lower left")
    binding = sum(1 for s in fit.essvi if s.psi**2 * (1 + abs(s.rho)) > 0.99 * 4 * s.theta)
    caption(
        fig,
        f"eSSVI's curvature condition psi^2 (1 + |rho|) <= 4 theta is binding (within 1%) on "
        f"{binding} of {len(fit.essvi)} slices: "
        "the conditions are not what limits it.",
    )
    return fig


def densities() -> Figure:
    fit = _fit()
    t = np.array([e.maturity for e in fit.expiries])
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "The distributions the S&P 500 options price",
        f"Risk-neutral densities of S_T / F from the constrained fit, {_day()}. Each integrates to "
        f"one and has the forward as its "
        "mean to rounding, and matches Breeden and Litzenberger's second difference of call prices.",
    )
    ax1 = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    for index, days in enumerate((7, 30, 91, 365, 730)):
        i = int(np.argmin(np.abs(t - days / 365.0)))
        e, s = fit.expiries[i], fit.svi[i].slice
        k = np.linspace(-1.2, 0.5, 2001)
        p = density(s, k) * np.exp(-k)  # density of S_T / F = e^k
        x = np.exp(k)
        colour = SERIES[index % len(SERIES)]
        ax1.plot(x, p, color=colour, lw=1.5, label=f"{e.expiry} ({days} days)")
        ax2.semilogy(x, np.maximum(p, 1e-12), color=colour, lw=1.5)
    tidy(ax1, title="density of S_T / F", xlabel="S_T / F", ylabel="density")
    ax1.set_xlim(0.5, 1.3)
    ax1.legend(fontsize=8)
    tidy(ax2, title="the same, on a log scale: the tails", xlabel="S_T / F", ylabel="density")
    ax2.set_xlim(0.3, 1.6)
    ax2.set_ylim(1e-6, 100)
    stats = [moments(f.slice) for f in fit.svi]
    caption(
        fig,
        f"Over all {len(stats)} slices: |mass - 1| <= {max(abs(m.mass - 1) for m in stats):.0e}, |E[S_T]/F - 1| <= "
        f"{max(abs(m.forward - 1) for m in stats):.0e}, Breeden-Litzenberger within "
        f"{max(m.breeden_litzenberger for m in stats):.0e} of the peak. "
        "Dips to zero: where the quotes alone would imply a negative density, the fit sits on its bound g = 1e-6.",
    )
    return fig


def time_interpolation() -> Figure:
    fit = _fit()
    t_nodes = np.array([e.maturity for e in fit.expiries])
    surface = VolSurface(
        _v1(), np.array([e.forward for e in fit.expiries]), np.array([e.discount for e in fit.expiries]), fit.surface.spot or 1.0
    )
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Between expiries: interpolate prices, not variances",
        "Total implied variance against maturity at fixed log-moneyness, and the smallest density over strikes at each maturity. "
        "Gatheral and Jacquier's price interpolation keeps every intermediate smile a valid distribution; beyond the last expiry "
        "the distribution is convolved with log-normal noise.",
    )
    ax1 = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    tt = np.concatenate([np.geomspace(0.004, t_nodes[-1], 120), np.linspace(t_nodes[-1], 5.0, 12)[1:]])
    for index, kv in enumerate((-0.3, -0.1, 0.0, 0.1)):
        w = np.array([float(fit.surface.total_variance(np.array([kv]), x)[0]) for x in tt])
        ax1.plot(tt, w, color=SERIES[index], lw=1.6, label=f"k = {kv:+.1f}")
        ax1.plot(t_nodes, [float(s.slice.total_variance(kv)) for s in fit.svi], "o", ms=2.5, color=SERIES[index])
    ax1.axvline(t_nodes[-1], color=PALETTE["muted"], lw=0.8, ls=":")
    ax1.text(t_nodes[-1] * 1.03, ax1.get_ylim()[1] * 0.9, "last expiry", fontsize=8, color=PALETTE["muted"])
    tidy(ax1, title="total variance, price-interpolated (dots: fitted slices)", xlabel="years to expiry", ylabel="w(k, t)")
    ax1.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    kk = np.linspace(-0.6, 0.3, 361)
    h = kk[1] - kk[0]
    times = np.geomspace(t_nodes[0] * 1.02, t_nodes[-1] * 0.98, 140)
    v1_min, new_min = [], []
    for x in times:
        v1_min.append(np.min(_g(surface.total_variance(kk, x), kk)[3:-3]))
        c = fit.surface.call(kk, x)
        dens = np.exp(-kk[1:-1]) * ((c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 - (c[2:] - c[:-2]) / (2 * h))
        new_min.append(np.min(dens))
    ax2.plot(times, v1_min, color=PALETTE["red"], lw=1.3, label="v1.0: min g (variance interpolation)")
    ax2.plot(times, new_min, color=PALETTE["blue"], lw=1.6, label="price interpolation: min density")
    ax2.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    ax2.set_xscale("log")
    ax2.set_yscale("symlog", linthresh=1e-3)
    tidy(ax2, title="worst point of each intermediate smile", xlabel="years to expiry", ylabel="(negative: arbitrage)")
    ax2.legend(fontsize=8, loc="lower left")
    caption(
        fig,
        f"{int(np.sum(np.array(v1_min) < 0))} of {len(times)} intermediate v1.0 smiles have a negative density; "
        f"{int(np.sum(np.array(new_min) < 0))} of the price-interpolated ones.",
    )
    return fig


def lee_wings() -> Figure:
    fit = _fit()
    t = np.array([e.maturity for e in fit.expiries])
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "What the wings say about the moments of the index",
        "Roger Lee's moment formula: a wing of total-variance slope beta means S_T has finite "
        "moments up to an order fixed by beta, "
        "and no slope may exceed 2. The S&P 500 put wing is the steeper at every maturity, about three times the call wing a "
        "week out, and every slope is far inside the bound.",
    )
    left = np.array([f.slice.wing_slopes[0] for f in fit.svi])
    right = np.array([f.slice.wing_slopes[1] for f in fit.svi])
    ax1 = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax1.plot(t, left, "o-", ms=3, color=PALETTE["blue"], label="put wing b(1 - rho)")
    ax1.plot(t, right, "o-", ms=3, color=PALETTE["green"], label="call wing b(1 + rho)")
    ax1.axhline(2.0, color=PALETTE["accent"], lw=1.4, label="Lee's bound, 2")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    tidy(ax1, title="wing slopes of total variance", xlabel="years to expiry", ylabel="slope")
    ax1.legend(fontsize=8)

    def order(beta: Array) -> Array:
        c = (2.0 - beta) / 4.0
        with np.errstate(divide="ignore"):
            return np.asarray(c**2 / (1.0 - 2.0 * c), dtype=np.float64)  # Lee (2004): the critical p for slope beta

    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    ax2.plot(t, 1.0 + order(right), "o-", ms=3, color=PALETTE["green"], label="E[S^(1+p)] finite up to 1 + p (calls)")
    ax2.plot(t, order(left), "o-", ms=3, color=PALETTE["blue"], label="E[S^(-q)] finite up to q (puts)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    tidy(ax2, title="the highest finite moments the wings imply", xlabel="years to expiry", ylabel="moment order")
    ax2.legend(fontsize=8)
    caption(
        fig,
        "Lee (2004): slope beta = 2 - 4 (sqrt(p^2 + p) - p). Beyond the last quoted strike the "
        "wings are SVI's extrapolation, so these "
        "are the moments the fitted surface implies, not ones the quotes prove.",
    )
    return fig


def snapshot_available() -> bool:
    return chains.latest("SPX") is not None
