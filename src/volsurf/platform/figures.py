"""The report's figures, drawn from one :class:`~volsurf.platform.analysis.Analysis` in the gallery's house style."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from ..gallery.style import PALETTE, SERIES, caption, new_figure, tidy, title_block
from ..localvol.dupire import local_vol
from ..localvol.sabr import hagan_vol
from ..smile.density import density
from ..stochvol.calibration import model_vols
from .analysis import Analysis


def _picks(n: int, count: int = 6) -> list[int]:
    return [int(i) for i in np.unique(np.linspace(0, n - 1, min(n, count)).astype(int))]


def surface(a: Analysis) -> Figure:
    fit = a.fit
    t = np.geomspace(fit.expiries[0].maturity, fit.expiries[-1].maturity, 40)
    z = np.linspace(-2.5, 1.5, 45)
    vol = np.array([fit.surface.implied_vol(z * np.sqrt(float(fit.surface.theta(x))), float(x)) for x in t])
    fig = new_figure(12.0, 7.4)
    title_block(
        fig,
        f"The {a.source.chain.ticker} implied volatility surface, {a.source.day}",
        f"Raw SVI under hard butterfly and calendar constraints, joined by price interpolation: free of static arbitrage on the "
        f"whole line. {int(a.results['quotes']):,} quotes over {int(a.results['expiries'])} expiries; median error "
        f"{100 * a.results['surface.median_rmse_vol']:.2f} vol points.",
    )
    ax = fig.add_axes((0.02, 0.04, 0.96, 0.8), projection="3d")
    assert isinstance(ax, Axes3D)
    zz, tt = np.meshgrid(z, np.log10(365 * t))
    ax.plot_surface(zz, tt, 100 * vol, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    days = [d for d in (7, 30, 90, 180, 365, 730) if 365 * t[0] <= d <= 365 * t[-1] * 1.01]
    ax.set_yticks(np.log10(days), [f"{d}d" for d in days])
    ax.set_xlabel("ln(K/F) / sqrt(w_ATM)")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=24, azim=-128)
    return fig


def smiles(a: Analysis) -> Figure:
    fig = new_figure(14.0, 8.4)
    h = a.heston.heston
    title_block(
        fig,
        "Smiles: quotes, the arbitrage-free surface, Heston and SABR",
        f"{a.source.chain.ticker}, {a.source.day}. Bars span each quote's bid-ask implied volatility; the surface is raw SVI "
        f"under hard no-arbitrage constraints (median error {100 * a.results['surface.median_rmse_vol']:.2f} vol points). Heston "
        f"(one parameter set for all expiries) misses by {100 * a.heston.rmse:.2f} points; SABR is fitted expiry by expiry.",
    )
    expiries = a.expiries
    picks = _picks(len(expiries))
    vols = model_vols(h, expiries)
    sabr = dict(a.sabr)
    for index, i in enumerate(picks):
        e = expiries[i]
        row, col = divmod(index, 3)
        ax = fig.add_axes((0.055 + col * 0.322, 0.53 - row * 0.41, 0.27, 0.29))
        grid = np.linspace(float(np.min(e.k)), float(np.max(e.k)), 81)
        ax.vlines(e.k, 100 * (e.iv - e.width / 2), 100 * (e.iv + e.width / 2), color=PALETTE["slate"], lw=1.4, label="bid-ask")
        ax.plot(grid, 100 * a.fit.surface.implied_vol(grid, e.maturity), color=PALETTE["blue"], lw=1.8, label="SVI surface")
        ax.plot(e.k, 100 * vols[i], "o", ms=3.5, color=PALETTE["red"], label="Heston")
        fit = sabr.get(e.maturity)
        if fit is not None:
            ax.plot(
                grid,
                100 * hagan_vol(1.0, np.exp(grid), e.maturity, fit.params),
                color=PALETTE["green"],
                lw=1.2,
                ls="--",
                label="SABR",
            )
        tidy(
            ax,
            title=f"{365 * e.maturity:.0f} days",
            xlabel="ln(K/F)" if row == 1 else None,
            ylabel="implied vol (%)" if col == 0 else None,
        )
        if index == 0:
            ax.legend(fontsize=7.5)
    caption(fig, f"Heston: v0 {h.v0:.4f}, kappa {h.kappa:.2f}, theta {h.theta:.4f}, vol of vol {h.sigma:.2f}, rho {h.rho:.2f}.")
    return fig


def term_structure(a: Analysis) -> Figure:
    ts = a.market.term_structure()
    t = np.array([e.maturity for e in a.fit.expiries])
    atm = np.array([float(a.fit.surface.implied_vol(np.array([0.0]), x)[0]) for x in t])
    skew = np.array([float(np.diff(a.fit.surface.implied_vol(np.array([-0.01, 0.01]), x))[0] / 0.02) for x in t])
    fig = new_figure(14.0, 5.4)
    title_block(
        fig,
        "Term structure: rates and carry from parity, volatility and skew from the surface",
        "Each expiry's forward and discount factor come from a put-call parity regression on its own quotes. The at-the-money "
        "volatility and skew are read off the arbitrage-free surface.",
    )
    ax = fig.add_axes((0.05, 0.15, 0.27, 0.6))
    ax.plot(365 * ts["maturity"], 100 * ts["curve_rate"], "o-", color=PALETTE["blue"], ms=3.5, label="rate (curve)")
    ax.plot(365 * ts["maturity"], 100 * ts["carry"], "s-", color=PALETTE["green"], ms=3.5, label="carry r - q")
    ax.set_xscale("log")
    tidy(ax, title="rate and carry", xlabel="days to expiry", ylabel="% a year")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.38, 0.15, 0.27, 0.6))
    ax2.plot(365 * t, 100 * atm, "o-", color=PALETTE["blue"], ms=3.5)
    ax2.set_xscale("log")
    tidy(ax2, title="at-the-money volatility", xlabel="days to expiry", ylabel="%")
    ax3 = fig.add_axes((0.71, 0.15, 0.26, 0.6))
    ax3.loglog(t * 365, -skew, "o", color=PALETTE["violet"], ms=3.5, label="- d sigma / d ln K")
    slope, level = np.polyfit(np.log(t), np.log(np.maximum(-skew, 1e-6)), 1)
    ax3.loglog(t * 365, np.exp(level) * t**slope, color=PALETTE["ink"], lw=1.0, label=f"power law T^{slope:.2f}")
    tidy(ax3, title="at-the-money skew", xlabel="days to expiry")
    ax3.legend(fontsize=8)
    caption(fig, f"{len(ts)} expiries kept by the Day 3 cleaning; {int(ts['quotes'].sum()):,} quotes.")
    return fig


def densities(a: Analysis) -> Figure:
    fig = new_figure(14.0, 5.4)
    title_block(
        fig,
        "Risk-neutral densities and local volatility",
        "Left: the density of ln(S_T/F) implied by each fitted slice, positive by construction. Right: Dupire local volatility "
        "from the monthly surface, in closed form from the price interpolation.",
    )
    ax = fig.add_axes((0.05, 0.15, 0.4, 0.6))
    picks = _picks(len(a.fit.svi), 5)
    for colour, i in zip(SERIES, picks, strict=False):
        f = a.fit.svi[i]
        k = np.linspace(-0.6, 0.4, 400)
        ax.plot(k, density(f.slice, k), color=colour, lw=1.6, label=f"{365 * f.maturity:.0f} days")
    tidy(ax, title="densities of ln(S_T / F)", xlabel="ln(K/F)", ylabel="density")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.55, 0.15, 0.4, 0.6))
    times = np.linspace(float(a.monthly.maturities[0]), float(a.monthly.maturities[-1]), 40)
    k = np.linspace(-0.4, 0.2, 61)
    grid = np.array([local_vol(a.monthly, k, float(t)) for t in times])
    mesh = ax2.pcolormesh(k, 365 * times, 100 * np.clip(grid, 0, 1), cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax2, label="local volatility (%)")
    tidy(ax2, title="local volatility", xlabel="ln(K/F)", ylabel="days")
    caption(fig, "Densities from the closed-form Durrleman expression; local volatility clipped at 100% for the colour scale.")
    return fig


def calibration(a: Analysis) -> Figure:
    fig = new_figure(14.0, 5.0)
    title_block(
        fig,
        "Heston calibration residuals",
        "Model minus mid implied volatility, in units of each quote's bid-ask width: inside +-0.5 "
        "the model is inside the spread.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.9, 0.6))
    ax.axhspan(-0.5, 0.5, color=PALETTE["band"], zorder=0, label="inside the bid-ask spread")
    for i, (e, r) in enumerate(zip(a.expiries, a.heston.residuals, strict=True)):
        ax.scatter(
            np.full(r.size, i) + np.linspace(-0.3, 0.3, r.size), r / e.width, s=12, color=SERIES[i % len(SERIES)], zorder=3
        )
    ax.axhline(0, color=PALETTE["ink"], lw=0.7, zorder=2)
    ax.set_xticks(range(len(a.expiries)), [f"{365 * e.maturity:.0f}d" for e in a.expiries])
    tidy(ax, xlabel="expiry", ylabel="residual (bid-ask widths)")
    caption(fig, f"RMSE {100 * a.heston.rmse:.2f} vol points over {sum(e.k.size for e in a.expiries)} quotes.")
    return fig


def exotics(a: Analysis) -> Figure | None:
    if a.exotics is None:
        return None
    b = a.exotics
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Exotics under three models of the same surface",
        "Prices relative to local volatility, which reprices the surface's vanillas exactly. Path-dependent payoffs separate the "
        "models that the vanillas cannot.",
    )
    ax = fig.add_axes((0.24, 0.15, 0.72, 0.6))
    names = list(b.contracts)
    y = np.arange(len(names))
    for offset, (model, colour) in enumerate((("Black-Scholes", PALETTE["slate"]), ("Heston", PALETTE["green"]))):
        ratio = [100 * (b.prices[(model, n)].value / b.prices[("local volatility", n)].value - 1) for n in names]
        ax.barh(y + (offset - 0.5) * 0.36, ratio, height=0.34, color=colour, label=model)
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_yticks(y, [f"{n} ({b.prices[('local volatility', n)].value:.4f})" for n in names])
    ax.invert_yaxis()
    tidy(ax, xlabel="difference from local volatility (%)")
    ax.legend(fontsize=8)
    dates = 126 if a.config.quick else 252
    caption(
        fig,
        f"One year, {dates} monitoring dates, {a.config.exotic_paths:,} paths per model; "
        "local-volatility prices per unit of spot in brackets.",
    )
    return fig


def hedging(a: Analysis) -> Figure | None:
    if a.hedging is None:
        return None
    res = a.hedging
    tab = res.table
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Hedging a three-month at-the-money call",
        "Sold at the surface's price and hedged in a Heston and a local-volatility world fitted to the same surface, with each "
        "model's delta. The spread of deltas is model risk, and so is the hedging error each leaves.",
    )
    ax = fig.add_axes((0.16, 0.15, 0.26, 0.6))
    names = list(res.deltas)
    ax.barh(names, [res.deltas[n] for n in names], color=[PALETTE["slate"], PALETTE["blue"], PALETTE["red"], PALETTE["violet"]])
    ax.invert_yaxis()
    tidy(ax, title="delta at the sale", xlabel="delta")
    ax2 = fig.add_axes((0.52, 0.15, 0.45, 0.6))
    strategies = list(dict.fromkeys(tab["strategy"]))
    for j, world in enumerate(("Heston", "local volatility")):
        rows = tab[tab.world == world].set_index("strategy")
        values = [float(rows.loc[s, "std_pct_premium"]) if s in rows.index else np.nan for s in strategies]
        ax2.bar(np.arange(len(strategies)) + (j - 0.5) * 0.38, values, width=0.36, color=SERIES[j], label=f"{world} world")
    labels = [s.replace("minimum-variance", "min-variance").replace(" delta", "\ndelta") for s in strategies]
    ax2.set_xticks(np.arange(len(strategies)), labels, fontsize=8)
    tidy(ax2, title="std of hedged P&L", ylabel="% of premium")
    ax2.legend(fontsize=8)
    caption(
        fig, f"Premium {res.premium:.4f} of spot, implied volatility {res.implied:.1%}; {a.config.hedge_paths:,} paths per world."
    )
    return fig


def book(a: Analysis) -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "A dealer book's stress under each smile dynamic",
        "Short puts, long calls, delta-hedged with the index; repriced off the surface after each index move under each rule for "
        "how the smile moves with the index.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4 if a.historical is not None else 0.88, 0.6))
    for (name, values), colour in zip(
        a.ladder.items(), (PALETTE["muted"], PALETTE["slate"], PALETTE["violet"], PALETTE["blue"]), strict=False
    ):
        ax.plot(100 * a.ladder.index.to_numpy(), values / 1e6, "o-", ms=3, color=colour, lw=1.5, label=name)
    ax.axhline(0, color=PALETTE["ink"], lw=0.7)
    tidy(ax, title="P&L against the index move", xlabel="index move (%)", ylabel="$ million")
    ax.legend(fontsize=8)
    if a.historical is not None:
        hr = a.historical
        ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
        ax2.hist(hr.full / 1e6, bins=np.linspace(-1.5, 1.0, 101).tolist(), color=PALETTE["slate"])
        ax2.axvline(-hr.var() / 1e6, color=PALETTE["accent"], lw=1.4, label=f"99% VaR ${hr.var() / 1e3:,.0f}k (revalued)")
        ax2.axvline(
            -hr.var(which="greeks") / 1e6,
            color=PALETTE["blue"],
            lw=1.4,
            label=f"from Greeks ${hr.var(which='greeks') / 1e3:,.0f}k",
        )
        ax2.set_yscale("log")
        tidy(ax2, title="every day since 1990, applied to today's book", xlabel="one-day P&L ($ million)", ylabel="days")
        ax2.legend(fontsize=8)
    positions = ", ".join(f"{c:+.0f} {n}" for n, c in zip(a.book.names, a.book.contracts, strict=True))
    caption(fig, f"Positions: {positions}; contracts of $100 a point on an index of {a.book.spot:,.0f}.")
    return fig


def all_figures(a: Analysis) -> dict[str, Figure]:
    out: dict[str, Figure | None] = {
        "surface": surface(a),
        "smiles": smiles(a),
        "term-structure": term_structure(a),
        "densities-local-vol": densities(a),
        "heston-residuals": calibration(a),
        "exotics": exotics(a),
        "hedging": hedging(a),
        "book": book(a),
    }
    return {k: v for k, v in out.items() if v is not None}
