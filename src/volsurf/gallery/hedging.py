"""Day 8 charts: hedging under each model, thirty-six years of the S&P 500 and the VIX, and an options book's risk.

The first three charts need only the code; the S&P 500 charts are drawn from the local
option-chain snapshot and the local index and VIX history, and skipped where there is none.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from ..localvol.dynamics import monthly
from ..market import chains
from ..market.cleaning import prepare
from ..risk import book as bk
from ..risk import history
from ..risk.hedging import black_scholes_study
from ..risk.hedging import wrong_volatility as wrong_vol_world
from ..risk.models import CrossModel, cross_model, heston_study
from ..smile.build import SurfaceFit, build_surface
from ..stochvol.characteristic import Heston
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

SPX_HESTON = Heston(0.0130, 4.42, 0.0530, 1.98, -0.59)  # Day 5's calibration to the same snapshot
TEXTBOOK_HESTON = Heston(0.04, 2.0, 0.04, 0.6, -0.7)


@lru_cache(maxsize=1)
def _fit() -> SurfaceFit:
    return build_surface(prepare(chains.load("^SPX")))


@lru_cache(maxsize=1)
def _day() -> str:
    return chains.load("^SPX").as_of.strftime("%d %B %Y")


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    return history.load()


def _skew_30d(fit: SurfaceFit) -> tuple[float, float]:
    """The 30-day expiry's at-the-money volatility and skew, d sigma / d ln K at the money."""
    e = min(fit.expiries, key=lambda e: abs(e.maturity * 365 - 30))
    iv = fit.surface.implied_vol(np.array([-0.01, 0.0, 0.01]), e.maturity)
    return float(iv[1]), float((iv[2] - iv[0]) / 0.02)


def _plain_ticks(ax: Axes, values: Sequence[float]) -> None:
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(list(values), [f"{v:g}" for v in values])


def hedging_frequency() -> Figure:
    study = black_scholes_study()
    t = study.table
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Hedging in Black-Scholes' world, more and more often",
        "A three-month at-the-money call, sold at its value and hedged in its own model. A delta "
        "hedge leaves an error that falls "
        "as one over the square root of the number of rebalances, exactly as Derman and Kamal's formula says. Adding a gamma "
        "hedge with a six-month call removes the leading term, and the error falls faster.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    premium = float(t["premium"].iloc[0])
    ax.loglog(
        t["rebalances"],
        100 * t["derman_kamal"] / premium,
        "--",
        color=PALETTE["ink"],
        lw=1.2,
        label="Derman-Kamal: sqrt(pi/4) vega sigma / sqrt(N)",
    )
    ax.loglog(t["rebalances"], 100 * t["delta"] / premium, "o-", color=PALETTE["blue"], ms=5, lw=1.6, label="delta hedge")
    ax.loglog(
        t["rebalances"],
        100 * t["delta_gamma"] / premium,
        "s-",
        color=PALETTE["green"],
        ms=5,
        lw=1.6,
        label="delta and gamma hedge",
    )
    slope = np.polyfit(np.log(t["rebalances"]), np.log(t["delta_gamma"]), 1)[0]
    ax.text(0.03, 0.05, f"delta-gamma slope {slope:.2f}", transform=ax.transAxes, fontsize=8.5, color=PALETTE["green"])
    _plain_ticks(ax, t["rebalances"])
    tidy(
        ax,
        title="hedging error against rebalancing",
        xlabel="rebalances over three months",
        ylabel="std of hedged P&L (% of premium)",
    )
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    bins = np.linspace(-25, 25, 81)
    for name, colour in (("delta", PALETTE["blue"]), ("delta and gamma", PALETTE["green"])):
        pnl = 100 * study.pnl[name] / premium
        ax2.hist(pnl, bins=bins.tolist(), color=colour, alpha=0.55, label=f"{name}: std {pnl.std():.1f}%")
    tidy(ax2, title=f"P&L at {study.shown} rebalances (daily)", xlabel="hedged P&L (% of premium)", ylabel="paths")
    ax2.legend(fontsize=8)
    caption(
        fig,
        "20,000 antithetic paths, volatility 20%, rate 4%, dividend yield 1%. The gamma hedge is a "
        "six-month call of the same strike.",
    )
    return fig


def wrong_volatility() -> Figure:
    w = wrong_vol_world()
    fig = new_figure(14.0, 5.8)
    corr = float(np.corrcoef(w.pnl, w.predicted)[0, 1])
    corr_vol = float(np.corrcoef(w.pnl, w.realised_vol**2)[0, 1])
    title_block(
        fig,
        "Hedging at the wrong volatility",
        "A call sold and delta-hedged at 20% volatility in a world where it is 25%. Path by path the P&L is the option's gamma "
        "times the gap between hedging and realised variance, summed where it was realised - the formula on the left. Realised "
        f"volatility alone explains much less (right): correlation {corr_vol:.2f}, against {corr:.3f} for the gamma formula.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    p = 100 * w.pnl / w.premium
    q = 100 * w.predicted / w.premium
    ax.scatter(q, p, s=2, color=PALETTE["blue"], alpha=0.25, rasterized=True)
    lo, hi = float(min(np.min(p), np.min(q))), float(max(np.max(p), np.max(q)))
    ax.plot([lo, hi], [lo, hi], color=PALETTE["ink"], lw=0.9)
    tidy(
        ax,
        title="P&L against the gamma formula",
        xlabel="1/2 sum Gamma S^2 (sigma_h^2 dt - (dS/S)^2), % of premium",
        ylabel="hedged P&L (% of premium)",
    )
    ax.text(
        0.03,
        0.92,
        f"correlation {corr:.3f}\nresidual std {np.std(p - q):.1f}% of premium",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
    )
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    ax2.scatter(100 * w.realised_vol, p, s=2, color=PALETTE["violet"], alpha=0.25, rasterized=True)
    ax2.axvline(20, color=PALETTE["muted"], lw=0.8, ls="--")
    ax2.axhline(0, color=PALETTE["ink"], lw=0.7)
    tidy(
        ax2,
        title="P&L against realised volatility",
        xlabel="realised volatility over the life (%)",
        ylabel="hedged P&L (% of premium)",
    )
    caption(
        fig,
        f"Three-month at-the-money call, daily hedging, 20,000 paths. Mean P&L {100 * w.pnl.mean() / w.premium:.1f}% of premium.",
    )
    return fig


def heston_hedging() -> Figure:
    study = heston_study(TEXTBOOK_HESTON)
    t = study.table
    premium = float(t["premium"].iloc[0])
    fig = new_figure(14.0, 5.8)
    final = t[t["rebalances"] == t["rebalances"].max()].set_index("strategy")["std"] / premium * 100
    title_block(
        fig,
        "In Heston's world a delta hedge stops improving",
        "Volatility is a second source of risk, and no delta reaches it: rebalancing more often "
        "stops helping. Black-Scholes' delta "
        "at the implied volatility beats Heston's own delta, which ignores that volatility falls when the index rises; the "
        "minimum-variance delta does better still. Only a second option, hedging vega, lets the error keep falling.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    colours = {
        "Black-Scholes delta at the implied volatility": PALETTE["slate"],
        "Heston delta": PALETTE["red"],
        "minimum-variance delta": PALETTE["blue"],
        "delta and vega, with a longer call": PALETTE["green"],
    }
    for name, colour in colours.items():
        rows = t[t["strategy"] == name]
        ax.loglog(rows["rebalances"], 100 * rows["std"] / premium, "o-", ms=4, color=colour, lw=1.6, label=name)
    _plain_ticks(ax, sorted(t["rebalances"].unique()))
    tidy(
        ax,
        title="hedging error against rebalancing",
        xlabel="rebalances over three months",
        ylabel="std of hedged P&L (% of premium)",
    )
    ax.legend(fontsize=7.5)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    bins = np.linspace(-120, 80, 81)
    for name in ("Black-Scholes delta at the implied volatility", "minimum-variance delta", "delta and vega, with a longer call"):
        label = f"{final[name]:.0f}%: {name}"
        ax2.hist(100 * study.pnl[name] / premium, bins=bins.tolist(), color=colours[name], alpha=0.5, label=label)
    tidy(ax2, title="P&L with daily rebalancing", xlabel="hedged P&L (% of premium)", ylabel="paths")
    ax2.legend(fontsize=7.5, loc="upper left")
    caption(
        fig,
        "Three-month at-the-money call; Heston v0 = theta = 0.04, kappa 2, vol of vol 0.6, rho "
        "-0.7; QE paths, 20,000; prices and "
        f"Greeks from a COS grid. Implied volatility at sale {study.implied:.1%}. Rate 4%, dividend yield 1%.",
    )
    return fig


@lru_cache(maxsize=1)
def _cross() -> CrossModel:
    fit = _fit()
    e = min(fit.expiries, key=lambda e: abs(e.maturity - 0.25))
    t = e.maturity
    r = float(-np.log(e.discount) / t)
    b = float(np.log(e.forward / (fit.surface.spot or e.forward)) / t)
    return cross_model(monthly(fit.surface), SPX_HESTON, t, r, b)


def hedging_across_models() -> Figure:
    res = _cross()
    fig = new_figure(14.0, 6.0)
    tab = res.table
    lv_in_lv = float(
        tab[(tab.world == "local volatility") & (tab.strategy == "local-volatility delta")]["std_pct_premium"].iloc[0]
    )
    best_h = float(tab[tab.world == "Heston"]["std_pct_premium"].min())
    title_block(
        fig,
        "Hedging the S&P 500 with the wrong model",
        f"S&P 500, {_day()}: a three-month at-the-money call sold at the market's price. Fitted to the same surface, the models "
        f"disagree about its delta, from {min(res.deltas.values()):.2f} to {max(res.deltas.values()):.2f}. In local volatility's "
        f"world its own delta leaves {lv_in_lv:.0f}% of the premium; in the calibrated Heston world no delta leaves less than "
        f"{best_h:.0f}%, and the best is the minimum-variance one.",
    )
    ax = fig.add_axes((0.16, 0.15, 0.26, 0.6))
    names = list(res.deltas)
    colours = [PALETTE["slate"], PALETTE["blue"], PALETTE["red"], PALETTE["violet"]]
    ax.barh(names, [res.deltas[n] for n in names], color=colours)
    for i, n in enumerate(names):
        ax.text(res.deltas[n] + 0.01, i, f"{res.deltas[n]:.3f}", va="center", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    tidy(ax, title="delta at the sale", xlabel="delta")
    ax2 = fig.add_axes((0.52, 0.15, 0.45, 0.6))
    strategies = list(dict.fromkeys(tab["strategy"]))
    worlds = ["Heston", "local volatility"]
    width = 0.38
    for j, world in enumerate(worlds):
        rows = tab[tab.world == world].set_index("strategy")
        values = [float(rows.loc[s, "std_pct_premium"]) if s in rows.index else np.nan for s in strategies]
        ax2.bar(
            np.arange(len(strategies)) + (j - 0.5) * width, values, width=width * 0.95, color=SERIES[j], label=f"{world} world"
        )
    labels = [s.replace("minimum-variance", "min-variance").replace(" delta", "\ndelta") for s in strategies]
    ax2.set_xticks(np.arange(len(strategies)), labels, fontsize=8)
    tidy(ax2, title="std of hedged P&L, daily rebalancing", ylabel="% of premium")
    ax2.legend(fontsize=8)
    mean_h = 100 * float(tab[tab.world == "Heston"]["mean"].mean()) / res.premium
    caption(
        fig,
        f"Premium {res.premium:.4f} of spot (implied volatility {res.implied:.1%}); 20,000 paths "
        f"per world, 63 rebalances. Heston as "
        f"calibrated on Day 5 (vol of vol 1.98); its world's mean P&L, {mean_h:.1f}% of premium, is its pricing error.",
    )
    return fig


def variance_premium() -> Figure:
    h = _history()
    trades = history.backtest(h).table
    monthly_trades = trades.iloc[::21]
    fig = new_figure(15.0, 6.0)
    corr = float(np.corrcoef(trades["pnl"], trades["predicted"])[0, 1])
    share = float((trades["pnl"] > 0).mean())
    ratio = float(trades["pnl"].mean() / trades["premium"].mean())
    worst = trades.nsmallest(1, "pnl").iloc[0]
    title_block(
        fig,
        "Thirty-six years of selling S&P 500 straddles at the VIX",
        f"A 30-day at-the-money straddle sold at the VIX every trading day since {h.index[0].year} "
        f"and delta-hedged daily. It made "
        f"money {share:.0%} of the time and kept {ratio:.0%} of the premium on average: implied volatility exceeded realised by "
        f"{100 * float((trades['vix'] - trades['realised']).mean()):.1f} points. The losses come together, in 2008 and 2020, and "
        f"the gamma formula explains each trade (correlation {corr:.3f}).",
    )
    ax = fig.add_axes((0.05, 0.15, 0.27, 0.6))
    ax.plot(monthly_trades["start"], 100 * monthly_trades["pnl"].cumsum(), color=PALETTE["blue"], lw=1.5)
    tidy(ax, title="cumulative P&L, one straddle a month", ylabel="% of the index")
    ax2 = fig.add_axes((0.38, 0.15, 0.27, 0.6))
    ax2.scatter(100 * trades["predicted"], 100 * trades["pnl"], s=2, color=PALETTE["violet"], alpha=0.3, rasterized=True)
    lim = (
        100 * float(min(trades["pnl"].min(), trades["predicted"].min())),
        100 * float(max(trades["pnl"].max(), trades["predicted"].max())),
    )
    ax2.plot(lim, lim, color=PALETTE["ink"], lw=0.8)
    tidy(
        ax2,
        title="each trade against the gamma formula",
        xlabel="1/2 sum Gamma S^2 (sigma^2 dt - r^2), % of index",
        ylabel="hedged P&L (% of index)",
    )
    ax3 = fig.add_axes((0.71, 0.15, 0.26, 0.6))
    x = 100 * trades["pnl"] / trades["premium"]
    ax3.hist(x, bins=np.linspace(-250, 100, 71).tolist(), color=PALETTE["slate"])
    ax3.axvline(float(x.mean()), color=PALETTE["accent"], lw=1.4, label=f"mean {x.mean():.0f}%")
    tidy(ax3, title="P&L per trade", xlabel="% of the premium", ylabel="trades")
    ax3.set_yscale("log")
    ax3.legend(fontsize=8)
    caption(
        fig,
        f"{len(trades):,} overlapping trades; S&P 500 and VIX closes from Yahoo Finance, rate from "
        f"the 13-week bill, dividend yield 1.5%. Worst: "
        f"sold {worst['start'].date()} at VIX {100 * worst['vix']:.0f}, realised {100 * worst['realised']:.0f}, lost "
        f"{-100 * worst['pnl']:.1f}% of the index. The VIX exceeds at-the-money volatility, so the premium is an upper bound.",
    )
    return fig


def spot_vol_dynamics() -> Figure:
    h = _history()
    sv = history.spot_vol(h)
    atm, skew = _skew_30d(_fit())
    errors = history.hedge_errors(h)
    errors["year"] = pd.to_datetime(errors["date"]).dt.year
    by_year = errors.groupby("year").apply(lambda g: 1.0 - g["error_mv"].var() / g["error_bs"].var(), include_groups=False)
    overall = 1.0 - float(errors["error_mv"].var() / errors["error_bs"].var())
    c = sv.beta / skew
    fig = new_figure(15.0, 6.0)
    title_block(
        fig,
        "How the S&P 500's volatility moves with the index",
        f"Daily, the VIX moves {sv.beta:.2f} volatility points per point of index return (R^2 "
        f"{sv.r2:.2f}). Today's 30-day smile has a "
        f"slope of {skew:.2f}, so the market moves at-the-money volatility {c:.1f} times as far as sticky strike would: between "
        f"sticky strike (1) and local volatility (2). Adding that move to the delta - the minimum-variance delta - removes "
        f"{overall:.0%} of the variance of daily hedging errors.",
    )
    ret = np.log(h["spot"]).diff().dropna()
    dv = h["vix"].diff().reindex(ret.index)
    ax = fig.add_axes((0.05, 0.15, 0.27, 0.6))
    ax.scatter(100 * ret, 100 * dv, s=2, color=PALETTE["blue"], alpha=0.2, rasterized=True)
    xs = np.linspace(-12, 12, 3)
    ax.plot(xs, sv.beta * xs, color=PALETTE["ink"], lw=1.0, label=f"slope {sv.beta:.2f}")
    tidy(ax, title="daily VIX change against index return", xlabel="index log-return (%)", ylabel="VIX change (vol points)")
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.38, 0.15, 0.27, 0.6))
    ax2.plot(sv.rolling.index, sv.rolling.to_numpy(), color=PALETTE["blue"], lw=1.2, label="one-year rolling slope")
    for value, name, colour in (
        (0.0, "sticky moneyness", PALETTE["muted"]),
        (skew, "sticky strike", PALETTE["slate"]),
        (2 * skew, "local volatility", PALETTE["violet"]),
    ):
        ax2.axhline(value, color=colour, ls="--", lw=1.0, label=f"{name} ({value:.2f})")
    tidy(ax2, title="d sigma / d ln S, and what each dynamic implies", ylabel="vol points per point of return")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax3 = fig.add_axes((0.71, 0.15, 0.26, 0.6))
    ax3.bar(by_year.index, 100 * by_year.to_numpy(), color=PALETTE["green"])
    ax3.axhline(100 * overall, color=PALETTE["ink"], lw=1.0, label=f"all years {overall:.0%}")
    tidy(ax3, title="variance removed by the minimum-variance delta", xlabel="year", ylabel="% of daily hedging-error variance")
    ax3.legend(fontsize=8)
    caption(
        fig,
        f"VIX and S&P 500 closes since 1990. Skew: the 30-day S&P 500 smile on {_day()}, "
        f"at-the-money volatility {atm:.1%}. Hedging errors: "
        "30-day at-the-money calls struck monthly, marked daily at the VIX; the slope is estimated only on data before each day.",
    )
    return fig


def book_risk() -> Figure:
    fit = _fit()
    book = bk.build(fit)
    h = _history()
    sv = history.spot_vol(h)
    _, skew = _skew_30d(fit)
    dynamics = {
        "sticky moneyness": 0.0,
        "sticky strike": 1.0,
        "local volatility": 2.0,
        f"market ({sv.beta / skew:.1f})": sv.beta / skew,
    }
    moves = np.linspace(-0.2, 0.1, 31)
    ladder = bk.stress_ladder(book, moves, dynamics)
    day = bk.historical(book, h, 1)
    ten = bk.historical(book, h, 10)
    fig = new_figure(15.0, 6.2)
    at = float(ladder.loc[ladder.index[np.argmin(np.abs(moves + 0.1))]].iloc[0])
    title_block(
        fig,
        "An options book's risk under each smile dynamic",
        "A dealer's S&P 500 book - short puts sold for protection, long calls bought from overwriters, delta-hedged with the "
        f"index - is short skew. If the index falls 10% it loses ${-at / 1e6:.1f}m when the smile moves with the index, and "
        f"${-float(ladder.iloc[np.argmin(np.abs(moves + 0.1)), 2]) / 1e6:.1f}m under local "
        f"volatility's dynamic. Its Greeks explain quiet "
        "days and misstate crashes.",
    )
    ax = fig.add_axes((0.05, 0.15, 0.27, 0.6))
    for (name, values), colour in zip(
        ladder.items(), (PALETTE["muted"], PALETTE["slate"], PALETTE["violet"], PALETTE["blue"]), strict=True
    ):
        ax.plot(100 * moves, values / 1e6, color=colour, lw=1.8 if name.startswith("market") else 1.3, label=name)
    ax.axhline(0, color=PALETTE["ink"], lw=0.7)
    tidy(ax, title="stress: P&L against the index move", xlabel="index move (%)", ylabel="P&L ($ million)")
    ax.legend(fontsize=7.5)
    ax2 = fig.add_axes((0.38, 0.15, 0.27, 0.6))
    ax2.hist(day.full / 1e6, bins=np.linspace(-1.5, 1.0, 101).tolist(), color=PALETTE["slate"])
    for value, colour, label in (
        (-day.var(), PALETTE["accent"], f"99% VaR, full revaluation: ${day.var() / 1e3:,.0f}k"),
        (-day.var(which="greeks"), PALETTE["blue"], f"99% VaR from the Greeks: ${day.var(which='greeks') / 1e3:,.0f}k"),
    ):
        ax2.axvline(value / 1e6, color=colour, lw=1.4, label=label)
    ax2.set_yscale("log")
    tidy(ax2, title="every day since 1990, applied to today's book", xlabel="one-day P&L ($ million)", ylabel="days")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax3 = fig.add_axes((0.71, 0.15, 0.26, 0.6))
    size = np.abs(np.log(h["spot"].to_numpy()[10:] / h["spot"].to_numpy()[:-10]))
    sc = ax3.scatter(ten.full / 1e6, ten.greeks / 1e6, c=100 * size, s=3, cmap="viridis", rasterized=True)
    lo = float(min(np.min(ten.full), np.min(ten.greeks))) / 1e6
    hi = float(max(np.max(ten.full), np.max(ten.greeks))) / 1e6
    ax3.plot([lo, hi], [lo, hi], color=PALETTE["ink"], lw=0.8)
    cbar = fig.colorbar(sc, ax=ax3, fraction=0.05, pad=0.02)
    cbar.set_label("|index move| (%)")
    tidy(
        ax3,
        title="ten-day P&L: Greeks against revaluation",
        xlabel="full revaluation ($ million)",
        ylabel="second-order Greeks ($ million)",
    )
    positions = ", ".join(f"{c:+.0f} {n}" for n, c in zip(book.names, book.contracts, strict=True))
    caption(
        fig,
        f"Positions, {_day()}: {positions} (contracts of $100 a point). "
        f"Ten-day 97.5% ES: ${ten.es() / 1e6:.1f}m revalued, ${ten.es(which='greeks') / 1e6:.1f}m from Greeks.",
    )
    return fig
