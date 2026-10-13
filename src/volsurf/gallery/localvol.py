"""Day 6 charts: Dupire local volatility from the arbitrage-free surface, SABR and its failures, and smile dynamics.

The SABR accuracy and density charts need only the code and QuantLib; the others are
drawn from the local S&P 500 snapshot and are skipped where there is none.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import QuantLib as ql
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..localvol.dupire import implied_from_calls, local_variance, solve_forward
from ..localvol.dynamics import black_delta, monthly, sabr_delta, smile_moves
from ..localvol.sabr import SABR, calibrate, density_from_vols, hagan_vol, reference_vols
from ..market import chains
from ..market.black import black_price
from ..market.cleaning import prepare
from ..market.implied import implied_vol
from ..smile.build import SurfaceFit, build_surface
from ..smile.interpolation import Surface
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

Array = NDArray[np.float64]
EQUITY = SABR(0.2, 1.0, -0.7, 1.0)
RATES = SABR(0.035, 0.5, -0.3, 0.6)


@lru_cache(maxsize=1)
def _fit() -> SurfaceFit:
    return build_surface(prepare(chains.load("^SPX")))


@lru_cache(maxsize=1)
def _day() -> str:
    return chains.load("^SPX").as_of.strftime("%d %B %Y")


def _round_trip_error(surface: Surface, targets: Array, space: int, per_year: int, short_end: str) -> Array:
    sol = solve_forward(
        lambda k, t: local_variance(surface, k, t, short_end), targets, space_steps=space, steps_per_year=per_year
    )
    errors = []
    for i, t in enumerate(targets):
        theta = float(surface.thetas[np.searchsorted(surface.maturities, t)])
        sel = (sol.k > -3 * np.sqrt(theta)) & (sol.k < 1.2 * np.sqrt(theta))
        model = implied_from_calls(sol.k[sel], sol.calls[i][sel], float(t))
        errors.append(1e4 * float(np.nanmax(np.abs(model - surface.implied_vol(sol.k[sel], float(t))))))
    return np.array(errors)


def local_vol_surface() -> Figure:
    fit = _fit()
    full, month = fit.surface, monthly(fit.surface)
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "S&P 500 local volatility, in closed form from the arbitrage-free surface",
        f"{_day()}. Dupire's local variance is twice the time derivative of the call over its "
        f"density; on the Day 4 surface both are "
        "exact, and local variance is positive by construction. Across daily weeklies, though, small differences between "
        "neighbouring slices are divided by small time steps, and the wings turn to noise.",
    )
    ax = fig.add_axes((0.055, 0.15, 0.27, 0.6))
    k = np.linspace(-0.5, 0.15, 261)
    for index, days in enumerate((30, 91, 182, 365, 730)):
        t = days / 365.0
        ax.plot(k, 100 * np.sqrt(local_variance(month, k, t)), color=SERIES[index], lw=1.6, label=f"{days} days")
        ax.plot(k, 100 * month.implied_vol(k, t), color=SERIES[index], lw=0.9, ls="--")
    ax.set_ylim(0, 80)
    tidy(ax, title="local (solid) and implied (dashed) volatility", xlabel="ln(K/F)", ylabel="volatility (%)")
    ax.legend(fontsize=7.5)
    ax2 = fig.add_axes((0.385, 0.15, 0.27, 0.6))  # smiles
    ts = np.geomspace(month.maturities[0], 3.0, 80)
    ax2.plot(
        ts,
        [100 * float(np.sqrt(local_variance(month, np.array([0.0]), t))[0]) for t in ts],
        color=PALETTE["blue"],
        lw=1.6,
        label="local, at the money",
    )
    ax2.plot(
        ts,
        [100 * float(month.implied_vol(np.array([0.0]), t)[0]) for t in ts],
        color=PALETTE["slate"],
        lw=1.2,
        ls="--",
        label="implied, at the money",
    )
    ax2.set_xscale("log")
    tidy(ax2, title="term structure (monthly expiries)", xlabel="years", ylabel="volatility (%)")
    ax2.legend(fontsize=8)
    ax3 = fig.add_axes((0.715, 0.15, 0.26, 0.6))
    early = np.linspace(full.maturities[0] * 1.001, 45 / 365, 300)
    for kk, colour in ((-0.1, PALETTE["red"]), (-0.05, PALETTE["accent"]), (0.0, PALETTE["blue"])):
        ax3.plot(
            365 * early,
            [100 * float(np.sqrt(local_variance(full, np.array([kk]), t))[0]) for t in early],
            color=colour,
            lw=1.1,
            label=f"all expiries, k = {kk:+.2f}",
        )
    for kk, colour in ((-0.1, PALETTE["red"]), (0.0, PALETTE["blue"])):
        ax3.plot(
            365 * early,
            [100 * float(np.sqrt(local_variance(month, np.array([kk]), t))[0]) for t in early],
            color=colour,
            lw=2.0,
            ls=":",
            label=f"monthly, k = {kk:+.2f}",
        )
    ax3.set_yscale("log")
    tidy(ax3, title="the first six weeks: noise from the weeklies", xlabel="days", ylabel="local volatility (%)")
    ax3.legend(fontsize=7)
    caption(
        fig,
        f"Monthly expiries: the {len(month.maturities)} of the {len(full.maturities)} fitted ones "
        f"from 25 days, each at least 25% longer than the last.",
    )
    return fig


def dupire_round_trip() -> Figure:
    fit = _fit()
    full, month = fit.surface, monthly(fit.surface)
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Repricing the surface from its own local volatility",
        "The forward Dupire equation solved from expiry with the local volatility, against the surface's implied volatilities. "
        "Before the first expiry the price interpolation keeps an atom at the forward, which no diffusion can carry: from it the "
        "solution stops improving and drifts back up as the grid is refined. A self-similar short end fixes that; the daily "
        "weeklies' noise remains.",
    )
    grids = ((600, 250), (1200, 500), (2400, 1000), (4800, 2000))
    targets_full = full.maturities[full.maturities > 0.08][:4]
    targets_month = month.maturities[month.maturities > 0.08][:4]
    variants = (
        ("all expiries, interpolation's short end (atom)", full, targets_full, "interpolation", PALETTE["red"]),
        ("all expiries, self-similar short end", full, targets_full, "self-similar", PALETTE["accent"]),
        ("monthly expiries, self-similar short end", month, targets_month, "self-similar", PALETTE["blue"]),
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    for label, surface, targets, short_end, colour in variants:
        worst = [float(np.max(_round_trip_error(surface, targets, nx, nt, short_end))) for nx, nt in grids]
        ax.loglog([g[0] for g in grids], worst, "o-", ms=4, lw=1.6, color=colour, label=label)
    tidy(
        ax,
        title="largest error over the first four expiries after a month",
        xlabel="space steps (time steps rise with them)",
        ylabel="basis points of volatility",
    )
    ax.legend(fontsize=7.5)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    errors = _round_trip_error(month, month.maturities, 4800, 2000, "self-similar")
    ax2.semilogx(month.maturities, errors, "o-", ms=4, color=PALETTE["blue"], lw=1.6)
    tidy(
        ax2,
        title="monthly surface, finest grid: every expiry",
        xlabel="years to expiry",
        ylabel="largest error (bp of volatility)",
    )
    caption(fig, "Errors over strikes within three ATM standard deviations below and 1.2 above the forward.")
    return fig


def sabr_accuracy() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Hagan's SABR formula against the model it approximates",
        "Implied volatility from Hagan's expansion minus an exact reference: for beta = 1 the forward is log-normal given the "
        "volatility path, so prices are Black prices averaged over simulated paths (Willard's conditioning). The expansion is "
        "accurate while nu^2 T is below about 0.3; past 1 it is off by tens, then hundreds, of "
        "basis points, most in the put wing.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    for index, t in enumerate((0.1, 0.5, 1.0, 2.0)):
        k = np.exp(np.linspace(-2, 1, 13) * 0.2 * np.sqrt(t))
        ref = reference_vols(1.0, k, t, EQUITY, paths=200_000, steps=200, seed=3)
        error = 1e4 * (hagan_vol(1.0, k, t, EQUITY) - ref.vols)
        ax.errorbar(
            np.log(k),
            error,
            yerr=2e4 * ref.stderr,
            fmt="o-",
            ms=3,
            lw=1.4,
            capsize=2,
            color=SERIES[index],
            label=f"T = {t:g} (nu^2 T = {EQUITY.nu**2 * t:g})",
        )
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    tidy(
        ax,
        title=f"alpha {EQUITY.alpha}, beta 1, rho {EQUITY.rho}, nu {EQUITY.nu}",
        xlabel="ln(K/F)",
        ylabel="Hagan minus exact (bp of vol)",
    )
    ax.legend(fontsize=8)
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    for index, nu in enumerate((0.3, 0.6, 1.0, 1.5)):
        p = SABR(0.2, 1.0, -0.7, nu)
        xs, ys = [], []
        for t in (0.1, 0.25, 0.5, 1.0, 2.0):
            ref = reference_vols(1.0, np.array([1.0]), t, p, paths=100_000, steps=200, seed=5)
            xs.append(nu * nu * t)
            ys.append(1e4 * float(hagan_vol(1.0, np.array([1.0]), t, p)[0] - ref.vols[0]))
        ax2.plot(xs, ys, "o-", ms=4, color=SERIES[index], lw=1.4, label=f"nu = {nu}")
    ax2.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    ax2.set_xscale("log")
    tidy(ax2, title="at the money, by nu^2 T", xlabel="nu^2 T", ylabel="Hagan minus exact (bp of vol)")
    ax2.legend(fontsize=8)
    caption(
        fig,
        "Error bars: two standard errors of the reference. The reference agrees with QuantLib's "
        "SABR finite-difference engine at beta = 0.999 to within its noise.",
    )
    return fig


def _fd_vols(f: float, strikes: Array, days: int, p: SABR) -> Array:
    today = ql.Date(13, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    engine = ql.FdSabrVanillaEngine(f, p.alpha, p.beta, p.nu, p.rho, flat, 200, 400, 100, 1, 0.8)
    prices = []
    for k in strikes:
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call if k >= f else ql.Option.Put, float(k)), ql.EuropeanExercise(today + days)
        )
        option.setPricingEngine(engine)
        prices.append(option.NPV())
    return implied_vol(np.array(prices), f, strikes, days / 365.0, 1.0, strikes >= f)


def sabr_density() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Where Hagan's formula stops being a model",
        "A rates-style SABR (forward 3%, beta 0.5). At long maturities the expansion's smile no "
        "longer belongs to any distribution: "
        "the density it implies turns negative at low strikes, and its volatilities part from the model's own, priced by "
        "QuantLib's SABR finite-difference engine.",
    )
    f = 0.03
    ax = fig.add_axes((0.06, 0.15, 0.4, 0.6))
    ax2 = fig.add_axes((0.56, 0.15, 0.4, 0.6))
    strikes = np.linspace(0.002, 0.08, 40)
    fine = np.linspace(0.0005, 0.08, 1600)
    for index, years in enumerate((1, 5, 10)):
        t = float(years)
        colour = SERIES[index]
        ax.plot(100 * strikes, 100 * hagan_vol(f, strikes, t, RATES), color=colour, lw=1.6, label=f"Hagan, {years}y")
        model = _fd_vols(f, strikes, 365 * years, RATES)
        # where the PDE price is too small to invert, the volatility is left out rather than drawn at zero
        model = np.where(np.isfinite(model) & (model > 0.01), model, np.nan)
        ax.plot(100 * strikes, 100 * model, color=colour, lw=1.2, ls="--", label=f"model (FD), {years}y")
        density = density_from_vols(f, fine, t, hagan_vol(f, fine, t, RATES))
        ax2.plot(100 * fine[1:-1], density, color=colour, lw=1.4, label=f"{years}y")
    ax2.axhline(0.0, color=PALETTE["ink"], lw=0.7)
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-150, 100)
    tidy(ax, title="implied volatility: Hagan (solid) and the model (dashed)", xlabel="strike (%)", ylabel="volatility (%)")
    ax.legend(fontsize=7, ncol=2)
    tidy(ax2, title="density implied by Hagan's volatilities", xlabel="strike (%)", ylabel="density of F_T")
    ax2.legend(fontsize=8)
    caption(
        fig,
        f"alpha {RATES.alpha}, beta {RATES.beta}, rho {RATES.rho}, nu {RATES.nu}. Densities by "
        f"second differences of call prices; below zero is arbitrage.",
    )
    return fig


@lru_cache(maxsize=1)
def _sabr_fits() -> list[tuple[float, SABR, float, float]]:
    fit = _fit()
    month = monthly(fit.surface)
    out = []
    for expiry, quotes, svi in zip(fit.expiries, fit.quotes, fit.svi, strict=True):
        if not np.any(np.isclose(month.maturities, expiry.maturity)):
            continue
        sel = (quotes.k / np.sqrt(expiry.maturity) > -2.5) & (quotes.k / np.sqrt(expiry.maturity) < 1.0)
        sf = calibrate(1.0, np.exp(quotes.k[sel]), expiry.maturity, quotes.iv[sel], quotes.weight[sel])
        svi_rmse = float(np.sqrt(np.mean((svi.slice.implied_vol(quotes.k[sel], expiry.maturity) - quotes.iv[sel]) ** 2)))
        out.append((expiry.maturity, sf.params, sf.rmse, svi_rmse))
    return out


def sabr_spx() -> Figure:
    fits = _sabr_fits()
    t = np.array([f[0] for f in fits])
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "SABR fitted to each S&P 500 expiry",
        f"{_day()}, monthly expiries, beta = 1, three parameters an expiry. Beyond two months SABR fits about as well as SVI; "
        "the shortest expiries it misses by a point. And its parameters move with maturity - vol-of-vol falls from over 3 to "
        "0.5 - which one SABR model, with one vol-of-vol, cannot do.",
    )
    ax = fig.add_axes((0.06, 0.15, 0.27, 0.6))
    ax.semilogx(t, [f[1].rho for f in fits], "o-", ms=4, color=PALETTE["blue"], label="rho")
    axb = ax.twinx()
    axb.semilogx(t, [f[1].nu for f in fits], "s-", ms=4, color=PALETTE["green"], label="nu (right)")
    axb.grid(False)
    tidy(ax, title="correlation and vol-of-vol", xlabel="years to expiry", ylabel="rho")
    ax.legend(fontsize=8, loc="upper left")
    axb.legend(fontsize=8, loc="upper right")
    ax2 = fig.add_axes((0.42, 0.15, 0.25, 0.6))
    ax2.semilogx(t, [100 * f[1].alpha for f in fits], "o-", ms=4, color=PALETTE["violet"])
    tidy(ax2, title="alpha", xlabel="years to expiry", ylabel="alpha (%)")
    ax3 = fig.add_axes((0.74, 0.15, 0.23, 0.6))
    ax3.semilogx(t, [100 * f[2] for f in fits], "o-", ms=4, color=PALETTE["blue"], label="SABR")
    ax3.semilogx(t, [100 * f[3] for f in fits], "o-", ms=4, color=PALETTE["slate"], label="SVI (Day 4)")
    tidy(ax3, title="fit error per expiry", xlabel="years to expiry", ylabel="RMSE (vol pts)")
    ax3.legend(fontsize=8)
    caption(
        fig,
        "Quotes within 2.5 ATM standard deviations below and 1 above the forward, weighted by "
        "bid-ask width; Obloj's form of the expansion.",
    )
    return fig


def smile_dynamics() -> Figure:
    fit = _fit()
    month = monthly(fit.surface)
    i = int(np.argmin(np.abs(month.maturities - 0.25)))
    t = float(month.maturities[i])
    fits = {round(f[0], 6): f[1] for f in _sabr_fits()}
    sabr = fits[round(t, 6)]
    shifts = np.log(np.linspace(0.94, 1.06, 9))
    strikes = np.exp(np.linspace(-0.12, 0.08, 41))
    move = smile_moves(month, sabr, t, shifts, strikes)
    atm_lv = np.array([np.interp(np.exp(s), strikes, row) for s, row in zip(shifts, move.local_vol, strict=True)])
    atm_sabr = np.array([np.interp(np.exp(s), strikes, row) for s, row in zip(shifts, move.sabr, strict=True)])
    atm_sticky = np.array([np.interp(np.exp(s), strikes, move.sticky_strike[0]) for s in shifts])
    skew = float((month.implied_vol(np.array([0.01]), t) - month.implied_vol(np.array([-0.01]), t))[0] / 0.02)
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "Same smile today, different smiles tomorrow",
        f"S&P 500, {t * 365:.0f} days. Local volatility is fixed in absolute strike, so when the index rises its at-the-money "
        "volatility falls along the local-volatility skew - about twice the implied skew, as Hagan et al. (2002) showed. SABR "
        "carries the smile with the forward. The difference is a delta.",
    )
    ax = fig.add_axes((0.055, 0.15, 0.27, 0.6))
    moves = 100 * (np.exp(shifts) - 1)
    ax.plot(
        moves,
        100 * atm_lv,
        "o-",
        ms=3.5,
        color=PALETTE["red"],
        lw=1.6,
        label=f"local vol: slope {np.polyfit(shifts, atm_lv, 1)[0]:.2f}",
    )
    ax.plot(
        moves,
        100 * atm_sabr,
        "o-",
        ms=3.5,
        color=PALETTE["blue"],
        lw=1.6,
        label=f"SABR: slope {np.polyfit(shifts, atm_sabr, 1)[0]:.2f}",
    )
    ax.plot(moves, 100 * atm_sticky, "--", color=PALETTE["slate"], lw=1.2, label=f"sticky strike: slope {skew:.2f}")
    tidy(ax, title="at-the-money volatility after the move", xlabel="move in the index (%)", ylabel="volatility (%)")
    ax.legend(fontsize=7.5)
    ax2 = fig.add_axes((0.385, 0.15, 0.27, 0.6))  # smiles
    for j in range(len(shifts)):
        if j not in (1, len(shifts) - 2):
            continue
        ax2.plot(strikes, 100 * move.local_vol[j], color=PALETTE["red"], lw=1.4, ls="-" if j == 1 else "--")
        ax2.plot(strikes, 100 * move.sabr[j], color=PALETTE["blue"], lw=1.4, ls="-" if j == 1 else "--")
    ax2.plot(strikes, 100 * move.sticky_strike[0], color=PALETTE["slate"], lw=2.0, label="today")
    ax2.plot([], [], color=PALETTE["red"], label="local vol")
    ax2.plot([], [], color=PALETTE["blue"], label="SABR")
    ax2.plot([], [], color=PALETTE["ink"], ls="-", label=f"index {100 * (np.exp(shifts[1]) - 1):+.1f}%")
    ax2.plot([], [], color=PALETTE["ink"], ls="--", label=f"index {100 * (np.exp(shifts[-2]) - 1):+.1f}%")
    tidy(ax2, title="smiles after the move, by absolute strike", xlabel="strike / today's forward", ylabel="implied vol (%)")
    ax2.legend(fontsize=7, loc="lower left")
    ax3 = fig.add_axes((0.80, 0.15, 0.17, 0.6))
    vol = float(month.implied_vol(np.array([0.0]), t)[0])
    up = smile_moves(month, sabr, t, np.array([np.log(1.01)]), np.array([1.0])).local_vol[0, 0]
    down = smile_moves(month, sabr, t, np.array([np.log(0.99)]), np.array([1.0])).local_vol[0, 0]
    lv_delta = float((black_price(1.01, 1.0, t, up) - black_price(0.99, 1.0, t, down)) / 0.02)
    deltas = {
        "Black (sticky strike)": black_delta(1.0, 1.0, t, vol),
        "local vol": lv_delta,
        "SABR, alpha held": sabr_delta(1.0, 1.0, t, sabr, bartlett=False),
        "SABR, Bartlett": sabr_delta(1.0, 1.0, t, sabr),
    }
    colours = [PALETTE["slate"], PALETTE["red"], PALETTE["violet"], PALETTE["blue"]]
    ax3.barh(list(deltas), list(deltas.values()), color=colours)
    ax3.invert_yaxis()
    for y, value in enumerate(deltas.values()):
        ax3.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=8)
    ax3.set_xlim(0, 0.75)
    tidy(ax3, title="delta of the at-the-money call", xlabel="dC/dF")
    caption(
        fig,
        f"Monthly expiries; SABR beta = 1 fitted to this expiry (rho {sabr.rho:.2f}, nu "
        f"{sabr.nu:.2f}); implied skew dIV/dk = {skew:.2f}.",
    )
    return fig


__all__ = ["dupire_round_trip", "local_vol_surface", "sabr_accuracy", "sabr_density", "sabr_spx", "smile_dynamics"]
