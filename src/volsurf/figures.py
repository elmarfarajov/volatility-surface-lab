"""Publication-style matplotlib figures for the analysis report and README."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, PercentFormatter

from .heston import HestonParams, heston_price_cos
from .implied_vol import implied_vol_black76

if TYPE_CHECKING:
    from .calibration import CalibrationResult
    from .hedging import HedgingExperiment
    from .market_data import MarketSnapshot
    from .surface import VolSurface
    from .svi import SVIFit

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
MARKET = "#1f2933"
SVI_COLOR = "#2f6fdb"
HESTON_COLOR = "#e0632b"
STRATEGY_COLORS = {
    "Black-Scholes delta @ implied vol": "#2f6fdb",
    "Heston delta": "#e0632b",
    "Heston minimum-variance delta": "#1b9e77",
    "Black-Scholes delta": "#2f6fdb",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#cbd2d9",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _plain_log_ticks(axis, lo: float, hi: float, candidates: tuple[float, ...]) -> None:
    ticks = [c for c in candidates if lo * 0.95 <= c <= hi * 1.05] or [lo, hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


_LOG_CANDIDATES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 8, 16, 32, 64, 128, 256, 10, 20, 40, 100)


def _heston_smile(params: HestonParams, forward: float, discount: float, maturity: float, strikes: np.ndarray) -> np.ndarray:
    is_call = strikes >= forward
    prices = heston_price_cos(forward, strikes, maturity, discount, params, is_call)
    return implied_vol_black76(prices, forward, strikes, maturity, discount, is_call)


def quoted_standardised_range(snapshot: MarketSnapshot) -> tuple[float, float]:
    """Standardised-moneyness window k/sqrt(T) that most expiries actually quote, so plots
    show the market rather than extrapolated wings."""
    lows = [e.quotes["k"].min() / np.sqrt(e.maturity) for e in snapshot.expiries]
    highs = [e.quotes["k"].max() / np.sqrt(e.maturity) for e in snapshot.expiries]
    return float(np.percentile(lows, 75)), float(np.percentile(highs, 25))


def surface_grid(surface: VolSurface, snapshot: MarketSnapshot, n_t: int = 40, n_z: int = 45):
    """Implied vol on a (k / sqrt(T), T) grid restricted to the quoted window."""
    maturities = np.linspace(surface.maturities[0], surface.maturities[-1], n_t)
    z = np.linspace(*quoted_standardised_range(snapshot), n_z)
    vol = np.array([surface.implied_vol(z * np.sqrt(t), t) for t in maturities])
    return z, maturities, vol


def surface_3d(surface: VolSurface, snapshot: MarketSnapshot) -> Figure:
    z, maturities, vol = surface_grid(surface, snapshot)
    zz, tt = np.meshgrid(z, maturities)

    fig = plt.figure(figsize=(8.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(zz, tt, 100 * vol, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    q = snapshot.quotes()
    q = q.assign(z=q["k"] / np.sqrt(q["maturity"]))
    sel = (q["z"] >= z[0]) & (q["z"] <= z[-1])
    ax.scatter(q.loc[sel, "z"], q.loc[sel, "maturity"], 100 * q.loc[sel, "iv_mid"], s=5, c=MARKET, alpha=0.55, depthshade=False)
    ax.set_xlabel("standardised moneyness  ln(K/F)/sqrt(T)")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol (%)")
    ax.set_title(f"{snapshot.ticker} implied volatility surface  (SVI, arbitrage-controlled)")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    return fig


def smiles(snapshot: MarketSnapshot, fits: list[SVIFit], calibration: CalibrationResult | None) -> Figure:
    n = len(snapshot.expiries)
    cols = 4 if n > 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, e, fit in zip(axes.ravel(), snapshot.expiries, fits, strict=False):
        q = e.quotes
        k = q["k"].to_numpy()
        lo = 100 * (q["iv_mid"] - q["iv_bid"]).fillna(0).clip(lower=0)
        hi = 100 * (q["iv_ask"] - q["iv_mid"]).fillna(0).clip(lower=0)
        ax.errorbar(
            k,
            100 * q["iv_mid"],
            yerr=[lo, hi],
            fmt="o",
            ms=2.2,
            color=MARKET,
            ecolor="#9aa5b1",
            elinewidth=0.7,
            label="market bid/mid/ask",
        )
        grid = np.linspace(k.min(), k.max(), 120)
        ax.plot(grid, 100 * fit.slice.implied_vol(grid), color=SVI_COLOR, lw=1.6, label="SVI")
        if calibration is not None:
            strikes = e.forward * np.exp(grid)
            ax.plot(
                grid,
                100 * _heston_smile(calibration.params, e.forward, e.discount, e.maturity, strikes),
                color=HESTON_COLOR,
                lw=1.4,
                ls="--",
                label="Heston",
            )
        ax.set_title(f"{e.expiry}  (T={e.maturity:.2f}y)", fontsize=9.5)
        ax.set_xlabel("ln(K/F)", fontsize=8.5)
    axes[0, 0].set_ylabel("implied vol (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{snapshot.ticker} volatility smiles: market vs SVI vs calibrated Heston", y=1.035, fontweight="bold")
    fig.tight_layout()
    return fig


def skew_term_structure(snapshot: MarketSnapshot, fits: list[SVIFit], calibration: CalibrationResult | None) -> Figure:
    """ATM skew |d sigma / d k| against maturity on log-log axes, with a power-law fit.

    Equity-index skew empirically decays roughly like T^(-1/2) or steeper at the short end,
    while Heston's skew flattens to a constant as T -> 0; this plot makes that gap visible."""
    maturities = np.array([e.maturity for e in snapshot.expiries])
    market_skew = np.array(
        [abs(f.slice.dw_dk(0.0)) / (2.0 * np.sqrt(max(f.slice.total_variance(0.0), 1e-12) * f.slice.maturity)) for f in fits]
    )
    atm = np.array([e.atm_vol() for e in snapshot.expiries])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(maturities, 100 * atm, "o-", color=MARKET, ms=4, label="market ATM vol")
    if calibration is not None:
        heston_atm = [
            float(_heston_smile(calibration.params, e.forward, e.discount, e.maturity, np.array([e.forward]))[0])
            for e in snapshot.expiries
        ]
        ax1.plot(maturities, 100 * np.array(heston_atm), "s--", color=HESTON_COLOR, ms=4, label="Heston ATM vol")
    ax1.set_xlabel("maturity (years)")
    ax1.set_ylabel("ATM implied vol (%)")
    ax1.set_title("ATM volatility term structure")
    ax1.legend()

    ax2.loglog(maturities, market_skew, "o", color=SVI_COLOR, ms=5, label="market (SVI) ATM skew")
    slope, intercept = np.polyfit(np.log(maturities), np.log(market_skew), 1)
    grid = np.geomspace(maturities.min(), maturities.max(), 50)
    ax2.loglog(grid, np.exp(intercept) * grid**slope, color=SVI_COLOR, lw=1, alpha=0.7, label=f"power law  T^{slope:.2f}")
    if calibration is not None:
        h = 0.01
        heston_skew = []
        for e in snapshot.expiries:
            strikes = e.forward * np.exp(np.array([-h, h]))
            iv = _heston_smile(calibration.params, e.forward, e.discount, e.maturity, strikes)
            heston_skew.append(abs(iv[1] - iv[0]) / (2 * h))
        ax2.loglog(maturities, heston_skew, "s--", color=HESTON_COLOR, ms=4, label="Heston ATM skew")
    ax2.set_xlabel("maturity (years)")
    ax2.set_ylabel("|d sigma / d ln K| at the money")
    ax2.set_title("Skew term structure")
    _plain_log_ticks(ax2.xaxis, maturities.min(), maturities.max(), _LOG_CANDIDATES)
    y_all = np.concatenate([market_skew, heston_skew]) if calibration is not None else market_skew
    _plain_log_ticks(ax2.yaxis, float(np.min(y_all)), float(np.max(y_all)), _LOG_CANDIDATES)
    ax2.legend()
    fig.tight_layout()
    return fig


def local_vol_heatmap(surface: VolSurface, snapshot: MarketSnapshot) -> Figure:
    maturities = np.linspace(surface.maturities[0], surface.maturities[-1], 60)
    z = np.linspace(*quoted_standardised_range(snapshot), 70)
    lv = np.array([surface.local_vol(z * np.sqrt(t), t) for t in maturities])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mesh = ax.pcolormesh(
        z,
        maturities,
        100 * lv,
        shading="auto",
        cmap="magma",
        vmin=np.nanpercentile(100 * lv, 2),
        vmax=np.nanpercentile(100 * lv, 98),
    )
    fig.colorbar(mesh, ax=ax, label="local vol (%)")
    ax.set_xlabel("standardised moneyness  ln(K/F)/sqrt(T)")
    ax.set_ylabel("maturity (years)")
    ax.set_title("Dupire local volatility (from the SVI surface)")
    ax.grid(False)
    fig.tight_layout()
    return fig


def densities(fits: list[SVIFit], max_curves: int = 6) -> Figure:
    chosen = fits if len(fits) <= max_curves else [fits[i] for i in np.linspace(0, len(fits) - 1, max_curves).astype(int)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    cmap = plt.get_cmap("viridis")
    for i, fit in enumerate(chosen):
        sl = fit.slice
        width = 5.0 * np.sqrt(max(sl.total_variance(0.0), 1e-4))
        k = np.linspace(-width, 0.6 * width, 400)
        dens = sl.risk_neutral_density(k)
        ax.plot(np.exp(k) - 1.0, dens / np.exp(k), color=cmap(i / max(len(chosen) - 1, 1)), lw=1.5, label=f"T={sl.maturity:.2f}y")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("terminal return  S_T / F - 1")
    ax.set_ylabel("risk-neutral density")
    ax.set_title("Market-implied risk-neutral densities (Breeden-Litzenberger via SVI)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def calibration_residuals(calibration: CalibrationResult) -> Figure:
    r = calibration.residuals.dropna(subset=["error"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    limit = max(np.nanpercentile(np.abs(100 * r["error"]), 97), 0.25)
    sc = ax1.scatter(r["k"], r["maturity"], c=100 * r["error"], cmap="RdBu_r", vmin=-limit, vmax=limit, s=16, edgecolors="none")
    fig.colorbar(sc, ax=ax1, label="Heston - market (vol pts)")
    ax1.set_xlabel("log-moneyness  ln(K/F)")
    ax1.set_ylabel("maturity (years)")
    ax1.set_title("Where Heston misprices the surface")

    by = calibration.error_by_expiry()
    ax2.bar(np.arange(len(by)), by["rmse_vol_pts"], color=HESTON_COLOR, alpha=0.85)
    ax2.set_xticks(np.arange(len(by)))
    ax2.set_xticklabels([f"{t:.2f}" for t in by["maturity"]], rotation=45)
    ax2.set_xlabel("maturity (years)")
    ax2.set_ylabel("RMSE (vol pts)")
    ax2.set_title(f"Fit error by expiry  (total {100 * calibration.rmse_vol:.2f} vol pts)")
    fig.tight_layout()
    return fig


def hedging(bs_world: HedgingExperiment | None, heston_world: HedgingExperiment | None) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    if bs_world is not None:
        s = bs_world.summary()
        ax1.loglog(s["rebalances"], s["std_pct_premium"], "o-", color="#52606d", label="Black-Scholes world: simulated")
        premium = s["premium"].iloc[0]
        theory = [100 * bs_world.notes[f"derman_kamal_std_{n}"] / premium for n in s["rebalances"]]
        ax1.loglog(s["rebalances"], theory, ":", color="#52606d", label="Derman-Kamal  ~ N^(-1/2)")
    if heston_world is not None:
        s = heston_world.summary()
        for name, group in s[s["rebalances"] > 0].groupby("strategy", sort=False):
            ax1.loglog(
                group["rebalances"],
                group["std_pct_premium"],
                "s-",
                color=STRATEGY_COLORS.get(name, INK),
                label=f"Heston world: {name}",
            )
    ys = [line.get_ydata() for line in ax1.get_lines()]
    xs = [line.get_xdata() for line in ax1.get_lines()]
    if ys:
        x_ticks: list[float] = []
        for value in np.unique(np.concatenate(xs)):
            if not x_ticks or value / x_ticks[-1] >= 1.3:
                x_ticks.append(float(value))
        ax1.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax1.xaxis.set_minor_locator(NullLocator())
        _plain_log_ticks(ax1.yaxis, float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys))), _LOG_CANDIDATES)
    ax1.set_xlabel("rebalances over the option's life")
    ax1.set_ylabel("hedging error, std (% of premium)")
    ax1.set_title("Hedging error vs rebalancing frequency")
    ax1.legend(fontsize=7.5)

    if heston_world is not None:
        finest = max(r.rebalances for r in heston_world.runs)
        bins = None
        for run in heston_world.runs:
            if run.rebalances != finest:
                continue
            pnl = 100 * run.pnl / run.premium
            if bins is None:
                lo, hi = np.percentile(pnl, [0.5, 99.5])
                bins = np.linspace(lo, hi, 70)
            ax2.hist(
                pnl,
                bins=bins,
                histtype="step",
                lw=1.5,
                density=True,
                color=STRATEGY_COLORS.get(run.strategy, INK),
                label=run.strategy,
            )
        ax2.set_xlabel("hedged P&L at expiry (% of premium)")
        ax2.set_ylabel("density")
        ax2.set_title(f"P&L distribution, {finest} rebalances, Heston world")
        ax2.legend(fontsize=7.5)
    fig.tight_layout()
    return fig
