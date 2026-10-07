"""Day 1 charts: the analytic engine, its Greeks, and the references it is held to."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ..analytic import Carry, greeks, price
from ..analytic.published import HAUG_2007
from ..analytic.reference import DEFINITIONS, reference_greeks
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

GREEK_TITLES = {
    "delta": "Delta  dV/dS",
    "gamma": "Gamma  d2V/dS2",
    "vega": "Vega  dV/dsigma",
    "theta": "Theta  dV/dt",
    "rho": "Rho  dV/dr",
    "psi": "Psi  dV/dq",
    "vanna": "Vanna  d2V/dS dsigma",
    "volga": "Volga  d2V/dsigma2",
    "charm": "Charm  d delta/dt",
    "veta": "Veta  d vega/dt",
    "speed": "Speed  d gamma/dS",
    "zomma": "Zomma  d gamma/dsigma",
    "color": "Color  d gamma/dt",
    "ultima": "Ultima  d volga/dsigma",
    "dual_delta": "Dual delta  dV/dK",
    "dual_gamma": "Dual gamma  d2V/dK2",
}


def greek_atlas() -> Figure:
    fig = new_figure(14.0, 12.0)
    title_block(
        fig,
        "Sixteen Greeks of a European call, and how they change as expiry nears",
        "Strike 100, rate 4%, dividend yield 1%, volatility 25%. Each panel: the Greek against spot, three months to expiry, "
        "one month and one week. Time Greeks are per year of calendar time passing.",
    )
    spots = np.linspace(60.0, 140.0, 400)
    maturities = {"3 months": 0.25, "1 month": 1 / 12, "1 week": 1 / 52}
    for index, (name, label) in enumerate(GREEK_TITLES.items()):
        ax = fig.add_axes((0.05 + (index % 4) * 0.24, 0.72 - (index // 4) * 0.215, 0.20, 0.16))
        for colour, (tenor, maturity) in zip(SERIES, maturities.items(), strict=False):
            values = getattr(greeks(spots, 100.0, maturity, 0.04, 0.01, 0.25, True), name)
            ax.plot(spots, values, color=colour, lw=1.5, label=tenor)
        ax.axvline(100.0, color=PALETTE["grid"], lw=1.0, zorder=0)
        tidy(ax, title=label)
        ax.tick_params(labelsize=7.5)
        if index == 0:
            ax.legend(fontsize=7.5, loc="lower right")
    caption(fig, "volsurf.analytic.greeks; every curve checked against mpmath differentiation at 50 digits.")
    return fig


def reference_agreement(contracts: int = 30, seed: int = 1) -> Figure:
    fig = new_figure(14.0, 6.2)
    title_block(
        fig,
        "Every Greek against an independent reference",
        "Left: worst relative error of each analytic Greek against the price differentiated in arbitrary precision (mpmath, 50 "
        "digits), over random contracts. Right: Haug's (2007) worked examples, as printed.",
    )
    rng = np.random.default_rng(seed)
    worst = {name: 0.0 for name in DEFINITIONS}
    for _ in range(contracts):
        contract = (
            rng.uniform(50, 150),
            rng.uniform(50, 150),
            rng.uniform(0.05, 3),
            rng.uniform(-0.02, 0.1),
            rng.uniform(0, 0.06),
            rng.uniform(0.08, 0.8),
            bool(rng.random() < 0.5),
        )
        analytic = greeks(*contract).as_dict()
        reference = reference_greeks(*contract)
        for name in DEFINITIONS:
            scale = max(abs(reference[name]), 1e-12)
            worst[name] = max(worst[name], abs(float(analytic[name]) - reference[name]) / scale)
    ax = fig.add_axes((0.09, 0.15, 0.43, 0.64))
    names = list(worst)
    y = np.arange(len(names))[::-1]
    errors = np.maximum([worst[n] for n in names], 1e-17)
    ax.barh(y, errors, color=PALETTE["blue"], height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 1e-8)
    ax.axvline(2.2e-16, color=PALETTE["muted"], lw=0.8, ls=":")
    ax.text(2.4e-16, len(names) - 0.3, "machine epsilon", fontsize=7.5, color=PALETTE["muted"])
    ax.set_yticks(y, [n.replace("_", " ") for n in names], fontsize=8.5)
    tidy(ax, title=f"Worst relative error over {contracts} random contracts", xlabel="relative error, log scale")

    table = fig.add_axes((0.58, 0.15, 0.40, 0.66))
    table.axis("off")
    table.set_title("Haug (2007), four decimals as printed", loc="left")
    rows = []
    for example in HAUG_2007:
        value = float(
            getattr(
                greeks(
                    example.spot,
                    example.strike,
                    example.maturity,
                    example.rate,
                    example.yield_,
                    example.vol,
                    example.is_call,
                ),
                example.quantity,
            )
        )
        rows.append((example.title, f"{example.published:.4f}", f"{value:.4f}"))
    for i, (title, published, computed) in enumerate(rows):
        top = 0.93 - i * 0.1
        table.text(0.0, top, title, fontsize=8.5, color=PALETTE["ink"], transform=table.transAxes)
        table.text(0.74, top, published, fontsize=8.5, color=PALETTE["muted"], ha="right", transform=table.transAxes)
        table.text(
            0.92, top, computed, fontsize=8.5, color=PALETTE["green"], ha="right", fontweight="bold", transform=table.transAxes
        )
    table.text(0.74, 1.0, "printed", fontsize=8, color=PALETTE["muted"], ha="right", transform=table.transAxes)
    table.text(0.92, 1.0, "engine", fontsize=8, color=PALETTE["green"], ha="right", transform=table.transAxes)
    caption(
        fig,
        "The mpmath reference caught veta and color with their signs turned: as usually printed they are derivatives in time to "
        "expiry, not in time passing. QuantLib agrees on 500 more random contracts to 1e-10 (tests/analytic).",
    )
    return fig


def tail_precision() -> Figure:
    import mpmath as mp

    mp.mp.dps = 60
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Prices far in the wings, against 60-digit arithmetic",
        "Out-of-the-money calls and puts at z = ln(K/F) / (sigma sqrt T) standard "
        "deviations from the forward: the price falls through "
        "270 orders of magnitude, and the relative error stays below 5e-11 until the double-precision range ends.",
    )
    zs = np.linspace(0.25, 37.0, 80)
    forward, maturity, vol = 100.0, 1.0, 0.2
    exact_prices: dict[str, list[float]] = {"call": [], "put": []}
    errors: dict[str, list[float]] = {"call": [], "put": []}
    for z in zs:
        for side, sign in (("call", 1.0), ("put", -1.0)):
            strike = forward * np.exp(sign * z * vol)
            F, K, T, v = (mp.mpf(a) for a in (forward, strike, maturity, vol))
            s = v * mp.sqrt(T)
            d1 = mp.log(F / K) / s + s / 2
            d2 = d1 - s
            exact = F * mp.ncdf(d1) - K * mp.ncdf(d2) if sign > 0 else K * mp.ncdf(-d2) - F * mp.ncdf(-d1)
            ours = float(price(forward, strike, maturity, 0.0, 0.0, vol, sign > 0))
            exact_prices[side].append(float(exact))
            errors[side].append(abs(ours - float(exact)) / float(exact) if exact > 0 else np.nan)
    left = fig.add_axes((0.06, 0.16, 0.40, 0.62))
    for colour, side in zip(SERIES, ("call", "put"), strict=False):
        left.semilogy(zs, exact_prices[side], color=colour, lw=1.8, label=f"out-of-the-money {side}")
    tidy(left, title="The exact price", xlabel="standard deviations out of the money, z", ylabel="price (forward 100)")
    left.legend()
    right = fig.add_axes((0.56, 0.16, 0.40, 0.62))
    for colour, side in zip(SERIES, ("call", "put"), strict=False):
        right.semilogy(zs, np.maximum(errors[side], 1e-17), "o", color=colour, ms=3.5, label=side)
    right.axhline(5e-11, color=PALETTE["accent"], lw=1.0, ls="--")
    right.text(1.0, 7e-11, "5e-11", color=PALETTE["accent"], fontsize=8)
    tidy(right, title="Relative error of the engine", xlabel="standard deviations out of the money, z", ylabel="relative error")
    caption(
        fig,
        "Beyond z = 37 the price is below the smallest double (about 1e-308) and both are zero. "
        "SciPy's ndtr keeps the normal tail accurate; the call and put are each computed directly, never by parity.",
    )
    return fig


def pde_residual() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Every price solves the Black-Scholes equation",
        "dV/dt + (r - q) S dV/dS + sigma^2 S^2 / 2 d2V/dS2 - r V, from the "
        "analytic Greeks, relative to the size of its largest term. "
        "Property-tested on random contracts; here on a grid.",
    )
    moneyness = np.linspace(-1.2, 1.2, 241)
    maturities = np.geomspace(1 / 365, 10.0, 200)
    m, t = np.meshgrid(moneyness, maturities)
    for index, (label, call, rate, carry_yield, vol) in enumerate(
        (
            ("Call, r = 4%, q = 1%, sigma = 25%", True, 0.04, 0.01, 0.25),
            ("Put, r = -1%, q = 3%, sigma = 80%", False, -0.01, 0.03, 0.8),
        )
    ):
        g = greeks(100.0, 100.0 * np.exp(m), t, rate, carry_yield, vol, call)
        terms = np.stack([g.theta, (rate - carry_yield) * 100.0 * g.delta, 0.5 * vol**2 * 100.0**2 * g.gamma, -rate * g.price])
        residual = np.abs(terms.sum(axis=0)) / np.maximum(np.abs(terms).max(axis=0), 1e-300)
        ax = fig.add_axes((0.06 + index * 0.47, 0.16, 0.36, 0.62))
        image = ax.pcolormesh(m, t, np.maximum(residual, 1e-18), norm=LogNorm(1e-18, 1e-12), cmap="Blues", shading="auto")
        ax.set_yscale("log")
        tidy(ax, title=label, xlabel="ln(K / S)", ylabel="years to expiry")
        ax.grid(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
        bar.set_label("relative residual", fontsize=8)
        ax.text(
            0.02,
            0.03,
            f"largest: {residual.max():.1e}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    caption(
        fig,
        "Theta is the derivative in calendar time passing; a residual at machine "
        "precision means each Greek is the derivative it claims to be.",
    )
    return fig


def carry_conventions() -> Figure:
    fig = new_figure(14.0, 5.4)
    title_block(
        fig,
        "One model, five underlyings: the cost of carry decides",
        "The same call - strike 100, one year, volatility 25%, domestic rate 4% - "
        "on a stock, an index paying 2%, a future, a currency "
        "whose foreign rate is 6%, and a margined future.",
    )
    conventions = {
        "stock, no dividend (Black-Scholes)": Carry.stock(0.04),
        "index, 2% yield (Merton)": Carry.stock(0.04, 0.02),
        "future (Black-76)": Carry.future(0.04),
        "currency, 6% foreign rate (Garman-Kohlhagen)": Carry.currency(0.04, 0.06),
        "margined future (Asay)": Carry.margined_future(),
    }
    spots = np.linspace(60.0, 140.0, 300)
    colours = (*SERIES, PALETTE["accent"])
    for index, (quantity, label) in enumerate((("price", "Price"), ("delta", "Delta"), ("gamma", "Gamma"))):
        ax = fig.add_axes((0.05 + index * 0.325, 0.17, 0.27, 0.6))
        for colour, (name, carry) in zip(colours, conventions.items(), strict=False):
            values = getattr(greeks(spots, 100.0, 1.0, carry.rate, carry.yield_, 0.25, True), quantity)
            ax.plot(spots, values, color=colour, lw=1.6, label=name)
        tidy(ax, title=label, xlabel="spot (or futures price)")
        if index == 0:
            ax.legend(fontsize=7.5, loc="upper left")
    caption(
        fig,
        "The forward is S exp((r - q) T): a dividend or a foreign rate lowers "
        "it, a future's carry is zero, and the margined future "
        "is neither discounted nor carried. volsurf.analytic.Carry builds each from its rates.",
    )
    return fig


def expiry_limits() -> Figure:
    fig = new_figure(14.0, 5.4)
    title_block(
        fig,
        "At expiry an option is its intrinsic value: the Greeks reach their limits instead of dividing by zero",
        "A put struck at 100 on a spot of 95 (in the money) and 105 (out), as time to expiry falls to zero. The lab's original "
        "Greeks returned infinities and NaN at T = 0; the limits are drawn as points.",
    )
    maturities = np.geomspace(1e-6, 0.5, 300)
    for index, (name, label) in enumerate((("price", "Price"), ("delta", "Delta"), ("theta", "Theta, per year"))):
        ax = fig.add_axes((0.05 + index * 0.325, 0.17, 0.27, 0.6))
        for colour, spot in zip(SERIES, (95.0, 105.0), strict=False):
            curve = getattr(greeks(spot, 100.0, maturities, 0.04, 0.0, 0.3, False), name)
            limit = float(getattr(greeks(spot, 100.0, 0.0, 0.04, 0.0, 0.3, False), name))
            ax.semilogx(maturities, curve, color=colour, lw=1.6, label=f"spot {spot:.0f}")
            ax.plot([1e-6], [limit], "o", color=colour, ms=7, mfc="white", mew=1.8)
        tidy(ax, title=label, xlabel="years to expiry (log)")
        if index == 0:
            ax.legend(loc="upper left")
    caption(fig, "Theta at expiry is the decay of the discounted intrinsic value: for the in-the-money put, r K = 4.")
    return fig
