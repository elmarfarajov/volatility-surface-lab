"""Day 2 charts: lattices, finite differences and Monte Carlo, and the references they converge to."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ..analytic import greeks
from ..numerics.lattice import LATTICES, Contract, bbsr, price
from ..numerics.montecarlo import european, longstaff_schwartz
from ..numerics.pde import solve
from ..numerics.references import LONGSTAFF_SCHWARTZ_TABLE
from .style import PALETTE, SERIES, caption, new_figure, tidy, title_block

CALL = Contract(100.0, 105.0, 1.0, 0.05, 0.02, 0.25, True)
PUT_36 = Contract(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, False)
COLOURS = dict(zip((*LATTICES, "BBSR"), (*SERIES, PALETTE["accent"]), strict=False))


def lattice_convergence() -> Figure:
    fig = new_figure(14.0, 5.8)
    title_block(
        fig,
        "How fast each lattice converges, European and American",
        "Absolute error against the exact price as steps are added. CRR, Jarrow-Rudd and Tian oscillate with the position "
        "of the strike between nodes; Leisen-Reimer is smooth and second order for a European option; BBSR (smoothed, "
        "Richardson-extrapolated CRR) is the fastest for an American one.",
    )
    european_steps = np.unique(np.geomspace(20, 2000, 60).astype(int))
    american_steps = np.unique(np.geomspace(20, 1500, 28).astype(int))
    targets = (
        (CALL, "european", CALL.european(), european_steps),
        (PUT_36, "american", LONGSTAFF_SCHWARTZ_TABLE[0].american, american_steps),
    )
    for index, (contract, exercise, reference, steps) in enumerate(targets):
        ax = fig.add_axes((0.06 + index * 0.48, 0.15, 0.4, 0.62))
        for lattice in LATTICES:
            errors = [abs(price(contract, int(n), lattice, exercise).price - reference) for n in steps]
            ax.loglog(steps, np.maximum(errors, 1e-12), color=COLOURS[lattice], lw=1.3, label=lattice)
        smooth = [abs(bbsr(contract, int(n) // 2, "crr", exercise) - reference) for n in steps]
        ax.loglog(steps, np.maximum(smooth, 1e-12), color=COLOURS["BBSR"], lw=2.0, label="BBSR (CRR)")
        label = "European call, S 100, K 105" if exercise == "european" else "American put, S 36, K 40 (Longstaff-Schwartz case)"
        tidy(ax, title=label, xlabel="time steps", ylabel="absolute error")
        if index == 0:
            ax.legend(fontsize=8, ncol=2)
    caption(fig, "American reference: Andersen, Lake and Offengenden (2016), QuantLib's high-precision scheme, 4.486674.")
    return fig


def longstaff_schwartz_table() -> Figure:
    fig = new_figure(14.0, 6.2)
    title_block(
        fig,
        "Longstaff and Schwartz's 'American' prices are Bermudan prices",
        "Their Table 1 (2001) finite-difference column, the usual reference for "
        "least-squares Monte Carlo, against the American put "
        "(Andersen-Lake-Offengenden) and a put exercisable fifty times a year. Strike 40, rate 6%, twenty cases.",
    )
    rows = LONGSTAFF_SCHWARTZ_TABLE
    x = np.arange(len(rows))
    ax = fig.add_axes((0.06, 0.2, 0.9, 0.5))
    ax.axhline(0.0, color=PALETTE["muted"], lw=0.8)
    ax.bar(
        x - 0.2,
        [r.longstaff_schwartz - r.american for r in rows],
        width=0.38,
        color=PALETTE["red"],
        label="table minus American price",
    )
    ax.bar(
        x + 0.2,
        [r.longstaff_schwartz - r.bermudan_50 for r in rows],
        width=0.38,
        color=PALETTE["blue"],
        label="table minus Bermudan (50 a year)",
    )
    ax.axhspan(-0.0005, 0.0005, color=PALETTE["band"], zorder=0)
    labels = [f"S {r.spot:.0f}\nsigma {r.vol:.1f}\nT {r.maturity:.0f}" for r in rows]
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    tidy(ax, ylabel="difference, price units")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    bermudan = sum(abs(r.longstaff_schwartz - r.bermudan_50) < 5e-4 for r in rows)
    ax.text(
        1.0,
        1.04,
        f"{bermudan} of 20 within 0.0005 of the Bermudan price",
        transform=ax.transAxes,
        ha="right",
        fontsize=9.5,
        color=PALETTE["blue"],
        fontweight="bold",
    )
    caption(
        fig,
        "Band: +-0.0005, the table's rounding. The Bermudan values are this package's "
        "Crank-Nicolson engine, 1,600 x 1,600; they agree with QuantLib's to 1e-4.",
    )
    return fig


def rannacher() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Crank-Nicolson rings at the strike; four implicit half-steps stop it",
        "Gamma of a three-month at-the-money call from the finite-difference grid (800 space steps, 25 time steps) against the "
        "exact gamma. The payoff's kink excites a mode Crank-Nicolson barely "
        "damps; Rannacher's start damps it and keeps second order.",
    )
    contract = Contract(100.0, 100.0, 0.25, 0.05, 0.0, 0.2, True)
    for index, (label, steps) in enumerate(
        (("Crank-Nicolson", 0), ("Crank-Nicolson after 4 implicit half-steps (Rannacher)", 4))
    ):
        result = solve(contract, 800, 25, 0.5, steps)
        near = (result.grid > 80) & (result.grid < 120)
        gamma = np.gradient(np.gradient(result.values, result.grid), result.grid)[near]
        exact = greeks(result.grid[near], 100.0, 0.25, 0.05, 0.0, 0.2, True).gamma
        ax = fig.add_axes((0.06 + index * 0.48, 0.15, 0.4, 0.62))
        ax.plot(result.grid[near], exact, color=PALETTE["muted"], lw=3.0, alpha=0.6, label="exact")
        ax.plot(result.grid[near], gamma, color=SERIES[index], lw=1.2, label="finite differences")
        tidy(ax, title=label, xlabel="spot", ylabel="gamma")
        ax.legend(loc="upper right")
        ax.text(0.02, 0.92, f"worst error {np.max(np.abs(gamma - exact)):.1e}", transform=ax.transAxes, fontsize=9)
    caption(
        fig,
        "Rannacher (1984). The price is affected too: plain Crank-Nicolson is "
        "0.03 off at the money on this grid, Rannacher 0.0007.",
    )
    return fig


def exercise_boundary() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Where an American put should be exercised",
        "The early-exercise boundary from the Crank-Nicolson engine: below it "
        "the put is worth its intrinsic value and is exercised. "
        "Strike 40, rate 6%, one year. Near expiry the boundary approaches the strike only like K(1 - sigma sqrt(tau |ln tau|)).",
    )
    ax = fig.add_axes((0.06, 0.15, 0.5, 0.62))
    for colour, vol in zip(SERIES, (0.1, 0.2, 0.3, 0.4), strict=False):
        result = solve(Contract(36.0, 40.0, 1.0, 0.06, 0.0, vol, False), 800, 800, exercise="american")
        assert result.boundary is not None and result.boundary_times is not None
        ax.plot(result.boundary_times, result.boundary, color=colour, lw=1.6, label=f"volatility {vol:.0%}")
        tau = 1.0 - result.boundary_times
        tau = tau[(tau > 0) & (tau < 0.1)]  # the asymptote holds near expiry only
        ax.plot(1.0 - tau, 40.0 * (1 - vol * np.sqrt(tau * np.abs(np.log(tau)))), color=colour, lw=0.8, ls=":")
    ax.axhline(40.0, color=PALETTE["muted"], lw=0.8)
    ax.set_ylim(20, 41)
    tidy(
        ax, title="The boundary through the life of the option", xlabel="years from today", ylabel="spot below which to exercise"
    )
    ax.legend(loc="lower right")
    right = fig.add_axes((0.64, 0.15, 0.32, 0.62))
    result = solve(PUT_36, 800, 800, exercise="american")
    european_values = greeks(result.grid, 40.0, 1.0, 0.06, 0.0, 0.2, False).price
    window = (result.grid > 20) & (result.grid < 60)
    right.plot(result.grid[window], result.values[window], color=SERIES[1], lw=1.8, label="American")
    right.plot(result.grid[window], european_values[window], color=SERIES[0], lw=1.2, label="European")
    right.plot(
        result.grid[window],
        np.maximum(40.0 - result.grid[window], 0.0),
        color=PALETTE["muted"],
        lw=1.0,
        ls="--",
        label="intrinsic",
    )
    tidy(right, title="Today's value, one year to expiry", xlabel="spot", ylabel="value")
    right.legend()
    caption(
        fig,
        "Dotted, over the last tenth of the option's life: the near-expiry "
        "asymptote (Barles, Burdeau, Romano and Samsoen, 1995).",
    )
    return fig


def monte_carlo_convergence() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Monte Carlo error against the number of samples",
        "Standard error of a one-year European call by four kinds of sampling. "
        "Pseudo-random error falls like N^(-1/2); scrambled "
        "Sobol' points, randomised sixteen times so the error can be measured, fall close to N^(-1).",
    )
    ax = fig.add_axes((0.06, 0.15, 0.5, 0.62))
    sizes = 2 ** np.arange(10, 21)
    for colour, sampling in zip(SERIES, ("pseudo", "antithetic", "control", "sobol"), strict=False):
        errors = [european(CALL, int(n), sampling, seed=11).std_error for n in sizes]  # type: ignore[arg-type]
        ax.loglog(sizes, errors, "o-", color=colour, ms=4, lw=1.4, label=sampling)
    ax.loglog(sizes, 1.0 / np.sqrt(sizes), color=PALETTE["muted"], lw=0.8, ls=":")
    ax.loglog(sizes, 30.0 / sizes, color=PALETTE["muted"], lw=0.8, ls="--")
    ax.text(sizes[-1], 1.0 / np.sqrt(sizes[-1]) * 0.6, "N^-1/2", fontsize=8, color=PALETTE["muted"], ha="right")
    ax.text(sizes[-1], 30.0 / sizes[-1] * 1.3, "N^-1", fontsize=8, color=PALETTE["muted"], ha="right")
    tidy(ax, title="Standard error", xlabel="samples", ylabel="standard error")
    ax.legend()
    right = fig.add_axes((0.64, 0.15, 0.32, 0.62))
    n = 2**18
    names = ["pseudo", "antithetic", "control", "sobol"]
    at_size = np.array([european(CALL, n, s, seed=11).std_error for s in names])  # type: ignore[arg-type]
    gain = (at_size[0] / at_size) ** 2
    right.barh(np.arange(4)[::-1], gain, color=SERIES[:4])
    right.set_xscale("log")
    right.set_yticks(np.arange(4)[::-1], names)
    for y, value in zip(np.arange(4)[::-1], gain, strict=False):
        right.text(value * 1.1, y, f"{value:,.1f}x" if value < 10 else f"{value:,.0f}x", va="center", fontsize=9)
    tidy(right, title="Samples saved at 2^18, against pseudo-random", xlabel="variance reduction factor")
    caption(
        fig,
        "Owen-scrambled Sobol' points (SciPy), mapped to normals by the inverse "
        "normal; the control variate is the discounted terminal spot.",
    )
    return fig


def early_exercise_bounds() -> Figure:
    fig = new_figure(14.0, 5.6)
    title_block(
        fig,
        "Least-squares Monte Carlo, bracketed from both sides",
        "The Bermudan put (S 36, K 40, fifty dates). Lower bound: the fitted "
        "exercise rule followed on independent paths. Upper bound: "
        "duality, with a martingale built from the fitted value function. The truth must lie between them.",
    )
    reference = LONGSTAFF_SCHWARTZ_TABLE[0].bermudan_50
    dates = [k / 50 for k in range(1, 51)]
    configurations = (("powers of S/K", False), ("powers of S/K\nand the European price", True))
    ax = fig.add_axes((0.14, 0.15, 0.46, 0.62))
    gaps = []
    for index, (_label, basis) in enumerate(configurations):
        result = longstaff_schwartz(
            PUT_36, dates, 100_000, 100_000, dual_paths=4_000, inner_paths=200, seed=1, european_basis=basis
        )
        gaps.append(result.gap)
        y = index
        ax.errorbar(
            result.lower.price,
            y,
            xerr=1.96 * result.lower.std_error,
            fmt="o",
            color=SERIES[0],
            ms=7,
            capsize=4,
            label="lower bound" if index == 0 else None,
        )
        ax.errorbar(
            result.upper.price,
            y,
            xerr=1.96 * result.upper.std_error,
            fmt="s",
            color=SERIES[2],
            ms=7,
            capsize=4,
            label="upper bound" if index == 0 else None,
        )
        ax.plot([result.lower.price, result.upper.price], [y, y], color=PALETTE["grid"], lw=6, zorder=0)
        ax.plot(result.in_sample, y + 0.18, "x", color=PALETTE["muted"], ms=7, label="in-sample estimate" if index == 0 else None)
        ax.text(result.upper.price + 0.004, y, f"gap {result.gap:.4f}", va="center", fontsize=9)
    ax.axvline(reference, color=PALETTE["accent"], lw=1.4)
    ax.text(reference, 1.45, f"Bermudan value {reference:.4f}", color=PALETTE["accent"], fontsize=9, ha="center")
    ax.set_yticks([0, 1], [c[0] for c in configurations])
    ax.set_ylim(-0.5, 1.6)
    tidy(ax, title="Regression on", xlabel="price")
    ax.legend(loc="lower right", fontsize=8)
    note = fig.add_axes((0.64, 0.15, 0.32, 0.62))
    note.axis("off")
    text = (
        "Fitted and priced on the same paths, the estimate\n"
        "borrows foresight: it can sit above the truth.\n\n"
        "Followed on independent paths, the rule is a\n"
        "feasible policy: it cannot be worth more than the\n"
        "optimum, so it is a lower bound.\n\n"
        "For any martingale M starting at zero, the option is\n"
        "worth at most E[max_k (Z_k - M_k)] (Rogers, 2002;\n"
        "Haugh and Kogan, 2004). The better the fitted value\n"
        "function, the closer the martingale to the optimal one,\n"
        "and the tighter the bound: adding the European price\n"
        f"to the regression closes the gap from {gaps[0]:.2f} to {gaps[1]:.3f}."
    )
    note.text(0.0, 1.0, text, va="top", fontsize=9, color=PALETTE["ink"], family="monospace")
    caption(
        fig,
        "Inner simulations one exercise date ahead estimate the martingale's "
        "increments; their noise can only raise the upper bound.",
    )
    return fig
