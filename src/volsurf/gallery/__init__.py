"""The chart gallery: every figure the documentation shows, rebuilt from source with ``volsurf gallery``.

Each chart is a :class:`Item` - the file it writes, what it shows, the function
that draws it, and the day of the build it belongs to - so the gallery page
(``docs/GALLERY.md``) and the images cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from . import analytic, market, numerics
from .style import save

DAYS: dict[int, str] = {1: "The analytic foundation", 2: "Numerical engines", 3: "Implied volatility and the market"}


@dataclass(frozen=True)
class Item:
    filename: str
    title: str
    description: str
    build: Callable[[], Figure]
    day: int
    requires: str | None = None  # "snapshot": drawn from a local option-chain snapshot (``volsurf chains fetch``)

    def available(self) -> bool:
        return self.requires is None or market.snapshot_available()


def items() -> tuple[Item, ...]:
    return (
        Item(
            "greek-atlas.png",
            "Sixteen Greeks of a European call",
            "Every Greek against spot, three months, one month and one week from expiry.",
            analytic.greek_atlas,
            1,
        ),
        Item(
            "greek-references.png",
            "Every Greek against an independent reference",
            "Arbitrary-precision differentiation and Haug's worked examples.",
            analytic.reference_agreement,
            1,
        ),
        Item(
            "tail-precision.png",
            "Prices far in the wings",
            "The engine against 60-digit arithmetic, through 270 orders of magnitude.",
            analytic.tail_precision,
            1,
        ),
        Item(
            "pde-residual.png",
            "Every price solves the Black-Scholes equation",
            "The PDE residual from the analytic Greeks, over moneyness and maturity.",
            analytic.pde_residual,
            1,
        ),
        Item(
            "carry-conventions.png",
            "One model, five underlyings",
            "A stock, an index, a future, a currency and a margined future, through the cost of carry.",
            analytic.carry_conventions,
            1,
        ),
        Item(
            "expiry-limits.png",
            "The Greeks at expiry",
            "Price, delta and theta reaching their limits as time to expiry falls to zero.",
            analytic.expiry_limits,
            1,
        ),
        Item(
            "lattice-convergence.png",
            "How fast each lattice converges",
            "Five lattices and BBSR against the exact price, European and American.",
            numerics.lattice_convergence,
            2,
        ),
        Item(
            "longstaff-schwartz-table.png",
            "Longstaff and Schwartz's 'American' prices are Bermudan prices",
            "Their Table 1 against the American and the fifty-date Bermudan put.",
            numerics.longstaff_schwartz_table,
            2,
        ),
        Item(
            "rannacher.png",
            "Crank-Nicolson rings at the strike",
            "Gamma from the finite-difference grid, with and without Rannacher's start.",
            numerics.rannacher,
            2,
        ),
        Item(
            "exercise-boundary.png",
            "Where an American put should be exercised",
            "The early-exercise boundary through the option's life, and today's value.",
            numerics.exercise_boundary,
            2,
        ),
        Item(
            "monte-carlo-convergence.png",
            "Monte Carlo error against the number of samples",
            "Pseudo-random, antithetic, control-variate and scrambled Sobol' sampling.",
            numerics.monte_carlo_convergence,
            2,
        ),
        Item(
            "early-exercise-bounds.png",
            "Least-squares Monte Carlo, bracketed from both sides",
            "The out-of-sample lower bound and the dual upper bound around the Bermudan value.",
            numerics.early_exercise_bounds,
            2,
        ),
        Item(
            "black-cancellation.png",
            "The textbook Black formula cancels in the wings",
            "Out-of-the-money prices against 60-digit arithmetic: the literal formula and the Mills-ratio form.",
            market.black_cancellation,
            3,
        ),
        Item(
            "iv-accuracy.png",
            "Implied volatility to the last digit",
            "The v1.0 Newton inversion and the Jaeckel-style one, against exact prices from 1e-300 to the forward.",
            market.iv_accuracy,
            3,
        ),
        Item(
            "inversion-iterations.png",
            "How many steps the inversion takes",
            "Householder iterations over strike and volatility, and their distribution over random quotes.",
            market.inversion_iterations,
            3,
        ),
        Item(
            "parity-forward.png",
            "Forwards and discount factors from put-call parity",
            "S&P 500 options: the synthetic forward, its residuals, the implied rate curve and dividend yield.",
            market.parity_forward,
            3,
            "snapshot",
        ),
        Item(
            "american-parity.png",
            "Parity is an equality for European options only",
            "SPX against SPY and Apple: early exercise bends the synthetic forward.",
            market.american_parity,
            3,
            "snapshot",
        ),
        Item(
            "cleaning-funnel.png",
            "From a raw chain to clean quotes",
            "What each cleaning rule removes, and the largest executable arbitrage in the chain.",
            market.cleaning_funnel,
            3,
            "snapshot",
        ),
        Item(
            "market-smiles.png",
            "The S&P 500 smile across the term structure",
            "Bid, mid and ask implied volatility from a week to two years, with the vendor's figures.",
            market.market_smiles,
            3,
            "snapshot",
        ),
        Item(
            "vendor-iv.png",
            "Yahoo Finance's implied volatilities assume zero rates",
            "The vendor's figures against parity forwards, and the assumption that reproduces them.",
            market.vendor_iv,
            3,
            "snapshot",
        ),
    )


def build(out_dir: str | Path = "docs/images", only: str | None = None) -> list[Path]:
    """Draw every chart (or the one named, or one day's) into ``out_dir``."""
    written = []
    for item in items():
        if only and only not in (item.filename, str(item.day)):
            continue
        if not item.available():
            print(f"skipped {item.filename}: needs a local option-chain snapshot (volsurf chains fetch ^SPX SPY AAPL)")
            continue
        written.append(save(item.build(), Path(out_dir) / item.filename))
    return written


def markdown(prefix: str = "images") -> str:
    """The gallery page, generated, so it always lists exactly the charts there are."""
    lines = [
        "# Gallery",
        "",
        "Every chart is rebuilt from source with `volsurf gallery`. The figures of the live S&P 500 analysis "
        "(the surface, the smiles, the Heston fit, the hedging laboratory) come from `volsurf analyze` and are shown "
        "in the README.",
        "",
    ]
    for day, title in DAYS.items():
        lines += [f"## Day {day}: {title}", ""]
        for item in items():
            if item.day == day:
                lines += [f"**{item.title}** - {item.description}", "", f"![{item.title}]({prefix}/{item.filename})", ""]
    return "\n".join(lines)
