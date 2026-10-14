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

from ..risk import history
from . import analytic, exotics, hedging, heston, localvol, market, numerics, smiles
from .style import save

DAYS: dict[int, str] = {
    1: "The analytic foundation",
    2: "Numerical engines",
    3: "Implied volatility and the market",
    4: "The arbitrage-free surface",
    5: "Heston",
    6: "Local volatility and SABR",
    7: "Exotic options",
    8: "Hedging and model risk",
}


@dataclass(frozen=True)
class Item:
    filename: str
    title: str
    description: str
    build: Callable[[], Figure]
    day: int
    # "snapshot": drawn from a local option-chain snapshot (``volsurf chains fetch``); "history": from the
    # local S&P 500 and VIX history (``volsurf history fetch``); "snapshot+history": both
    requires: str | None = None

    def available(self) -> bool:
        needs = set((self.requires or "").split("+")) - {""}
        return ("snapshot" not in needs or market.snapshot_available()) and ("history" not in needs or history.available())


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
        Item(
            "vogt-slice.png",
            "A smile that fits and still admits arbitrage",
            "Axel Vogt's SVI slice: its negative density, and a refit under hard constraints.",
            smiles.vogt_slice,
            4,
        ),
        Item(
            "v1-arbitrage.png",
            "The original surface had arbitrage",
            "The v1.0 fit checked on the whole line: negative densities, crossing slices, and between expiries.",
            smiles.v1_arbitrage,
            4,
            "snapshot",
        ),
        Item(
            "smile-fits.png",
            "Fitting the S&P 500 smile without arbitrage",
            "v1.0 SVI, eSSVI and constrained SVI against bid-ask bands, a week to two years.",
            smiles.smile_fits,
            4,
            "snapshot",
        ),
        Item(
            "fit-quality.png",
            "What the no-arbitrage conditions cost in fit",
            "Error and share inside the spread for SSVI, eSSVI, v1.0 and constrained SVI.",
            smiles.fit_quality,
            4,
            "snapshot",
        ),
        Item(
            "risk-neutral-densities.png",
            "The distributions the S&P 500 options price",
            "Risk-neutral densities with their mass, mean and Breeden-Litzenberger checks.",
            smiles.densities,
            4,
            "snapshot",
        ),
        Item(
            "time-interpolation.png",
            "Between expiries: interpolate prices, not variances",
            "Total variance through time, and the worst density of every intermediate smile.",
            smiles.time_interpolation,
            4,
            "snapshot",
        ),
        Item(
            "lee-wings.png",
            "What the wings say about the moments",
            "Wing slopes against Lee's bound, and the moments of the index they imply.",
            smiles.lee_wings,
            4,
            "snapshot",
        ),
        Item(
            "heston-engines.png",
            "Heston prices to the last digit",
            "Five engines against a 40-digit reference, far into both wings.",
            heston.engine_accuracy,
            5,
        ),
        Item(
            "contour-damping.png",
            "Where the contour runs",
            "The integrand by damping, and why the quadrature is graded towards the origin.",
            heston.contour_damping,
            5,
        ),
        Item(
            "simulation-bias.png",
            "Simulating Heston across the Feller boundary",
            "Euler with full truncation and Andersen's QE against exact prices.",
            heston.simulation_bias,
            5,
        ),
        Item(
            "qe-transition.png",
            "QE against the exact variance transition",
            "One step near zero: exact non-central chi-squared, QE and Euler.",
            heston.qe_transition,
            5,
        ),
        Item(
            "moment-explosion.png",
            "The moment strip and the wings it implies",
            "Moment explosion through time, and Lee's wing slopes.",
            heston.moment_explosion,
            5,
        ),
        Item(
            "heston-calibration.png",
            "Heston calibrated to the S&P 500",
            "Smiles from a month to two years against the bid-ask, and where five parameters fall short.",
            heston.heston_calibration,
            5,
            "snapshot",
        ),
        Item(
            "heston-identifiability.png",
            "What the S&P 500 surface pins down",
            "Multistart, the profile in kappa, and the least determined direction of the Jacobian.",
            heston.heston_identifiability,
            5,
            "snapshot",
        ),
        Item(
            "local-vol-surface.png",
            "S&P 500 local volatility in closed form",
            "Local against implied volatility, the term structure, and the noise daily weeklies bring to the wings.",
            localvol.local_vol_surface,
            6,
            "snapshot",
        ),
        Item(
            "dupire-round-trip.png",
            "Repricing the surface from its own local volatility",
            "The forward Dupire equation against the surface, as the grid is refined.",
            localvol.dupire_round_trip,
            6,
            "snapshot",
        ),
        Item(
            "sabr-accuracy.png",
            "Hagan's SABR formula against the model",
            "The expansion against an exact conditional reference, by strike and by nu^2 T.",
            localvol.sabr_accuracy,
            6,
        ),
        Item(
            "sabr-density.png",
            "Where Hagan's formula stops being a model",
            "Negative densities and the model's own prices from QuantLib's SABR PDE.",
            localvol.sabr_density,
            6,
        ),
        Item(
            "sabr-spx.png",
            "SABR fitted to each S&P 500 expiry",
            "Parameters by maturity, and the fit against SVI.",
            localvol.sabr_spx,
            6,
            "snapshot",
        ),
        Item(
            "smile-dynamics.png",
            "Same smile today, different smiles tomorrow",
            "The smile after the index moves, under local volatility and SABR, and the deltas.",
            localvol.smile_dynamics,
            6,
            "snapshot",
        ),
        Item(
            "barrier-monitoring.png",
            "Barriers watched continuously or once a day",
            "Discrete monitoring by Monte Carlo, the BGK shifted barrier, and the Brownian bridge, against the closed form.",
            exotics.barrier_monitoring,
            7,
        ),
        Item(
            "asian-control-variate.png",
            "Arithmetic Asians: a control variate, and a reference that is not",
            "Variance removed by the geometric control, and QuantLib's Choi and Turnbull-Wakeman engines against it.",
            exotics.asians,
            7,
        ),
        Item(
            "lookback-monitoring.png",
            "Lookbacks: the extreme seen once a day",
            "Discrete against continuous lookbacks as monitoring is refined, and the closed forms against volatility.",
            exotics.lookbacks,
            7,
        ),
        Item(
            "digital-smile.png",
            "A digital is a call spread, and the smile prices it",
            "S&P 500 digital calls from the smile against Black-Scholes at one volatility.",
            exotics.digital_smile,
            7,
            "snapshot",
        ),
        Item(
            "exotic-model-risk.png",
            "One surface, three models, three prices",
            "A book of vanillas and exotics under Black-Scholes, Heston and local volatility.",
            exotics.exotic_model_risk,
            7,
            "snapshot",
        ),
        Item(
            "hedging-frequency.png",
            "Hedging in Black-Scholes' world, more and more often",
            "Delta hedging against Derman and Kamal's law, and delta-gamma hedging with a second option.",
            hedging.hedging_frequency,
            8,
        ),
        Item(
            "wrong-volatility.png",
            "Hedging at the wrong volatility",
            "The hedged P&L path by path against the gamma formula, and against realised volatility.",
            hedging.wrong_volatility,
            8,
        ),
        Item(
            "heston-hedging.png",
            "In Heston's world a delta hedge stops improving",
            "Black-Scholes, Heston and minimum-variance deltas, and a delta-vega hedge, against rebalancing frequency.",
            hedging.heston_hedging,
            8,
        ),
        Item(
            "hedging-across-models.png",
            "Hedging the S&P 500 with the wrong model",
            "A three-month call sold at the market and hedged daily in Heston and local-volatility "
            "worlds with each model's delta.",
            hedging.hedging_across_models,
            8,
            "snapshot",
        ),
        Item(
            "variance-premium.png",
            "Thirty-six years of selling S&P 500 straddles at the VIX",
            "Daily-hedged short straddles from 1990: the P&L, the gamma formula on real paths, and the losses.",
            hedging.variance_premium,
            8,
            "history",
        ),
        Item(
            "spot-vol-dynamics.png",
            "How the S&P 500's volatility moves with the index",
            "The VIX against index returns, the market's smile dynamic, and the minimum-variance delta on real data.",
            hedging.spot_vol_dynamics,
            8,
            "snapshot+history",
        ),
        Item(
            "book-risk.png",
            "An options book's risk under each smile dynamic",
            "A dealer's S&P 500 book: stress ladder, historical value at risk, and what the Greeks leave unexplained.",
            hedging.book_risk,
            8,
            "snapshot+history",
        ),
    )


def build(out_dir: str | Path = "docs/images", only: str | None = None) -> list[Path]:
    """Draw every chart (or the one named, or one day's) into ``out_dir``."""
    written = []
    for item in items():
        if only and only not in (item.filename, str(item.day)):
            continue
        if not item.available():
            print(f"skipped {item.filename}: needs local data ({item.requires}; volsurf chains fetch, volsurf history fetch)")
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
