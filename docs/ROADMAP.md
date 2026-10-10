# Roadmap

The lab was first published on 16 September 2026 (v1.0.0): Black-76 pricing, implied
volatility, lattices, Monte Carlo, an SVI surface, Dupire local volatility, Heston
calibration with the COS method and a delta-hedging laboratory, run on a live S&P 500
option chain.

From October 2026 it is rebuilt and deepened over nine days, one area of the stack a day.
Each day is a GitHub issue, a branch, a series of small commits, a pull request with its
rationale, green CI and a release. Every day holds its area to references it does not
share code with: published worked examples, QuantLib, arbitrary-precision arithmetic,
real market data, or properties that must hold for any input.

| Day | Area | Issue | Release | Status |
|---|---|---|---|---|
| 1 | The analytic foundation: generalised Black-Scholes-Merton, seventeen Greeks, independent references | #1 | v1.1.0 | ✅ Done |
| 2 | Numerical engines: lattices, finite differences and Monte Carlo against high-precision references | #2 | v1.2.0 | ✅ Done |
| 3 | Implied volatility and the market: robust inversion, forward extraction, real option chains | #3 | v1.3.0 | ✅ Done |
| 4 | The arbitrage-free surface: SVI and SSVI, calendar and butterfly conditions, wings and densities | #4 | v1.4.0 | Next |
| 5 | Heston: characteristic functions, Fourier pricing, calibration and simulation | #5 | v1.5.0 | |
| 6 | Local volatility and SABR: Dupire in implied-volatility form, Hagan and its corrections | #6 | v1.6.0 | |
| 7 | Exotic options: barriers, Asians, lookbacks and digitals, analytic against simulation | #7 | v1.7.0 | |
| 8 | Hedging and model risk: Greeks P&L explain, discrete hedging, the book's risk | #8 | v1.8.0 | |
| 9 | The platform: CLI, dashboard, reproducible reports and packaging | #9 | v1.9.0 | |

## Day 1 - the analytic foundation ✅

**Issue #1, release v1.1.0.**

- **One model for every underlying.** Generalised Black-Scholes-Merton
  (`volsurf.analytic`) prices options on stocks, indices, futures (Black-76), currencies
  (Garman-Kohlhagen) and margined futures (Asay) through the cost of carry.
- **Seventeen Greeks, to third order.** Delta, vega, theta, rho, psi, dual delta; gamma,
  vanna, volga, charm, veta, dual gamma; speed, zomma, color, ultima. At expiry and at
  zero volatility they are their limits, where the lab's original Greeks divided by zero.
- **Three independent references:**
  - every Greek against the price differentiated to 50 digits (mpmath), agreeing to
    1e-14;
  - QuantLib on 500 random contracts, to 1e-10;
  - Haug's (2007) worked examples, reproduced to the printed digit.

  The mpmath reference caught veta and color with their signs turned.
- **Properties for any input**, property-tested: the Black-Scholes PDE, parity in every
  Greek, homogeneity, Merton's model-free bounds, and strike monotonicity and
  convexity.
- **Foundation for the days that follow:**
  - this roadmap, a changelog and architecture decision records;
  - strict typing for rebuilt code;
  - a generated chart gallery with six charts.

**Notes:** [the analytic foundation](notes/the-analytic-foundation.md).

## Day 2 - numerical engines ✅

**Issue #2, release v1.2.0.**

- **Five lattices.** CRR, Jarrow-Rudd, Tian, Leisen-Reimer and trinomial trees, with
  European, Bermudan and American exercise, BBS smoothing and Richardson extrapolation.
  Tian and Leisen-Reimer are QuantLib's trees to 1e-11.
- **Finite differences.** Crank-Nicolson with Rannacher smoothing, which cuts the gamma
  error near the strike from 0.85 to 1.5e-5. Early exercise by projection, and the
  exercise boundary, which matches its near-expiry asymptote.
- **Monte Carlo with honest errors.** Randomised scrambled Sobol', about 25,000 times
  less variance than pseudo-random at 2^18 samples. Least-squares Monte Carlo bracketed
  by an out-of-sample lower bound and a dual upper bound, 0.003 apart.
- **The references corrected.** Longstaff and Schwartz's (2001) widely quoted American
  column agrees, in 15 of 20 cases, with a Bermudan put exercisable fifty times a year,
  not with the American put. American references are now Andersen, Lake and
  Offengenden's (2016) high-precision values.

**Notes:** [numerical engines](notes/numerical-engines.md).

## Day 3 - implied volatility and the market ✅

**Issue #3, release v1.3.0.**

- **An exact Black function.** Prices come through the Mills ratio, with quadrature where
  the difference would cancel. `ln b` matches 60-digit arithmetic to a few roundings,
  down to `ln b = -6.8 million`. The textbook formula keeps nine digits of sixteen at
  total volatility 0.02.
- **An exact inverse.** The inversion is Jaeckel-style, with Householder steps on
  log-price objectives. Its worst error is 4.0 condition numbers on 1,816 well-posed
  quotes, against 4.1 for Jaeckel's own implementation. The v1.0 Newton iteration was
  wrong by 18% at a price of 1e-20 and by a factor of four at 1e-300.
- **Forwards from parity.** A repeated-median start, trimmed weighted least squares and
  one Nelson-Siegel curve across 52 SPX expiries give forwards to a median of 0.04
  index points. American chains, SPY and Apple, bend parity, and are fitted near the
  money only.
- **Clean real chains.** Snapshots are stored locally, and every quote is kept or
  dropped by a named rule. Strike arbitrage is tested at the bid and the ask; the first
  removal rule ate 364 good quotes of one expiry before it was fixed. Yahoo Finance's
  own implied volatilities turn out to assume zero rates and dividends.

**Notes:** [implied volatility and the market](notes/implied-volatility-and-the-market.md).
