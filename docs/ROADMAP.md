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
| 2 | Numerical engines: lattices, finite differences and Monte Carlo against high-precision references | #2 | v1.2.0 | Next |
| 3 | Implied volatility and the market: robust inversion, forward extraction, real option chains | #3 | v1.3.0 | |
| 4 | The arbitrage-free surface: SVI and SSVI, calendar and butterfly conditions, wings and densities | #4 | v1.4.0 | |
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
