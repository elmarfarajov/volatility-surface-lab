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
| 4 | The arbitrage-free surface: SVI and SSVI, calendar and butterfly conditions, wings and densities | #4 | v1.4.0 | ✅ Done |
| 5 | Heston: characteristic functions, Fourier pricing, calibration and simulation | #5 | v1.5.0 | ✅ Done |
| 6 | Local volatility and SABR: Dupire in implied-volatility form, Hagan and its corrections | #6 | v1.6.0 | ✅ Done |
| 7 | Exotic options: barriers, Asians, lookbacks and digitals, analytic against simulation | #7 | v1.7.0 | ✅ Done |
| 8 | Hedging and model risk: Greeks P&L explain, discrete hedging, the book's risk | #8 | v1.8.0 | ✅ Done |
| 9 | The platform: CLI, dashboard, reproducible reports and packaging | #9 | v1.9.0 | ✅ Done |

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

## Day 4 - the arbitrage-free surface ✅

**Issue #4, release v1.4.0.**

- **Checks on the whole real line.** Butterfly and calendar arbitrage are checked with the
  wings settled analytically through Lee's bound and the finite part searched and
  refined. Vogt's slice is caught at the published `g = -0.0329`.
- **The original surface had arbitrage.** Its penalised fit left a negative density in 36
  of 48 SPX slices and crossings in 31 of 47 pairs, and its interpolation reached
  `g = -3.9` between expiries.
- **Hard constraints.** Raw SVI is fitted by the exchange method for semi-infinite
  programmes, started from eSSVI. All 48 slices are arbitrage-free, at a median error of
  0.19 vol points against 0.16 for the unconstrained fit. SSVI's three-parameter shape,
  not its conditions, keeps it 2.6 points away.
- **Prices between expiries.** Gatheral and Jacquier's price interpolation between
  expiries, and a log-normal convolution beyond the last, keep every intermediate smile a
  valid distribution. Densities integrate to one and have the forward as their mean to
  2e-16.

**Notes:** [the arbitrage-free surface](notes/the-arbitrage-free-surface.md).

## Day 5 - Heston ✅

**Issue #5, release v1.5.0.**

- **Prices to the last digit.** A contour integral prices each side out of the money,
  with a graded quadrature and the damping kept inside the moment strip. It is within
  1e-14 of Lewis's integral at 40 digits in both wings. The v1.0 COS and Gil-Pelaez engines
  were 47% and 110% wrong for a deep ten-year call.
- **A published value corrected.** Fang and Oosterlee's printed reference price,
  5.785155450, is off by 1.6e-8; the 40-digit value is 5.78515543437619.
- **The moment strip** comes from Andersen and Piterbarg's explosion time, derived from the
  Riccati equation. Testing where the characteristic function stops being real fails,
  because its continuation turns real again beyond the explosion.
- **Calibrated to the S&P 500**, with its identifiability measured. Every start reaches the
  same minimum, but `kappa` is fixed only to about a factor of two, and the fit misses by
  2.5 vol points. **Simulated** with QE, which is within noise where Euler is a hundred
  standard errors out.

**Notes:** [Heston](notes/heston.md).

## Day 6 - local volatility and SABR ✅

**Issue #6, release v1.6.0.**

- **Local volatility in closed form.** It comes from the Day 4 price interpolation and is
  positive by construction; the v1.0 finite differences gave NaN on 4% of the grid. The
  forward Dupire equation reprices the monthly surface to within a few basis points.
- **What the interpolation hides.** It has an atom before the first expiry, which no
  diffusion carries; it puts a smile between flat expiries; and daily weeklies turn the
  local-volatility wings to noise. All three are measured and stated.
- **SABR held to the model.** An exact conditional reference, cross-checked with QuantLib's
  SABR PDE, shows Hagan's error growing from 4 bp to over 400 bp with `nu^2 T`. Its density
  turns negative at low strikes for long maturities.
- **Smile dynamics measured.** Local volatility moves the at-the-money volatility by 1.98
  to 2.2 times the skew against the index, while SABR moves it with the index. The
  at-the-money deltas range from 0.39 to 0.63.

**Notes:** [local volatility and SABR](notes/local-volatility-and-sabr.md).

## Day 7 - exotic options ✅

**Issue #7, release v1.7.0.**

- **Closed forms held to QuantLib.** Reiner-Rubinstein barriers, geometric Asians,
  lookbacks and digitals agree with QuantLib to 1e-12 or better on hundreds of random
  contracts.
- **Monte Carlo held to the closed forms.** The Brownian bridge recovers continuous
  monitoring from 12 dates. Broadie, Glasserman and Kou's correction holds daily but
  overprices a near up-and-out by 9% monthly and 22% quarterly, so discrete contracts are
  priced on their own grid.
- **A reference that is not.** With the geometric control variate (variance cut about
  900 times), two Monte Carlo engines agree on an arithmetic Asian. QuantLib's Choi engine
  is 0.40% low at 52 fixings.
- **One surface, three models.** On the S&P 500, Heston and local volatility agree on the
  at-the-money call to 0.2% and price a 120% up-and-out call 35% apart.

**Notes:** [exotic options](notes/exotic-options.md).

## Day 8 - hedging and model risk ✅

**Issue #8, release v1.8.0.**

- **Hedging in every model's world with every model's hedge.** One self-financing account
  is used throughout. Heston Greeks come from a COS grid and local-volatility Greeks from a
  backward Dupire PDE, and both are held to references. Derman and Kamal are reproduced to
  the third digit.
- **Model risk on the S&P 500.** Fitted to one surface, the models give a three-month call
  deltas from 0.41 to 0.73. Daily-hedged errors are 10% to 17% of the premium in a
  local-volatility world and 56% to 69% in the calibrated Heston world.
- **Thirty-six years of data.** Daily-hedged short straddles sold at the VIX made money 85%
  of the time; the gamma formula explains each trade. The market's smile dynamic is 1.8
  times sticky strike, and the minimum-variance delta removes 40% of hedging-error
  variance.
- **An options book's risk.**
  - **Stress:** a 10% fall costs $2.0m to $7.6m depending on the smile dynamic.
  - **Explain:** eight Greeks explain quiet days.
  - **Crashes:** second-order Greeks misstate them in both directions, so risk is measured
    by full revaluation.

**Notes:** [hedging and model risk](notes/hedging-and-model-risk.md).

## Day 9 - the platform ✅

**Issue #9, release v1.9.0.**

- **One analysis over every rebuilt engine.** `volsurf analyze` takes a stored snapshot
  through the steps of Days 3 to 8: cleaning, surface, Heston, SABR, local volatility,
  exotics, hedging and book risk. It writes an HTML report, figures, tables and a manifest.
  The v1.0 pipeline is retired.
- **Every number can be got again.** The manifest fingerprints the input by SHA-256 and
  records the configuration, versions and results. `volsurf reproduce` reruns and compares:
  52 of 52 numbers on the S&P 500 run.
- **A dashboard over every engine**, offline by default.
- **A package that works on its own.** CI installs the wheel without the development extras
  and runs the platform from it.

**Notes:** [the platform](notes/the-platform.md).

## After the nine days

The rebuild is complete:

- 52 validation checks, all passing;
- 31 architecture decision records;
- a methodology note for every day;
- 52 charts drawn from source;
- one command from a stored option chain to a reproducible report.

The open problems are where the notes leave them:

- Heston's short-dated skew, the natural next models being Bates and rough Bergomi;
- a history of option quotes in place of the VIX proxy;
- transaction costs in hedging.
