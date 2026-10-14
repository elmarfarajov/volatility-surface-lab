# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-10-15

Day 9: the platform. One reproducible analysis over every rebuilt engine, its report and
manifest, a dashboard over every engine, and the wheel tested on its own. The last day of
the nine-day rebuild.

### Added

- **`volsurf.platform`**:
  - `analysis`: quotes to surface, Heston, SABR, local volatility, exotics, hedging and a
    book's risk in one run;
  - `sources`: stored snapshots and synthetic chains, fingerprinted by SHA-256;
  - `report`: an HTML report with an interactive surface, eight figures and six tables;
  - `manifest`: the run's record, and `reproduce`, which reruns it and compares every
    number.
- **`volsurf reproduce <dir>`**, and `--quick`, `--no-exotics`, `--no-hedging` and
  `--day` for `analyze` and `demo`.
- **`volsurf price`** quotes every rebuilt engine:
  - analytic, both trees, the PDE and Sobol' Monte Carlo;
  - for American exercise, the tree, the PDE and the Longstaff-Schwartz lower and dual
    upper bounds;
  - Heston by contour integral, COS and QE.
- **A `package` job in CI**: build, `twine check`, then the wheel installed alone runs
  `price`, `demo --quick` and `reproduce`. The package ships `py.typed`.
- **A validation row:** a rerun of the platform reproduces every number, and recovers
  the synthetic chain's Heston model.
- **A methodology note and ADRs 0029 to 0031.**

### Changed

- **`volsurf analyze` reads a stored snapshot** (`volsurf chains fetch`) instead of
  downloading. The `--raw-csv` and `--save-raw` flags are removed; fingerprinted
  snapshots replace them.
- **`volsurf demo` runs the same v2 analysis** on a synthetic Heston chain.
- **The dashboard is rebuilt over the v2 engines**, with six tabs (pricer, surface, models,
  exotics, hedging, risk). It reads stored snapshots and never downloads on its own.
- **The README's results come from the platform's run** on the 9 October 2026 snapshot,
  replacing the v1.0 figures.
- **Strict mypy now covers `volsurf.platform` and the command line.**
- **The gallery imports its chart modules lazily**, so the report needs no development
  extras.

### Removed

- **The v1.0 `pipeline`, `report`, `figures`, `calibration` and `hedging` modules**, their
  tests, the figures they drew for the README, and the v1 calibration validation row.
  Day 5's recovery row covers that check, and the Derman-Kamal row now uses
  `volsurf.risk`.

### Moved

- **The v1.0 `svi` and `surface` modules to `volsurf.legacy`.** They are kept only to
  reproduce the arbitrage Day 4 found in them.

### Found

- **The report could not run from an installed wheel without QuantLib.** It reached the
  gallery's house style, and with it every gallery module's top-level imports. The
  wheel-only CI job caught it.
- **The full S&P 500 analysis reproduces all 52 of its numbers** from its manifest.

## [1.8.0] - 2026-10-14

Day 8: hedging and model risk. Hedging measured in each model's world with each model's
hedge, thirty-six years of the S&P 500 and the VIX, and an options book's explain, stress
and historical risk.

### Added

- **`volsurf.risk.hedging`**:
  - a self-financing hedging account for any paths and any hedge, the underlying and other
    options included;
  - Derman-Kamal and delta-gamma studies;
  - the gamma P&L of hedging at the wrong volatility, path by path.
- **`volsurf.risk.pricers`**:
  - Heston prices and Greeks along paths, from a COS grid vectorised over strikes;
  - local-volatility prices and deltas from a backward Dupire PDE;
  - the minimum-variance delta.
- **`volsurf.risk.models`**: hedging in Heston's world with four strategies, and the S&P 500
  cross-model study (Heston and local-volatility worlds; Black-Scholes, local-volatility
  and Heston hedges).
- **`volsurf.risk.history`** and `volsurf history fetch`:
  - S&P 500, VIX, VIX3M and bill-rate history, stored locally;
  - a 1990-2026 backtest of daily-hedged short straddles;
  - the spot-volatility regression;
  - out-of-sample minimum-variance hedging errors.
- **`volsurf.risk.explain`** and **`volsurf.risk.book`**:
  - an eight-Greek P&L explain, with charm and veta;
  - a dealer book's stress ladder under four smile dynamics;
  - historical VaR and expected shortfall, from full revaluation and from Greeks.
- **Gallery:** charts may now require the local history as well as the snapshot.
- **Seven charts**, a methodology note, and ADRs 0025 to 0028.
- **Four rows in `volsurf validate`**:
  - the gamma formula;
  - the eight-term explain;
  - the local-volatility PDE against Black-Scholes;
  - the Heston grid against the contour integral.
- **Tests** for the accounting, both pricers, the studies, the backtest and the book, on
  synthetic data.

### Found

- **The minimum-variance delta wins in simulation and on data.**
  - **Heston's world:** it leaves 27% of the premium against 31% for Black-Scholes and 36%
    for Heston's own delta.
  - **S&P 500 data:** it removes 40% of daily hedging-error variance.
- **The S&P 500's smile moves 1.8 times as far as sticky strike**, measured from 36 years of
  the VIX against today's skew: between sticky strike and local volatility.
- **Short straddles sold at the VIX and delta-hedged** made money 85% of the time and kept
  21% of the premium. The worst trade, in September 2008, lost 9.6% of the index.
- **For a dealer book, the smile dynamic changes a 10% stress from $2.0m to $7.6m.**
- **Second-order Greeks misstate crash days in both directions**: they understate 2008 and
  overstate 2020.
- **A gamma hedge's error falls as `N^-0.67`, not `1/N`.**

## [1.7.0] - 2026-10-13

Day 7: exotic options. Closed forms held to QuantLib, Monte Carlo held to the closed forms,
and one book of exotics priced under every model of the S&P 500 surface.

### Added

- **`volsurf.exotics.closed_form`**:
  - Reiner-Rubinstein barriers with rebates;
  - Kemna-Vorst geometric Asians, continuous and discrete;
  - Goldman-Sosin-Gatto floating and Conze-Viswanathan fixed lookbacks;
  - cash- and asset-or-nothing digitals.
- **`volsurf.exotics.monte_carlo`**:
  - paths under Black-Scholes, Heston (QE) and local volatility on one interface;
  - discretely monitored barriers, the Brownian-bridge estimator of continuous monitoring,
    and Broadie, Glasserman and Kou's continuity correction;
  - arithmetic Asians with the geometric control variate.
- **`volsurf.exotics.model_risk`**: a seven-contract book priced under Black-Scholes, Heston
  and local volatility.
- **Tests:**
  - every closed form against QuantLib, and the in-out and digital parities;
  - the bridge, the correction and where it fails;
  - the control variate against QuantLib's Monte Carlo;
  - martingale and repricing checks for each path model, and the book on a Heston surface.
- **Five charts**, a methodology note, and ADRs 0021 to 0024.
- **Five rows in `volsurf validate`**: barriers, geometric Asians and lookbacks against
  QuantLib; the bridge; the continuity correction; the arithmetic Asian against QuantLib's
  Monte Carlo.

### Found

- **QuantLib's `ChoiAsianEngine` is 0.40% low at 52 fixings**, about 48 standard errors,
  though converged in its own parameters and right at 4 and 7 fixings. Two independent
  Monte Carlo engines agree with each other.
- **The BGK correction overprices an up-and-out call** 20% from spot by 9% monitored
  monthly and 22% quarterly; it holds daily.
- **On the S&P 500, Heston and local volatility agree on the at-the-money call to 0.2% but
  price a 120% up-and-out call 35% apart.**

### Fixed

- **`CITATION.cff`** had its schema version bumped in place of the software version since
  v1.3.0; it now reads `cff-version: 1.2.0`, `version: 1.7.0`.

## [1.6.0] - 2026-10-13

Day 6: local volatility and SABR. Dupire in closed form from the arbitrage-free surface,
Hagan's expansion against the model it approximates, and the smile dynamics, and deltas,
each model implies.

### Added

- **`volsurf.localvol.dupire`**:
  - local variance in closed form from the Day 4 price interpolation, positive by
    construction, with a self-similar short end before the first expiry;
  - a forward Dupire solver (Crank-Nicolson after Rannacher steps) that reprices the
    surface.
- **`volsurf.localvol.sabr`**:
  - Hagan's lognormal formula and Obloj's correction;
  - an exact conditional Monte Carlo reference for `beta = 1`;
  - Breeden-Litzenberger densities;
  - calibration at fixed `beta`.
- **`volsurf.localvol.dynamics`**: the smile after the index moves, under frozen local
  volatility and under SABR, and Black, local-volatility, SABR and Bartlett deltas.
- **Tests:**
  - a flat surface;
  - the forward-equation round trip;
  - Hagan against QuantLib, and the reference against QuantLib's SABR PDE;
  - negative densities and calibration;
  - the factor-of-two smile dynamics.
- **Six charts**, a methodology note, and ADRs 0018 to 0020.
- **Four rows in `volsurf validate`**: the Dupire round trip, local-volatility smile
  dynamics, Hagan against QuantLib, and the exact reference against QuantLib's PDE.

### Found

- **The Day 4 price interpolation keeps an atom at the forward before the first expiry.**
  No diffusion can carry it, so a forward Dupire solve from expiry stopped converging. The
  local-volatility short end is now self-similar.
- **Between expiries, the price interpolation of a flat surface has a smile**, 2.8 points
  at `ln(K/F) = -0.4`. It is free of arbitrage, but not minimal.
- **Local volatility from daily weeklies is noise in the wings**, swinging from 8% to over
  200% within two weeks.
- **The v1.0 `VolSurface.local_vol` returned NaN on 4% of the S&P 500 grid.**
  `volsurf analyze` still uses it until the Day 9 platform work.

## [1.5.0] - 2026-10-11

Day 5: Heston. Prices to the last digit in both wings, a published reference corrected,
calibration that reports what it can identify, and simulation across the Feller boundary.

### Added

- **`volsurf.stochvol.characteristic`**: the little-trap characteristic function, exact
  mean and stencil variance of the log-price, and the moment strip from Andersen and
  Piterbarg's explosion time.
- **`volsurf.stochvol.pricing`**: out-of-the-money prices by Carr and Madan's contour
  integral with Lee's extension. The damping is Lord and Kahl's, kept off the edges of the
  strip; the quadrature is composite Gauss-Legendre graded towards the origin, with one set
  of characteristic-function values per side. COS and the Carr-Madan FFT are kept for
  comparison.
- **`volsurf.stochvol.calibration`**: bid-ask-weighted calibration in implied volatility,
  with multistart, Jacobian singular values and a profile in `kappa`.
- **`volsurf.stochvol.simulation`**: Euler with full truncation and Andersen's QE with the
  martingale correction, the exact non-central chi-squared transition, and Monte Carlo
  estimates with standard errors.
- **Tests:**
  - against Lewis's integral at 40 digits, QuantLib and an independent quadrature;
  - explosion times and their limit;
  - calibration recovery and the profile;
  - QE against the exact transition, and the Feller-boundary bias.
- **Seven charts**, a methodology note, and ADRs 0015 to 0017.
- **Six rows in `volsurf validate`**:
  - the contour engine against 40 digits, the published price and QuantLib;
  - calibration recovery;
  - QE and Euler below the Feller boundary;
  - the moment strip.

### Found

- **The v1.0 Heston engines lose the out-of-the-money wing.** COS was 1e-5 wrong a year out
  at `ln(K/F) = 0.64`, and both COS and Gil-Pelaez were about 100% wrong for a deep ten-year
  call. `volsurf analyze` still uses them, and moves to `volsurf.stochvol` with the Day 9
  platform work.
- **Fang and Oosterlee's (2008) printed reference price is off in its eighth digit**:
  5.785155450 against 5.78515543437619 at 40 digits.

## [1.4.0] - 2026-10-11

Day 4: the arbitrage-free surface. Smiles and the surface between them are now free of
static arbitrage on the whole real line. The original fitter was not.

### Added

- **`volsurf.smile.svi`**: raw, natural and jump-wings SVI with exact conversions and
  analytic derivatives. Butterfly and calendar checks cover the whole real line: the wings
  through Lee's bound, the finite part by an asinh grid refined by Brent's method.
- **`volsurf.smile.ssvi`**:
  - SSVI with the power-law `phi`, arbitrage-free by its parameterisation;
  - eSSVI fitted slice by slice, with box bounds that carry the butterfly conditions and
    rising wing slopes.
- **`volsurf.smile.fit`**: raw SVI fitted under hard butterfly, Lee and calendar
  constraints by the exchange method for semi-infinite programmes, started from eSSVI.
- **`volsurf.smile.interpolation`**: Gatheral and Jacquier's price interpolation of
  out-of-the-money options between expiries, and a log-normal convolution, by
  Gauss-Hermite, beyond the last. Implied volatilities come through the Day 3 inversion.
- **`volsurf.smile.density`**: risk-neutral densities in closed form, with mass, martingale
  and Breeden-Litzenberger checks.
- **`volsurf.smile.build`**: from a cleaned Day 3 market to every model in one call.
- **Tests:**
  - Hypothesis round trips of the parameterisations;
  - Vogt's slice, and arbitrage beyond any fitting grid;
  - the SSVI theorem on random surfaces, and parameter recovery;
  - a constrained surface on a Heston chain, with its densities;
  - interpolation free of arbitrage and exact at the nodes, and extrapolation against
    brute-force convolution.
- **Seven charts**, a methodology note, and ADRs 0012 to 0014.
- **Five rows in `volsurf validate`**: Vogt found and repaired, the SSVI theorem, the
  constrained fit, densities, and price interpolation.

### Found

- **The v1.0 surface admits arbitrage.** On the SPX chain of 9 October 2026:
  - 36 of its 48 slices have a negative density somewhere;
  - 31 of 47 neighbouring pairs cross;
  - its interpolation reaches `g = -3.9` between expiries.

  `volsurf analyze` still uses it; the pipeline moves to `volsurf.smile` with the Day 9
  platform work.

## [1.3.0] - 2026-10-10

Day 3: implied volatility and the market. The Black function and its inverse are now
exact in the far tails, and real option chains are snapshotted and cleaned with a stated
reason for every quote dropped.

### Added

- **`volsurf.market.black`**: the normalised Black function through the Mills ratio,
  with 16-point Gauss-Legendre where the difference would cancel and Laplace's continued
  fraction in the tail. `ln b`, the log-vega and the distance to the upper bound are each
  available without cancellation or underflow.
- **`volsurf.market.implied`**: Jaeckel-style inversion with third-order Householder
  steps on `ln b` and `ln(b_max - b)`, bracketed, from asymptotic starting points. It
  returns the iterations and the branch used.
- **`volsurf.market.parity`**:
  - forward and discount factor per expiry from a repeated-median start and weighted,
    trimmed least squares, with standard errors and a bid-ask "inside" check;
  - a fixed-discount mode for American chains;
  - a three-factor Nelson-Siegel discount curve.
- **`volsurf.market.cleaning`**: eight named rules with a funnel that adds up, strike
  arbitrage tested at the bid and the ask, and implied volatilities at the bid, mid and
  ask.
- **`volsurf.market.chains`** and **`volsurf chains fetch|list`**: local snapshots of
  every listed expiry. SPX AM and SPXW PM settlement are told apart.
- **Tests:**
  - against mpmath at 50 and 60 digits, Jaeckel's `py_lets_be_rational` and QuantLib;
  - on synthetic chains with known forwards and injected stale quotes;
  - Hypothesis round trips;
  - a regression test for the cascade in the arbitrage filter.
- **Eight charts**, a methodology note, and ADRs 0009 to 0011.
- **Four rows in `volsurf validate`**: the Black function in the far tails, the inversion
  against its conditioning, prices from 1e-300 to 1e-10, and forward and rate recovery
  from parity.

### Changed

- **`black76_price`** now evaluates the Mills-ratio form. The literal
  `F N(d1) - K N(d2)` lost up to seven digits out of the money at small volatility.
- **`implied_vol_black76` and `implied_vol_bsm`** now call the new inversion. Their
  unused `tol` and `max_iter` arguments were removed.
- **`mypy --strict`** now covers `volsurf.market`. `py_lets_be_rational` joins the dev
  extras as an independent reference.

### Fixed

- **Implied volatility of small prices.** The v1.0 Newton iteration stopped on an
  absolute price tolerance. It was right to 4e-12 above a normalised price of 1e-5,
  which covers every real quote, but wrong by 18% by 1e-20 and by a factor of four by
  1e-300.

## [1.2.0] - 2026-10-09

Day 2: numerical engines. Lattices, finite differences and Monte Carlo, each held to
references whose own error is far below the engine's. One of those references was the
wrong option.

### Added

- **`volsurf.numerics.lattice`**: Cox-Ross-Rubinstein, Jarrow-Rudd, Tian, Leisen-Reimer
  and trinomial trees, with European, Bermudan and American exercise and delta and
  gamma read off the tree. Broadie and Detemple's BBS smoothing and Richardson
  extrapolation (BBSR) apply to any of them.
- **`volsurf.numerics.pde`**: the Black-Scholes PDE by the theta-scheme on a log-spot
  grid, with:
  - Crank-Nicolson with Rannacher's implicit half-steps by default;
  - early exercise by projection;
  - the early-exercise boundary.
- **`volsurf.numerics.montecarlo`**:
  - European prices by pseudo-random, antithetic, control-variate and randomised
    scrambled Sobol' sampling, each with a standard error;
  - Longstaff-Schwartz returning an out-of-sample lower bound and a dual upper bound,
    with the European value of the remaining option among the regressors.
- **`volsurf.numerics.references`**: Longstaff and Schwartz's Table 1 beside the
  American prices (Andersen, Lake and Offengenden, 2016, through QuantLib's
  high-precision scheme) and the fifty-date Bermudan prices.
- **Tests:**
  - trees identical to QuantLib's;
  - convergence orders measured;
  - Rannacher's effect on gamma;
  - the exercise boundary against its asymptote;
  - Monte Carlo errors and the two bounds;
  - Hypothesis properties of early exercise.
- **Six charts**, a methodology note, and ADRs 0005 to 0008.
- **Six rows in `volsurf validate`**: BBSR and Crank-Nicolson on the American
  references, what the Longstaff-Schwartz column is, the two bounds, and Sobol' against
  pseudo-random.

### Fixed

- **The American checks were measured against the wrong option.** `volsurf validate`
  held the American tree to 4.478, Longstaff and Schwartz's value. That value is
  measured here to be the Bermudan price, 0.0087 below the American. The tree is now
  held to 4.486674 and the least-squares Monte Carlo to the Bermudan 4.477792.

## [1.1.0] - 2026-10-07

Day 1: the analytic foundation. The Black-Scholes core becomes one generalised model for
every underlying, with seventeen Greeks to third order. Each is held to three references
it shares no code with.

### Added

- **`volsurf.analytic`**: generalised Black-Scholes-Merton. Stocks, indices, futures
  (Black-76), currencies (Garman-Kohlhagen) and margined futures (Asay) are priced
  through the cost of carry (`Carry`).
- **Seventeen Greeks**, first to third order: delta, vega, theta, rho, psi, dual delta,
  gamma, vanna, volga, charm, veta, dual gamma, speed, zomma, color and ultima. Each has
  stated units and conventions, and its limits at expiry and at zero volatility.
- **Model-free bounds**: Merton's price bounds and the strike conditions (monotonic,
  slope at most the discount factor, convex), reported as violations with their size.
- **Independent references**:
  - `analytic.reference` differentiates the price in 50-digit arithmetic (mpmath);
  - `analytic.published` holds Haug's (2007) worked examples;
  - QuantLib is checked on 500 random contracts.
- **Property tests** (Hypothesis): the Black-Scholes PDE, put-call parity in price and
  in every Greek, homogeneity, the bounds, and the limits at expiry.
- **`volsurf greeks`**, which prints a contract's price and seventeen Greeks.
- **`volsurf gallery`**, which redraws the documentation's charts and generates
  `docs/GALLERY.md`; six charts of the analytic engine.
- **Three rows in `volsurf validate`**: the published examples, every Greek against
  arbitrary precision, and the PDE on 20,000 random contracts.
- **Project records**: a roadmap of the nine days, this changelog, architecture
  decision records, a methodology note, and strict mypy for the rebuilt packages.

### Fixed

- **`bsm_greeks` divided by zero at expiry and at zero volatility.** It now returns the
  limits: delta a step, the curvature Greeks zero, and theta the decay of the discounted
  intrinsic value. Exactly at the money, where no limit exists, it returns NaN.

### Measured

- **The price in the far wings** has a relative error below 5e-11 against 60-digit
  arithmetic, through 270 orders of magnitude, until the double-precision range ends.
  The original documentation called this "machine precision"; it is close, and now
  measured.

## [1.0.0] - 2026-09-16

The lab as first published. It works from a live S&P 500 option chain through to a
calibrated stochastic-volatility model and a measurement of hedging error:

- Black-76 pricing and analytic Greeks;
- safeguarded Newton implied volatility;
- Leisen-Reimer and Cox-Ross-Rubinstein trees;
- Monte Carlo with control variates, Andersen's QE scheme and Longstaff-Schwartz;
- the Heston model with the COS method and Gil-Pelaez quadrature;
- raw SVI slices with arbitrage penalties;
- a monotone total-variance surface with Dupire local volatility;
- Heston calibration;
- a delta-hedging laboratory in Black-Scholes and Heston worlds;
- a 16-check validation suite, an HTML report and a Streamlit dashboard.
