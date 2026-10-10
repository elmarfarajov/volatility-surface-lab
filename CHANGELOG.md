# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
