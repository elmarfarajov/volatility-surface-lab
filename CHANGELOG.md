# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
