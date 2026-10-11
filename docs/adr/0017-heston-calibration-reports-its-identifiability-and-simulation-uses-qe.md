# 17. Heston calibration reports what it can identify, and simulation uses QE

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

A calibrated parameter set is often quoted as if each number were measured. Mean
reversion, long-run variance and vol-of-vol trade off against each other, and a single
optimiser run cannot show whether they do.

Fitted equity surfaces usually violate Feller's condition, which is where simple
discretisations of the variance fail.

## Decision

- **Calibration** minimises bid-ask-weighted errors in implied volatility, by trust-region
  least squares. Model prices come from the contour engine and are inverted with the Day 3
  inversion.
- **Every calibration can report three diagnostics:**
  - minima from a scrambled Sobol' design of starts;
  - the singular values and weakest direction of the Jacobian in log-parameters;
  - a profile of the error in `kappa`.
- **Simulation uses Andersen's QE with the martingale correction.** Euler with full
  truncation is kept as the comparison.

## Consequences

**On the S&P 500 surface of 9 October 2026** (13 expiries from a month to two years):

- every start reached the same minimum, and the Jacobian's condition number is 16;
- the weakest direction is mean reversion against long-run variance. Between `kappa` 2.5
  and 5 the error changes by 6% while `theta` falls from 0.064 to 0.051;
- the fit itself is the limitation: 2.5 volatility points, 3.5 bid-ask widths, with
  `sigma = 1.98` and a Feller ratio of 0.12. Heston cannot carry a steep one-month skew and
  a flat two-year one at once.

**Below the Feller boundary** (ratio 0.08):

- Euler is more than a hundred standard errors off at two steps a year, and still about ten off at 64;
- QE is within two from four steps.

**For one large step near zero**, QE puts 81% of the mass at zero where the exact law has
none. The exact law does have more than half its mass within 1e-4 of zero, and QE's mean is right
to 1e-5.
