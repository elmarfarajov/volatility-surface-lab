# 7. Monte Carlo always states its error, and early exercise is given as two bounds

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

A Monte Carlo price without an error is not a measurement. Two cases make this hard:

- **Quasi-random points** have no sampling variance of their own, so they produce no
  error estimate.
- **Longstaff-Schwartz, as usually reported, fits and prices on the same paths.** That
  gives a single number biased upwards by foresight, with no statement of how far the
  fitted exercise rule is from the optimal one.

## Decision

- Every estimator returns an `Estimate` with a standard error.
- Scrambled Sobol' sequences are randomised independently (16 times by default), and the
  spread of the estimates is their error.
- Early exercise returns an `EarlyExercise` holding:
  - the in-sample estimate;
  - a lower bound, the fitted rule followed on independent paths;
  - a dual upper bound (Rogers, 2002; Haugh and Kogan, 2004), whose martingale is built
    from the fitted value function, with its increments estimated by inner simulation
    one exercise date ahead.
- The regression includes the European value of the remaining option.

## Consequences

- At 2^18 samples randomised quasi-Monte Carlo has a standard error about 25,000 times
  smaller in variance than pseudo-random sampling. That figure is itself measured.
- For the Bermudan put the bounds close to within 0.003 of each other around the
  reference. Without the European regressor the dual bound sits 0.17 above it: a poor
  value function makes a poor martingale.
