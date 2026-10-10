# 14. SSVI and eSSVI are kept as arbitrage-free starting points, not as the fitted surface

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

SSVI (Gatheral and Jacquier, 2014) and its extension eSSVI (Hendriks and Martini, 2019)
are free of static arbitrage by conditions on their parameters. That is a strong reason
to use them as the surface itself.

On the S&P 500 chain of 9 October 2026, however:

- both missed the quotes by a median of 2.6 volatility points, with 6% of quotes inside
  their bid-ask range;
- fitted to a single expiry with no constraint at all, an SSVI slice missed by as much,
  2.6 to 3.1 points. The no-arbitrage conditions were not binding;
- eSSVI's curvature condition bound on 0 of 48 slices.

The three-parameter shape itself cannot follow the index smile.

The eSSVI calendar condition of Corbetta et al. (2019) makes both wing slopes rise. Used
alone, it left 12 of 47 pairs crossing near the money.

## Decision

- **SSVI** is fitted with its conditions held by the parameterisation:
  `eta = 2u / (1 + |rho|)` with `u` in `(0, 1]`, `gamma` in `(0, 1/2]`, and `theta` as a
  cumulative sum of non-negative increments.
- **eSSVI** is fitted slice by slice with box bounds that carry the conditions.
  - `psi` lies between a lower bound, from rising wing slopes, and an upper bound, from
    the butterfly conditions.
  - A penalty on a grid around the smile stops crossing between the slices.
  - Every pair is then checked on the whole line.
- **eSSVI slices start the constrained raw SVI fit** of
  [ADR 0012](0012-no-arbitrage-conditions-are-constraints-checked-on-the-whole-line.md),
  and the fitted surface is raw SVI.

## Consequences

- The fitted surface has five parameters a slice, fits as closely as the
  unconstrained fit, and is checked rather than assumed to be arbitrage-free.
- SSVI's theorem holds numerically: 200 random admissible surfaces showed no arbitrage
  on the whole line.
- The first eSSVI fitter used SLSQP on an unscaled objective and stalled at its
  starting point. Trust-region least squares on the reparameterised box fixed it.
- eSSVI's calendar penalty is soft. On the SPX chain, one of 47 pairs (2026-11-13) still
  crosses, by 1.1e-8 in total variance near the money. As a starting point that is
  harmless: the constrained SVI fit that follows has no crossing pair.
