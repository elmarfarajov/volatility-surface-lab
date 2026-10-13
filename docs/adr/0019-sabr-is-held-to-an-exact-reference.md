# 19. SABR's expansion is held to an exact reference, not to itself

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Hagan's SABR formula is an asymptotic expansion, but it is routinely used as if it were
the model. Comparing it with QuantLib's `sabrVolatility` checks the algebra, not the
approximation.

## Decision

- **Hagan's lognormal formula and Obloj's (2008) correction** are implemented and checked
  against QuantLib's `sabrVolatility`, to 2e-15 relative.
- **For `beta = 1`, the reference is exact in law.**
  - Given the volatility path, `ln F_T` is normal (Willard's conditioning), so a price is
    a Black price averaged over simulated paths of `alpha`.
  - Only the integral of `alpha^2` is discretised, by the trapezoid rule, and paths are
    antithetic.
  - The reference is checked against QuantLib's SABR finite-difference engine at
    `beta = 0.999`, which agrees to within the Monte Carlo noise.
- **For `beta < 1`**, QuantLib's SABR PDE is the reference.
- **The densities that Hagan's volatilities imply** are computed by Breeden and
  Litzenberger, to show where the formula stops describing a distribution.

## Consequences

- With `alpha 0.2, rho -0.7, nu 1`, Hagan's error in the put wing is 4 bp at five weeks,
  55 bp at six months, 150 bp at a year, and over 400 bp at two years. At the money it stays below 5 bp
  while `nu^2 T < 0.3`.
- Obloj's correction coincides with Hagan at `beta = 1` and matters only for `beta < 1`.
- **A rates-style smile** (`beta 0.5`, forward 3%) has a negative density below 0.75% at
  five years and below 1.6% at ten. Its volatilities part from the PDE's by 14 points at a
  0.5% strike at five years, and by 28 at ten.
- On the S&P 500, `beta = 1` fits each monthly expiry, but its parameters must change from
  expiry to expiry. A single SABR is a smile model, not a surface model.
