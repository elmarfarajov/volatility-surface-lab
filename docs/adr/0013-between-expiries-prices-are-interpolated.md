# 13. Between expiries prices are interpolated, and beyond the last the distribution is convolved

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

Interpolating total variance at fixed log-moneyness keeps calendar-free slices
calendar-free. It does not keep them butterfly-free. With a running maximum to patch
crossings, the v1.0 surface had a negative density in 24 of 140 intermediate smiles of
the SPX surface.

## Decision

- **Between expiries**, out-of-the-money prices are interpolated by Gatheral and
  Jacquier's (2014) convex combination, with weights from `sqrt(theta)` and `theta`
  linear in time. A convex combination of valid call-price functions is valid, and the
  weight's monotonicity gives calendar freedom.
  - Out-of-the-money prices are interpolated, not calls. Calls and puts differ by a
    linear intrinsic value, so the combination is the same, and nothing cancels when a
    deep in-the-money price is inverted. Interpolating calls first cost 8e-4 in
    volatility at `k = -0.8`.
- **Before the first expiry**, the lower slice is expiry itself, with `theta = 0`.
- **Beyond the last expiry**, the terminal distribution is convolved with independent
  log-normal noise of variance `theta_t - theta_n`. The extended process is a martingale
  growing in convex order, so it is free of static arbitrage by construction. The
  prices are a 60-point Gauss-Hermite average of the last slice's prices.
  - Mixing with the price bound 1 was rejected: calls would no longer tend to zero as
    the strike grows.
- **Implied volatilities** come from these prices through the Day 3 inversion.

## Consequences

- On the SPX surface the smallest density is non-negative at every intermediate
  maturity, and call prices never fall with maturity, out to twice the last expiry.
- The surface is continuous at the last expiry to 1e-10. The Gauss-Hermite extrapolation
  agrees with brute-force convolution to 1e-9, at about a hundredth of the cost.
- Interpolated smiles are not SVI, and local volatility needs a time derivative. Day 6
  takes that derivative from the price combination, not from a parametric form.
