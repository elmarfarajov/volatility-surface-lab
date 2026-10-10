# 12. No-arbitrage conditions are hard constraints, checked on the whole real line

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

The v1.0 SVI fitter added penalties for a negative density and for crossing the previous
slice, evaluated on 81 strikes. A penalty trades arbitrage against fit, and a grid says
nothing between or beyond its points.

Checked on the whole line, on the S&P 500 chain of 9 October 2026:

- 36 of its 48 slices had a negative density somewhere;
- 31 of 47 neighbouring pairs crossed;
- its interpolation between expiries reached `g = -3.9`.

## Decision

- **Every check covers the whole real line.**
  - The wings are settled analytically: `g` tends to `1/4 - b^2(1 +- rho)^2/16`, so Lee's
    bound `b(1 +- rho) <= 2` decides them.
  - The finite part is searched on a grid uniform in `asinh((k - m)/sigma)`, across 60
    smile widths either side, with every local minimum refined by Brent's method.
- **Each slice is fitted with the conditions as constraints**, solved as a semi-infinite
  programme by the exchange method:
  1. SLSQP fits on a finite set of strikes;
  2. the worst point on the whole line is found;
  3. that point is added to the set, and the fit runs again until the check passes.
- **The constraints carry small margins**, `g >= 1e-6` and
  `w - w_prev >= 1e-7 * theta`, because SLSQP meets an active constraint only to about
  1e-9.
- **Each fit starts from an eSSVI slice**, which is arbitrage-free by its own conditions.

## Consequences

- All 48 SPX slices are free of butterfly and calendar arbitrage on the whole line, in at
  most 9 exchange rounds.
- The median error is 0.19 volatility points against 0.16 for the v1.0 fit, and 68% of
  quotes sit inside their bid-ask range against 66%. Freedom from arbitrage costs
  almost nothing in fit.
- Where the quotes alone would imply a negative density, the fit sits on the bound
  `g = 1e-6`, and the density touches zero there. This is visible in the tails and stated
  rather than smoothed away.
- A candidate is accepted only if it fits at least as well as its arbitrage-free start.
  - On a synthetic chain, SLSQP once ended on "constraints incompatible" at a point that
    passed every check but missed the quotes by 23 volatility points. Accepted, it made
    the next expiry impossible to fit without crossing.
  - When no round succeeds, the start is kept if it does not cross the previous slice;
    otherwise the best arbitrage-free candidate is kept; otherwise the fit fails loudly.
- `volsurf analyze` still uses the v1.0 fitter. Moving the pipeline to `volsurf.smile` is
  part of the Day 9 platform work.
