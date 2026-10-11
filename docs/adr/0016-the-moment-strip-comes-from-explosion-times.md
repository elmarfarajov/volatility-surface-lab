# 16. The moment strip comes from explosion times, and closed forms are re-derived, not recalled

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

The contour integral needs `phi(-i(alpha + 1))` to be finite, so the damping must stay
inside the moment strip, the interval of `p` for which `E[S_T^p]` is finite. Lee's moment
formula then turns the strip's edges into the slopes of the smile's wings.

Three shortcuts were tried and failed.

- **Testing where `phi(-ip)` stops being a finite positive real number.** Beyond the
  explosion, the little-trap formula's analytic continuation turns real and positive
  again (`1e40` at `p = 40` for a strip whose edge is at 36). The edge jumped between 35
  and 76 from one maturity to the next.
- **The explosion time recalled for `D < 0`.** It tended to `2/k` as `D` rose to zero,
  where the true explosion time tends to infinity.
- **A closed form for the variance of `ln(S_T/F_T)`, recalled from the literature.** It
  disagreed with the characteristic function by a factor of 28.

## Decision

- **The explosion time is derived from the Riccati equation** for `psi` in
  `E[S_T^p] = exp(phi + psi v0)`, following Andersen and Piterbarg (2007). With
  `k = kappa - rho sigma p` and `D = k^2 - sigma^2 p(p - 1)`:
  - if `D >= 0` and `k >= 0`, it never explodes;
  - if `D >= 0` and `k < 0`, it explodes at `ln((|k| + sqrt D)/(|k| - sqrt D)) / sqrt D`;
  - if `D < 0`, it explodes at `2(pi/2 + arctan(k/gamma)) / gamma`, with `gamma = sqrt(-D)`.
- **The strip's edges solve `T*(p) = T`** by bisection, which is safe because `T*` is
  monotone in `p` away from [0, 1].
- **The variance of `ln(S_T/F_T)`**, which only sizes integration ranges, is the second
  derivative of `ln phi` by a five-point stencil.
- **Every closed form taken from memory is checked numerically** against the function it
  summarises before it is used.

## Consequences

- The edges vary smoothly with maturity. On the 80 edges of 40 random strips, `phi` is
  a finite positive real number 2% inside every edge, and is not one 2% outside it.
- The moment chart and Lee's wing slopes follow from the same function the engine uses.
