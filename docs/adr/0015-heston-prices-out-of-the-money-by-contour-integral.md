# 15. Heston prices out of the money by a contour integral, held to a 40-digit reference

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

The v1.0 engines, COS and Gil-Pelaez, price one side and take the other from put-call
parity. Far out of the money that is a difference of two numbers near the forward.
Against Lewis's integral evaluated to 40 digits with mpmath:

- **COS** is wrong by 1e-5 a year out at `ln(K/F) = 0.64`, and by 47% for a ten-year call
  worth 7e-8 of the forward.
- **Gil-Pelaez** is wrong by 110% for the deepest ten-year call.
- **QuantLib's analytic engine** reaches 3e-7 in the far ten-year tail.

## Decision

- **Out-of-the-money options are priced directly** by Carr and Madan's contour integral,
  with Lee's (2004) extension: damping `alpha > 0` gives the call, `alpha < -1` the put.
  Nothing is subtracted.
- **The damping** minimises the integrand at the origin (Lord and Kahl, 2007). It is kept
  out of the outer 10% of the admissible interval, where the poles of the denominator and
  of `phi` sit too close.
- **The quadrature** is composite 16-point Gauss-Legendre.
  - The first panel is graded geometrically down to 1e-8, because poles near the contour
    make the integrand peak at the origin.
  - The remaining panels resolve the oscillation `e^{-iuk}`.
  - The integral is truncated where the integrand's modulus falls below 1e-17 of its
    value at the origin.
- **Speed:** one damping per side lets every strike of an expiry share one set of
  characteristic-function values.
- COS and the FFT are kept for comparison only.

## Consequences

- **Accuracy:** against the 40-digit reference, the contour engine is within 1e-14 in
  both wings for one week, one year and ten years. The one exception is a one-week call
  worth 4e-12 of the forward: right to about 1e-21 of the forward, which is 3e-11 relative
  on Windows and 3e-10 on CI's Linux. The difference is the platform's maths library.
- **Agreement:** it agrees with QuantLib to 2e-14 near the money.
- **A published value was wrong.** Fang and Oosterlee's (2008) printed reference,
  5.785155450, is off by 1.6e-8; the 40-digit value is 5.78515543437619. The validation
  suite now checks against the latter.
- **Four faults were found while building it, each pinned by a test or a chart:**
  - stopping the integral where the oscillating real part happened to be near zero
    (1e-9 on puts);
  - uniform panels missing the peak near a pole (3.7e-5 at `alpha = -1.5`);
  - a damping chosen at the very edge of the strip (95%);
  - an eight-way bisection that stepped back past its own bracket.
