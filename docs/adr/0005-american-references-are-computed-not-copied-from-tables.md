# 5. American references are computed to high precision, not copied from tables

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The lab's validation measured its American tree against 4.478 for the put with S 36,
K 40, sigma 20% and one year, from Longstaff and Schwartz's (2001) Table 1. The tree was
off by 8e-3, and the check passed at a tolerance of 1e-2.

That table is the usual reference for least-squares Monte Carlo. Measured against
Andersen, Lake and Offengenden's (2016) high-precision method, its finite-difference
column is not the American price:

- In 15 of its 20 cases, every one-year case and every two-year case at 20% volatility,
  it agrees to within 0.0005 with a **Bermudan** put exercisable fifty times a year.
  That is the exercise schedule of their own Monte Carlo.
- It lies 0.003 to 0.009 below the American put.
- In the other five cases it lies between the two.

## Decision

- American references are computed by methods whose own error is far below the
  engine's. The standard is QuantLib's `QdFpAmericanEngine` in its high-precision
  scheme, and five independent methods converge to it.
- Bermudan references come from this package's Crank-Nicolson engine on a fine grid,
  checked against QuantLib's finite differences.
- Both are stored with their provenance in `numerics.references`, so the tests do not
  depend on QuantLib.
- A published table is compared with these, not used as truth.

## Consequences

- The American tree is now measured to 2.4e-4 and BBSR to 1e-5, where before it was 8e-3
  against the wrong option.
- The finding is kept as a test and a chart. It is stated as measured; what the table's
  authors computed is not claimed.
