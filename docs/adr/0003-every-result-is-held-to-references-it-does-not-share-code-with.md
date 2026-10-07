# 3. Every result is held to references it does not share code with

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Tests written from the same formulas as the code check the transcription, not the
formulas. An error made in the derivation appears identically in both. The lab's
original Greek tests compared analytic Greeks with finite differences at a tolerance of
1e-4 to 1e-7. That catches a missing term, but at one contract and five Greeks.

## Decision

Every analytic result is checked against at least one reference that shares nothing
with it but the definition of the model:

- **Arbitrary precision.** mpmath differentiates the price to 50 digits, for every Greek
  to third order, on random contracts, to a relative 1e-11.
- **An independent library.** QuantLib's analytic European engine, on 500 random
  contracts, to 1e-10.
- **Published worked examples**, as printed. Haug's (2007) nine examples are
  reproduced to the fourth decimal.
- **Properties that must hold for any input**, property-tested with Hypothesis: the
  Black-Scholes PDE, parity, homogeneity and the model-free bounds.

QuantLib, mpmath and Hypothesis are development dependencies. The library itself depends
on none of them.

## Consequences

- The days that follow hold their engines to references in the same way: lattices and
  finite differences to high-precision prices, implied volatility to its definition, and
  the surface to its no-arbitrage conditions.
- A published example that cannot be confirmed is not used. A remembered vega example
  that did not reproduce was left out rather than adjusted.
