# 8. Each tree follows its paper, and the differences from QuantLib are stated

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

Several published binomial parameterisations share a name. QuantLib's "CRR" and
"Jarrow-Rudd" trees work in log-space with a probability that matches the drift to first
order. Cox, Ross and Rubinstein (1979) use the exact martingale probability
`p = (e^{b dt} - d) / (u - d)`.

## Decision

- CRR uses the 1979 paper's exact martingale probability.
- Jarrow-Rudd puts the drift in the moves and uses the risk-neutral probability.
- Tian matches the first three moments.
- Leisen-Reimer uses the Peizer-Pratt inversion and forces an odd number of steps.
- The trinomial tree uses `dx = sigma sqrt(3 dt)`.
- BBS smoothing and Richardson extrapolation (BBSR) are available on any lattice.

## Consequences

- Tian and Leisen-Reimer reproduce QuantLib's trees to 1e-11, which tests are pinned to.
- CRR and Jarrow-Rudd differ from QuantLib's variants by about 5e-5 at 301 steps. Both
  converge to the same price, and the difference is documented rather than hidden.
