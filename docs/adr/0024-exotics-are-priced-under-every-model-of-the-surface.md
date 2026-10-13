# 24. Exotics are priced under every model of the surface, and the spread is reported

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Three models can be fitted to the same S&P 500 surface:

- Black-Scholes at one volatility;
- Heston (Day 5), to its calibration error;
- Dupire local volatility (Day 6), exactly.

Vanillas fix only the distribution of the index at each expiry. A path-dependent payoff
also depends on how the index moves between expiries, which vanillas do not determine.

## Decision

- **One path interface serves all three models** (`volsurf.exotics.monte_carlo`):
  - Black-Scholes by exact log-normal steps;
  - Heston by QE with Andersen's martingale correction;
  - local volatility by log-Euler on `ln(S/F)`, which keeps the price a martingale by
    construction.
- **A book of seven contracts** (`volsurf.exotics.model_risk`) is priced under each model:
  two vanillas to check the fit, then a digital, two knock-outs, a one-touch and an Asian.
- **The spread across models is reported as model risk**, with every Monte Carlo error.

## Consequences

The table is for the S&P 500 at one year, 9 October 2026, per unit of spot, with 100,000
paths per model.

| Contract | Black-Scholes | Heston | Local volatility |
|---|---|---|---|
| at-the-money call | 0.0848 | 0.0896 | 0.0898 |
| 90% put | 0.0150 | 0.0278 | 0.0266 |
| digital call at 105% | 0.426 | 0.538 | 0.527 |
| down-and-out call, barrier 80% | 0.0847 | 0.0859 | 0.0872 |
| up-and-out call, barrier 120% | 0.0196 | 0.0459 | 0.0341 |
| one-touch at 85% | 0.236 | 0.260 | 0.272 |
| Asian call, monthly fixings | 0.0499 | 0.0489 | 0.0489 |

- **Heston and local volatility price the at-the-money call within 0.2% of each other, but
  the up-and-out call 35% apart.** Both fit the expiry distributions; they differ in the
  path between them, and the barrier is priced by the path.
- **Black-Scholes at one volatility misses the skew.** It prices the 90% put 44% low and
  the 105% digital 19% low.
- **The Asian barely depends on the model.** The average is driven by the at-the-money
  variance, which every model matches.
