# 26. The minimum-variance delta is the hedge ratio, and the smile dynamic behind it is measured

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

A delta is a statement about how the smile moves when the index moves. Three models fitted
to the same S&P 500 surface give a three-month at-the-money call deltas of:

| Model | Delta |
|---|---|
| Black-Scholes, sticky strike | 0.572 |
| local volatility | 0.435 |
| Heston, variance held | 0.728 |
| Heston, minimum variance | 0.410 |

The choice has to be made on evidence about the hedge, not on which model is preferred.

## Decision

- **The hedge ratio is the minimum-variance delta.** It adds to the model delta the
  expected volatility move per unit spot move:
  - under Heston, `dC/dS + (rho sigma / S) dC/dv` (Bakshi, Cao and Chen, 1997);
  - on data, `delta_BS + vega beta / S`, with `beta` the regression of daily VIX changes on
    index returns, estimated only on data before each day (Hull and White, 2017).
- **The smile dynamic is reported as a ratio, `c = beta / skew`:**
  - 0 is sticky moneyness;
  - 1 is sticky strike;
  - about 2 is local volatility (Day 6).

## Consequences

- **In a Heston world (vol of vol 0.6, `rho = -0.7`), with daily rebalancing:**

  | Hedge | P&L std, % of premium |
  |---|---|
  | Heston delta | 36% |
  | Black-Scholes delta at the implied volatility | 31% |
  | minimum-variance delta | 27% |
  | delta and vega, with a longer option | 7% |

  Heston's own delta is the worst: it ignores that volatility falls when the index rises.
- **In the calibrated S&P 500 Heston world (vol of vol 1.98)**, the minimum-variance delta
  leaves 56% of the premium. Black-Scholes leaves 66%, local volatility 64% and Heston's
  delta 69%.
- **On thirty-six years of data:**
  - the VIX moves -1.16 points per point of index return (R² 0.62);
  - the 30-day skew on 9 October 2026 is -0.65, so the market's dynamic is `c = 1.8`,
    between sticky strike and local volatility;
  - the minimum-variance delta removes 40% of the variance of daily hedging errors.
- **The ratio uses one day's skew against 36 years of slope.** The one-year rolling slope
  ranges from -0.5 to -2.3, so `c` is a statement about the present smile, not a constant.
