# 22. Discrete monitoring is priced on its own grid; the continuity correction is checked, not assumed

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Real barriers and lookbacks are monitored at fixings, usually daily closes. Broadie,
Glasserman and Kou (1997) showed that the discrete price is, to first order, the continuous
price with the barrier moved away from spot by `exp(beta sigma sqrt(dt))`, where
`beta = -zeta(1/2)/sqrt(2 pi) = 0.5826`.

The correction is an asymptotic result in `dt`. Its error grows when the barrier is close
to the region where the option pays, measured in units of `sigma sqrt(dt)`.

## Decision

- **Discretely monitored contracts are priced on their own monitoring grid** by Monte
  Carlo. The continuity correction is a fast estimate, used only where it has been checked.
- **The validation suite** holds the correction to discrete Monte Carlo at daily
  monitoring, for a down-and-out and an up-and-out call.
- **Lookbacks are never priced continuously for a discrete contract.**

## Consequences

The table is for a call struck at 100: one year, volatility 25%, rate 5%, carry 3%, and
200,000 paths.

| Contract | Monitoring | Continuous | BGK | Discrete Monte Carlo |
|---|---|---|---|---|
| down-and-out, barrier 85 | daily | 9.923 | 10.101 | 10.115 ± 0.039 |
| down-and-out, barrier 85 | quarterly | 9.923 | 10.824 | 10.816 ± 0.039 |
| up-and-out, barrier 120 | daily | 0.673 | 0.794 | 0.796 ± 0.006 |
| up-and-out, barrier 120 | weekly | 0.673 | 0.957 | 0.937 ± 0.007 |
| up-and-out, barrier 120 | monthly | 0.673 | 1.331 | 1.221 ± 0.008 |
| up-and-out, barrier 120 | quarterly | 0.673 | 1.968 | 1.614 ± 0.009 |

- **The discrete premium is large even daily.** Monitored daily, the up-and-out is worth
  18% more than continuously.
- **The correction holds for the down-and-out at every grid.** That barrier sits far from
  where the call pays.
- **For the up-and-out the correction runs high:** by 2% weekly, 9% monthly and 22%
  quarterly. That barrier sits just above where the call pays most.
- **Lookbacks converge slowly.** The floating-strike lookback call is worth 19.30
  continuously and 18.64 monitored daily. The discrete minimum misses the true one by
  about `0.58 sigma sqrt(dt)`, so the gap shrinks only as `sqrt(dt)`.
