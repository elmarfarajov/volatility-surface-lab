# 23. The arithmetic Asian reference is Monte Carlo with the geometric control, not an analytic engine

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

The arithmetic-average option has no closed form. Fast alternatives exist:
Turnbull-Wakeman's moment matching, and semi-analytic engines such as Choi's, which
QuantLib provides as `ChoiAsianEngine`. It is tempting to treat a converged engine as the
reference.

## Decision

- **The reference is Monte Carlo with the geometric Asian as control variate** (Kemna and
  Vorst). The control's exact price is the Day 7 closed form, and every price comes with
  its standard error.
- **It is checked against an independent Monte Carlo**, QuantLib's
  `MCDiscreteArithmeticAPEngine` with its own control variate, in `volsurf validate`.
- **Analytic engines are measured against it**, not the other way round.

## Consequences

- **The control removes most of the variance.** With weekly fixings it divides the
  variance by:
  - about 900 at the money at volatility 25%;
  - over 5,000 at volatility 10%;
  - about 200 at volatility 50%.

  It is weakest far out of the money, where the two averages are less often in the money
  together: the variance falls by a factor of 60 at strike 120 and volatility 10%.
- **Two Monte Carlo engines agree.** For an at-the-money call with 52 weekly fixings, this
  engine gives 6.3596 ± 0.0005 and QuantLib's 6.3580 ± 0.0007.
- **QuantLib's Choi engine does not agree, though it has converged in its own
  parameters.** With `lambda = 60` and 150 levels it gives 6.33446. It is right to within
  noise at 4 and 7 fixings, but at 52 it is 0.40% low, about 48 standard errors. This is
  reproduced in the gallery; the cause has not been traced inside QuantLib.
- **Turnbull-Wakeman is 0.3% to 0.4% high** at every number of fixings.
