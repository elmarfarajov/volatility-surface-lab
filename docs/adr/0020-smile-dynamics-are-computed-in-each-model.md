# 20. Smile dynamics are computed inside each model, and the delta follows

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Local volatility and SABR can be fitted to the same smile today and still disagree about
the smile after the index moves. A hedge ratio depends on that disagreement. Hagan, Kumar,
Lesniewski and Woodward (2002) argued that local volatility moves the at-the-money
volatility against the index by about twice the implied skew. That is a claim to measure,
not to quote.

## Decision

- **Local volatility after a move** is computed in the model itself. Local variance is held
  fixed in absolute strike, `v'(k', t) = v(k' + ln(F'/F), t)`, and the forward Dupire
  equation is solved again for the new forward.
- **SABR after a move** keeps its parameters; with `beta = 1`, only the forward changes.
- **Four deltas of the at-the-money call** are reported side by side:
  - Black's, which is sticky strike;
  - local volatility's own finite difference;
  - SABR's with `alpha` held fixed;
  - Bartlett's (2006), with `alpha` moving by its regression on the forward,
    `rho nu / F^beta` per unit.

## Consequences

- **On a Heston surface**, the at-the-money volatility under local volatility moves by
  1.98 times the implied skew per log-move. That is Hagan's factor of two, measured, and
  the validation suite holds it between 1.5 and 2.8.
- **On the S&P 500's three-month expiry** the factor is 2.2: the at-the-money volatility
  falls from 22% to 12% as the index goes from 5% down to 5% up. SABR with `beta = 1`
  barely moves it.
- **The deltas of the at-the-money call** are 0.514 for Black, 0.388 for local
  volatility, 0.632 for SABR with `alpha` held, and 0.398 for Bartlett. Local volatility
  and Bartlett nearly agree, and both sit well below Black.
