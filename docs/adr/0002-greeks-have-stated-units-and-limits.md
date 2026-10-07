# 2. Every Greek has stated units, a stated time convention, and a limit

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

Desks differ on Greek conventions. Vega can be quoted per unit or per vol point; theta
per year or per day; and the time Greeks with respect to time to expiry or to calendar
time passing. Common references print veta and color in time to expiry, while
practitioners read theta in time passing. A library that leaves this implicit invites
sign errors in hedging and P&L explain.

At expiry and at zero volatility the formulas divide by zero. The lab's original Greeks
returned infinities there.

## Decision

- **Units:** vega-type Greeks are per unit of volatility, and rho-type Greeks per unit
  of rate.
- **Time:** every time Greek (theta, charm, veta, color) is the derivative in calendar
  time passing, per year. `Greeks.per_day` converts.
- **Each Greek is defined by the derivative it is.** `analytic.reference.DEFINITIONS`
  records the variables and the order, and the tests compare the analytic Greek with
  that derivative computed to 50 digits.
- **At zero total volatility each Greek is its limit:**
  - the price is the discounted intrinsic value on the forward;
  - delta is a step;
  - the curvature Greeks are zero;
  - theta is the decay of the discounted intrinsic value.

  Exactly at the money, where the limits do not exist, delta is the average of its two
  sides and the curvature Greeks are NaN.

## Consequences

- The reference caught veta and color with their signs turned, as they are usually
  printed. They are now correct in the stated convention.
- A NaN at expiry at the money is deliberate. Code that hedges an option on its expiry
  date must decide what to do there, rather than receive a number with no meaning.
