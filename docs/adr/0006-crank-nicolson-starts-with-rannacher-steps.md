# 6. Crank-Nicolson always starts with Rannacher's implicit half-steps

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

Crank-Nicolson is second order in time and the standard scheme for the Black-Scholes
PDE. It damps high-frequency modes only weakly, and a payoff's kink at the strike
excites exactly those modes. Prices at the grid's nodes away from the strike look fine.
The Greeks near the strike oscillate when the time step is large relative to the
spatial step squared, which is the regime a desk runs in. On a three-month at-the-money
call (800 x 25 grid), the gamma error near the strike is 0.85, against a gamma of 0.04.

## Decision

The theta-scheme replaces its first two Crank-Nicolson steps with four implicit Euler
half-steps (Rannacher, 1984) by default (`rannacher_steps=4`). Plain Crank-Nicolson stays
available for comparison.

## Consequences

- The gamma error on the same grid falls to 1.5e-5, and the scheme stays second order.
- American and Bermudan exercise are applied by projection at each exercise time. This
  converges at close to first order near the free boundary; Richardson extrapolation
  recovers accuracy when needed.
