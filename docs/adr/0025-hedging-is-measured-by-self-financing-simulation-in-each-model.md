# 25. Hedging is measured by self-financing simulation, in any model's world with any model's hedge

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

The v1.0 hedging laboratory hard-wired one world to one set of hedges. Model risk shows up
when the world and the hedge come from different models: a local-volatility delta used in
a Heston world, or a Black-Scholes delta in either. Measuring that needs three things:

- accounting that does not know which model produced the paths;
- each model's value and Greeks on every path at every rebalancing date;
- references that check both.

## Decision

- **One self-financing account** (`volsurf.risk.hedging.hedge`):
  - the option is sold for a premium;
  - stock and any other hedging instruments are rebalanced on a schedule;
  - cash earns the rate and the stock pays its dividend yield;
  - the P&L is discounted to today path by path.
- **Greeks along paths come from grids solved once:**
  - **Heston:** prices on a (variance, log-moneyness) grid at each date, by a COS method
    vectorised over strikes, read through a bicubic spline.
  - **Local volatility:** a backward Crank-Nicolson solve (after Rannacher steps) in
    `ln(S/F_t)`, with Dupire variance from the Day 4 surface.
- **References for both:**
  - the Heston grid against the Day 5 contour integral and its finite-difference delta;
  - the local-volatility PDE against Black-Scholes on a flat surface;
  - the accounting against Derman and Kamal's law and the gamma formula for hedging at the
    wrong volatility.

## Consequences

- **The pricers match their references.**
  - Heston grid: prices to 3e-7 and deltas to 7e-6.
  - Local-volatility PDE: prices to 1.4e-5 and deltas to 3e-4.
  - Both are in `volsurf validate`.
- **Derman and Kamal are reproduced.** Delta hedging in Black-Scholes' world leaves 0.00217
  of spot at 252 rebalances, against their 0.00218.
- **The gamma formula holds path by path.** Hedging at 20% in a 25% world, it correlates
  0.996 with the hedged P&L.
- **A gamma hedge with a second option makes the error fall as `N^-0.67`, not `1/N`.** Near
  expiry, gamma changes within a rebalancing interval for paths close to the strike.
- **A Heston grid for three months of daily dates takes about a minute.** The gallery's
  Heston charts take minutes; the tests use coarse grids.
