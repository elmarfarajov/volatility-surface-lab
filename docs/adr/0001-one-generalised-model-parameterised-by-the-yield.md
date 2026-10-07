# 1. One generalised model, parameterised by the yield of the underlying

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

A stock option, an index option, a futures option and a currency option are priced by
one model, which differs only in what it costs to carry the underlying. Haug (2007)
writes this with the cost of carry `b`, so that the forward is `S exp(bT)`. The lab's
original code had two entry points:

- a Black-76 core in the forward;
- a spot formula with a dividend yield.

Nothing in the code named the carry convention a caller had in mind.

## Decision

`volsurf.analytic` prices every European option through one generalised
Black-Scholes-Merton model. The second parameter is the **yield** `q = r - b` of the
underlying rather than `b`: the dividend yield, the foreign rate, or the rate itself for
a future.

`Carry` builds the pair for each convention:

- `Carry.stock(r, q)`;
- `Carry.future(r)`;
- `Carry.currency(r_d, r_f)`;
- `Carry.margined_future()`.

## Consequences

- One set of formulas and one set of tests covers every underlying.
- The yield is the quantity a desk quotes (a dividend yield, a foreign deposit rate),
  so inputs are read off the market rather than converted.
- The rate sensitivities are stated for the yield held fixed (rho) and for the yield
  moving (psi). For a futures option the rate sensitivity is their sum, `-T V`.
