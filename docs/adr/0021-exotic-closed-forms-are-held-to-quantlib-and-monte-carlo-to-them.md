# 21. Exotic closed forms are held to QuantLib, and Monte Carlo is held to the closed forms

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Barriers, geometric Asians, lookbacks and digitals have closed forms under Black-Scholes.
They are long, sign-sensitive and easy to mistype: the Reiner-Rubinstein barrier alone is
sixteen cases built from six terms. Most exotics met in practice have no closed form, so
the Monte Carlo that prices them must first be trusted where an exact answer exists.

## Decision

- **Closed forms** (`volsurf.exotics.closed_form`) follow Haug (2007) in the cost-of-carry
  form of Day 1:
  - Reiner-Rubinstein barriers, with rebates;
  - Kemna-Vorst geometric Asians, continuous and discrete;
  - Goldman-Sosin-Gatto floating and Conze-Viswanathan fixed lookbacks;
  - cash- and asset-or-nothing digitals.

  A barrier already reached is refused, not priced.
- **Each is compared with QuantLib** on random contracts, in the tests and in
  `volsurf validate`: 400 barriers of four kinds with and without rebate, 200 discrete
  geometric Asians and 200 lookbacks.
- **Monte Carlo is held to the closed forms** before it is used where none exists. The
  Brownian-bridge estimator is checked against the continuous barrier price, and the
  discretely monitored price against the Broadie-Glasserman-Kou corrected one.
- **Parities are tested exactly:**
  - knock-in plus knock-out is the vanilla;
  - the asset digital less strike times the cash digital is the call.

## Consequences

- **The closed forms agree with QuantLib** to 5e-14 for barriers, 1e-13 for geometric
  Asians, 1e-12 for lookbacks and 1e-13 for digitals.
- **The bridge recovers the continuous price** from 12 monitoring dates within the Monte
  Carlo error. A continuously monitored barrier can therefore be priced on a coarse grid.
