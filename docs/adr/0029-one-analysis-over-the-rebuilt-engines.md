# 29. One analysis runs over the rebuilt engines, and the v1 pipeline is retired

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

`volsurf analyze` and the dashboard still ran the v1.0 pipeline after eight days of
rebuilding. Days 4 to 8 found that pipeline wrong in specific ways:

- its penalised SVI fit left arbitrage in 36 of 48 S&P 500 slices;
- its local volatility returned NaN on 4% of the grid;
- its Heston calibration was not the identifiable one;
- its hedging lab knew one world and one set of hedges.

The command line and dashboard are what a reader runs first, and they showed the oldest code.

## Decision

- **One analysis, `volsurf.platform.analysis.run`**, goes from quotes to a book's risk in the
  order the days were built:
  1. cleaning (Day 3);
  2. the constrained surface (Day 4);
  3. multistart Heston (Day 5);
  4. SABR and Dupire (Day 6);
  5. exotics under three models (Day 7);
  6. cross-model hedging and the dealer book (Day 8).
- **Every surface uses it:** `volsurf analyze` on a stored snapshot, `volsurf demo` on a
  synthetic chain, the report, and the dashboard.
- **`volsurf price`** quotes every pricing engine from the rebuilt packages: analytic,
  lattices, the PDE, Monte Carlo, Longstaff-Schwartz bounds, and Heston by contour
  integral, COS and QE.
- **The v1 pipeline, report, figures, Heston calibration and hedging lab are deleted.**
- **The v1 SVI and surface move to `volsurf.legacy`.** They are kept only because the Day 4
  chart and validation row that show their arbitrage need the original code.
- **The v1 engines that serve as independent references stay where they are:**
  `heston.py`, `black_scholes.py`, `monte_carlo.py`, `lattice.py`, `implied_vol.py`, and the
  synthetic-chain generator in `market_data.py`. The validation suite cross-checks the
  rebuilt code against them.

## Consequences

- **What a reader runs is what the notes describe.**
- **On a synthetic chain, the analysis recovers its Heston model** (rho -0.737 against
  -0.74, vol of vol 0.80) in under a minute in quick mode. That is now a validation row.
- **`volsurf analyze` no longer downloads.** It reads a snapshot stored by
  `volsurf chains fetch`. The old `--raw-csv` replay and `--save-raw` flags are gone, and
  stored snapshots replace them (ADR 0030).
