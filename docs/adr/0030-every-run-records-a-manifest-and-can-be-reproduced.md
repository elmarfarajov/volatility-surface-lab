# 30. Every run records a manifest, and `volsurf reproduce` reruns it and compares every number

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

A number in a report is only as good as the ability to get it again. Live option chains
change by the second, Monte Carlo depends on its seeds, and library versions move.
Reproducing a result needs four things:

- the exact input;
- the configuration;
- the code;
- a way to check the rerun against the original.

## Decision

- **Inputs are fingerprinted.**
  - **Stored snapshots** are identified by ticker, day, and the SHA-256 of their files.
  - **Synthetic chains** are identified by model and seed, and the SHA-256 of the
    generated quotes.
  - **The local S&P 500 and VIX history** is fingerprinted whenever the book's VaR uses it.
- **`manifest.json` sits next to every report.** It holds:
  - the source and history fingerprints, and the configuration;
  - the volsurf version and git commit;
  - the Python and library versions;
  - the timings;
  - every headline number, flat and named, in `Analysis.results`.
- **`volsurf reproduce <dir>` reruns the recorded analysis and compares every number.**
  - It rebuilds the source and refuses to go on if the bytes differ.
  - It reruns with the recorded configuration.
  - It compares each number to a relative 1e-6, and exits non-zero if any differs.
- **Every random step is seeded:** Heston's Sobol' starts, the exotic and hedging paths.

## Consequences

- **A quick synthetic run reproduces all of its numbers**, 49 of 49 with hedging and 39 of
  39 without. This is a row of `volsurf validate`, and CI runs it again from the installed
  wheel (ADR 0031).
- **A changed input is refused with the two fingerprints**, rather than compared and
  misreported as a numerical difference.
- **The tolerance allows for floating-point libraries that differ between platforms in the
  last digits, and nothing more.** A rerun on another machine that disagrees beyond it is
  a finding, not noise.
