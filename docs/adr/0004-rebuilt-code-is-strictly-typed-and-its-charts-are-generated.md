# 4. Rebuilt code is strictly typed, and its charts are generated

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

The lab is rebuilt one area a day. The original modules were not type-checked, and its
figures were produced by the analysis pipeline for one market snapshot.

## Decision

- Every package rebuilt during the nine days is checked with `mypy --strict`. The list
  in `pyproject.toml` grows day by day: `volsurf.analytic` and `volsurf.gallery` on
  Day 1. SciPy, mpmath and QuantLib are treated as untyped, because SciPy's stubs package
  breaks mypy on Python 3.10.
- Every documentation chart is a `gallery.Item`, which records the file it writes, what
  it shows, the function that draws it and its day. `volsurf gallery` redraws them and
  generates `docs/GALLERY.md`, so the page and the images cannot drift.

## Consequences

- Type errors in the rebuilt code fail CI.
- The figures of the live analysis (`volsurf analyze`) stay with the pipeline. The gallery
  holds the charts that explain and validate each day's work.
