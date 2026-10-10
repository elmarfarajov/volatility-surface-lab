# 11. Market data is snapshotted locally, and every quote is kept or dropped for a stated reason

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

A chart drawn from a live feed cannot be reproduced the next day. Raw option chains also
contain quotes that are not prices:

- quotes with no bid, or crossed;
- auto-quotes nobody holds;
- stale quotes left over from another day.

In the SPX chain of 9 October 2026, a put last traded in June was quoted 49 points away
from its neighbours. A filter that removes "outliers" without saying which, or why,
cannot be checked.

Yahoo Finance's terms do not allow its quotes to be redistributed.

## Decision

- `volsurf chains fetch` stores every listed expiry as one compressed file per ticker and
  trading day under `data/chains/`, with the spot and the fetch time.
- `data/` is ignored by git. Charts drawn from a snapshot are committed as images, and
  the gallery skips them where no snapshot exists, as in CI.
- Each expiry and contract root is cleaned by named rules, in order. Each rule records
  how many quotes it removed, and the counts add up to the raw chain.
- Strike arbitrage is tested at the bid and the ask, not the mid, so only a pattern that
  could be traded is arbitrage. Quotes are removed one at a time, each round the one
  whose removal leaves the least arbitrage.
- Forwards and discount factors come from put-call parity:
  - a Siegel repeated-median start;
  - then weighted least squares, trimmed at three bid-ask half-widths;
  - and one Nelson-Siegel curve across expiries.
- American chains fit parity only near the money, with discount factors from a European
  market's curve.

## Consequences

- The cleaning funnel and the largest arbitrage it removed are charts, not claims.
- The first removal rule blamed the quote involved in the most violations. A stale quote
  and its honest neighbour share every butterfly, so that rule ate whole runs of good
  quotes: 364 of one expiry's quotes. Trying each removal fixed it.
  - That expiry now loses 25 quotes.
  - In the worst remaining expiry, every removed quote sits at least 51 half-spreads off
    its neighbours' line.
- Parity has a stated limit. A coordinated block of stale pairs within about four
  half-widths of the line is admitted, and moves the forward. Tests pin both sides of
  that boundary.
- The quote-width standard errors of each expiry's rate understate the scatter around the
  curve by about three times (median |z| of 2.1). The likely cause is non-synchronous
  quotes, and it is reported.
