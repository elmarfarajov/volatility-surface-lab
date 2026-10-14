# 28. A book's risk is full revaluation under stated smile dynamics; the Greeks explain, they do not measure

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

An options book's risk report has two jobs:

- to explain each day's P&L by Greeks;
- to measure what large moves would cost.

The second-order Taylor expansion serves the first. Whether it serves the second is
measured here rather than assumed. A stress on the index alone also needs a rule for how
the smile moves with it, and the rules disagree.

## Decision

- **Revaluation is full.** Every option is repriced off the Day 4 surface at
  `sigma(k' + c s, T) + d sigma(T)`, where `c` is the smile dynamic:
  - 0 is sticky moneyness;
  - 1 is sticky strike;
  - 2 is local volatility;
  - 1.8 is the market's (ADR 0026).
- **Historical scenarios** apply every daily and ten-day move since 1990 to today's book.
  The VIX and VIX3M moves are applied as at-the-money moves by expiry, and longer
  expiries' moves decay as `(91 days / T)^0.42`. The 0.42 comes from the regression of
  VIX3M moves on VIX moves.
- **The explain has eight terms:** delta, gamma, vega, vanna, volga, theta, and the time
  decay of delta and of vega (charm and veta). Over a day this book's charm is larger
  than its gamma, because short-dated options away from the money dominate it.

## Consequences

The book is a dealer's S&P 500 position:

- short 200 one-month 95% puts, 150 three-month 90% puts and 100 one-year 80% puts;
- long 150 three-month 105% calls and 100 six-month 110% calls;
- delta-hedged with the index.

It is nearly vega-neutral and short skew.

- **The smile dynamic decides the stress.** For a 10% fall:

  | Smile dynamic | Loss |
  |---|---|
  | sticky moneyness | $2.0m |
  | sticky strike | $4.4m |
  | the market's | $6.9m |
  | local volatility | $7.6m |

- **The Greeks explain one-day moves of about 1%.** On random options and moves, the
  median unexplained P&L is 0.03% of the first-order P&L.
- **They misstate crashes, in both directions:**
  - **2008:** the Greeks understate the losses. On 15 October 2008 they show $2.3m against
    $4.0m revalued.
  - **2020:** they overstate them. On 16 March 2020 they show $12.6m against $10.1m.
  - **Value at risk:** the one-day 99% VaR is $456k revalued and $548k from Greeks. The
    ten-day 97.5% expected shortfall is $4.8m and $5.9m.
