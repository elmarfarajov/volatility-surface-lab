# 27. Real-data backtests state their proxies, are checked against theory, and keep their data local

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

The lab has one S&P 500 option-chain snapshot, not a history of option prices. Thirty-six
years of daily index, VIX, VIX3M and bill-rate closes are freely available. They can show
what hedged short-volatility positions earned, but only through proxies.

## Decision

- **The history is stored locally and never committed:**
  - `volsurf history fetch` downloads it from Yahoo Finance, up to the last complete day,
    into the git-ignored `data/history`;
  - charts that need it are skipped where it is absent, as snapshot charts are.
- **The proxies are named wherever results appear:**
  - **Premium.** The VIX is a variance-swap rate, above at-the-money volatility by the skew's
    value, so a straddle sold at the VIX overstates its premium.
  - **Marking.** Options are marked daily at the VIX as an at-the-money proxy.
  - **Dividends** are a constant 1.5% yield.
  - **Rates** are the 13-week bill's discount yield, converted to continuous compounding.
- **Theory checks the backtest on the real paths.** Each trade's P&L is compared with the
  gamma formula evaluated on the same path.

## Consequences

The backtest sells a 30-day at-the-money straddle every trading day since 1990 (9,210
trades):

- **85% of trades made money**, and the average P&L was 21% of the premium. The VIX
  exceeded the volatility later realised by 4.0 points on average.
- **The losses cluster in crises.** The worst was a straddle sold on 9 September 2008 at
  VIX 25: the index realised 66% and the trade lost 9.6% of the index.
- **The gamma formula explains the P&L.** It correlates 0.976 with the trades' P&L on real,
  fat-tailed paths.
- **The results are an upper bound on what was earned, and say so.** They cover a
  variance-swap-like position, not a quoted straddle.
