# 9. The Black function is computed through the Mills ratio, never as a difference of two prices

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

An out-of-the-money Black call is `F N(d1) - K N(d2)`. When the total volatility is small,
the two terms are nearly equal and their difference cancels. At `ln(K/F) = 0.6` and total
volatility 0.02, the literal formula keeps 9 of 16 digits. Further out it keeps fewer, and
the price has no meaning long before it underflows.

The implied-volatility inversion of [ADR 0010](0010-implied-volatility-is-inverted-on-log-price-objectives.md)
needs `ln b` and its derivatives far into the tails, down to prices of `1e-300` and
below. A price that cancels cannot be inverted.

## Decision

- Prices come from `b = envelope * [Y(h + t) - Y(h - t)]`, where `Y` is the Mills ratio,
  `h = x/s`, `t = s/2`, and the envelope is the normalised vega
  (Jaeckel, 2015, equation 2.4).
- `ln b` is computed directly, so it is available where `b` underflows.
- Where `t` is small against `|h|`, the difference of Mills ratios is computed as the
  integral of `Y'(z) = 1 + z Y(z)`, by 16-point Gauss-Legendre.
- Where `z < -4`, `Y'(z)` comes from Laplace's continued fraction, which has no
  subtraction. Elsewhere the two terms differ by a factor of about 1.4 or more.
- `black76_price`, which every original module uses, now calls this function.

## Consequences

- `ln b` agrees with 60-digit arithmetic to a few roundings of its own size,
  including in the deep tails. That covers values down to `ln b = -6.8e6`,
  far below any representable price.
- What remains is the rounding of `(x/s)^2 / 2` in the exponent. For a price of `1e-200`
  that is a relative error of about `1e-13`, the same as Jaeckel's reference
  implementation.
- Prices agree with QuantLib's `blackFormula` to 1e-11 on 400 random contracts, and the
  original test suite passes unchanged.
