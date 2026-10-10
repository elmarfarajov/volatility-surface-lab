# 10. Implied volatility is inverted on log-price objectives, and its error is measured against the conditioning

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The v1.0 inversion was Newton's method on the price, stopped when the price residual fell
below `1e-12` times the price plus `1e-15`. The absolute term ends the iteration
early once the price itself is small.

Measured against 60-digit prices, v1.0 was:

- accurate to 4e-12 above a normalised price of `1e-5`;
- wrong by up to 6e-8 down to `1e-10`;
- wrong by 18% by `1e-20`;
- wrong by a factor of four by `1e-300`.

"Accurate" also needs a definition. Near the price's upper bound, and for very deep quotes
in the money, one rounding of the price moves the volatility by far more than one
rounding. No inversion can do better than that conditioning.

## Decision

- Invert the normalised price `b(x, s)` for the total volatility, following Jaeckel's
  *Let's be rational* (2015):
  - below half the upper bound, on `ln b(s) - ln beta`;
  - above it, on `ln(b_max - b(s)) - ln(b_max - beta)`, a sum of two positive tails;
  - with third-order Householder steps whose derivatives are closed-form.
- Both objectives are concave, so a bracket tightens at every step. A step that leaves
  the bracket is replaced by bisection, geometric while the bracket spans orders of
  magnitude.
- Start from asymptotics rather than Jaeckel's rational-cubic guesses:
  - below: the small-`s` expansion, solved by bisection in `ln s`;
  - above: `b_max - b ~ 2 cosh(x/2) N(-s/2)`, exact at the money.
- Stop when the step or the objective is within eight roundings of its own evaluation.
- Measure accuracy as error divided by the condition number `eps * beta / (s b'(s))`.
- Quotes outside the static bounds return NaN. A quote within rounding of intrinsic value
  returns zero.

## Consequences

- On 1,816 well-posed quotes the worst error is 4.0 condition numbers. Jaeckel's
  reference implementation (`py_lets_be_rational`) gets 4.1 on the same quotes, and the
  two agree to a median of 2e-16. Tests pin the bound at 8.
- Two million random quotes converge in at most 7 iterations, with a mean of 4.7.
  Jaeckel's starting points take two. The difference costs speed, not accuracy, and is
  stated rather than hidden.
- Two stopping-rule faults were found by stress tests and fixed. One was a bracket that
  replaced a converged iterate with a bisection. The other was a one-ulp limit cycle,
  caught on a random sample of 200,000 quotes.
- `implied_vol_black76` and `implied_vol_bsm` keep their names and call the new
  inversion. Their unused `tol` and `max_iter` arguments are gone.
