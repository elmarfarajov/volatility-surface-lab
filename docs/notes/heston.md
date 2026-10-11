# Heston: priced to the last digit, calibrated to the S&P 500, simulated across the Feller boundary

Heston (1993) is the stochastic-volatility model every desk knows. Variance mean-reverts and
is correlated with the stock, and European options have a semi-closed form through the
characteristic function. Day 5 rebuilds three things:

- **pricing**, so that it is right far into the wings;
- **calibration**, so that it says what the surface can and cannot pin down;
- **simulation**, so that it is right where the variance touches zero.

Every result is held to references the code does not share: Lewis's integral evaluated
to 40 digits, QuantLib, the exact variance transition, and published values. One of those
published values turned out to be wrong.

---

## 1. Pricing far into the wings

![Heston prices to the last digit](../images/heston-engines.png)

The v1.0 engines price one side and take the other from put-call parity. Far out of the
money that subtracts two numbers near the forward. Against the 40-digit reference:

| Engine | One year, far call | Ten years, deepest call |
|---|---|---|
| v1.0 COS | 9e-4 | 100% |
| v1.0 Gil-Pelaez | 6e-8 | 110% |
| Carr-Madan FFT (interpolated) | 2e-2 | a factor of ten |
| QuantLib analytic | 7e-9 | 3e-7 |
| **contour, out of the money** | **4e-15** | **1e-14** |

The new engine prices each side directly
([ADR 0015](../adr/0015-heston-prices-out-of-the-money-by-contour-integral.md)). It uses
Carr and Madan's contour integral as extended by Lee (2004): a damping `alpha > 0` gives the
call, `alpha < -1` the put, and the residues the contour crosses are exactly put-call
parity.

![Where the contour runs](../images/contour-damping.png)

Getting it right took four corrections, and each one is now a test:

1. **Truncation.** The integral was cut off when the oscillating real part fell below a
   tolerance at a single point. That point can sit near a zero. The integral now stops on
   the integrand's modulus.
2. **Graded panels.** Near a pole the integrand peaks sharply at the origin: the
   denominator's pole as `alpha` nears -1, or `phi`'s at the edge of the moment strip.
   Uniform panels missed the peak by 3.7e-5 at `alpha = -1.5`. Panels graded geometrically
   towards the origin take every damping inside the strip to 1e-15.
3. **The damping.** Lord and Kahl's choice was pulled to the very edge of the strip by
   `e^{-alpha k}`, where even graded panels fail (95% error). The outer 10% of the
   interval is now off-limits.
4. **The bisection.** An eight-way bisection could step back past its own bracket.

**A published value is wrong.** Fang and Oosterlee's (2008) reference price for their Heston
test case, used by most implementations, is printed as 5.785155450. Lewis's integral at 40
digits gives **5.78515543437619**: the eighth significant digit is off by 1.6e-8. The
contour engine matches the 40-digit value to 4e-15.

## 2. The moment strip

![The moment strip and the wings it implies](../images/moment-explosion.png)

`E[S_T^p]` is finite only for `p` in a strip around [0, 1] that narrows with maturity. The
contour must stay inside it, and through Lee's moment formula its edges fix the slopes of
the implied-variance wings.

The edges come from Andersen and Piterbarg's explosion time, derived here from the
Riccati equation
([ADR 0016](../adr/0016-the-moment-strip-comes-from-explosion-times.md)). Two shortcuts
failed on the way:

- **Testing where `phi(-ip)` stops being real.** Beyond the explosion, the formula's
  continuation turns real again, and the edge jumped between 35 and 76 from one maturity to
  the next.
- **The explosion time as first recalled.** It had the wrong limit at `D = 0`.

## 3. Calibrated to the S&P 500

![Heston calibrated to the S&P 500](../images/heston-calibration.png)

The calibration fits 13 expiries from a month to two years, with 15 quotes each, weighted
by bid-ask width. Residuals are in implied volatility, through the Day 3 inversion:

    v0 = 0.0130, kappa = 4.42, theta = 0.0530, sigma = 1.98, rho = -0.59,
    Feller ratio 0.12, error 2.54 vol pts (3.5 bid-ask widths).

Heston cannot carry the one-month skew and the two-year one at once. Vol-of-vol is pushed
to almost 2 to steepen the short end, which is the well-known limit of the model on index
options. The error is 0.7 points a year out and 4.7 points at six weeks, mostly in the put
wing.

![What the S&P 500 surface pins down](../images/heston-identifiability.png)

**Identifiability** ([ADR 0017](../adr/0017-heston-calibration-reports-its-identifiability-and-simulation-uses-qe.md)).

- **Multistart:** every start of a scrambled Sobol' design reached the same minimum, to
  1e-4 in `kappa`.
- **The Jacobian:** in log-parameters its singular values run from 437 to 28, a condition
  number of 16. The weakest direction is `-0.81 ln kappa + 0.35 ln v0 + 0.21 ln theta -
  0.41 ln sigma`: slower mean reversion, compensated by a higher level and lower vol-of-vol.
- **The profile:** between `kappa` 2.5 and 5 the error changes by 6% while `theta` falls
  from 0.064 to 0.051. Further out, the fit degrades: 5.1 widths at `kappa = 0.3`.

So the surface determines the parameters, but `kappa` only to about a factor of two.

## 4. Simulation across the Feller boundary

![Simulating Heston across the Feller boundary](../images/simulation-bias.png)

Against the contour price of the at-the-money option, with 200,000 paths:

| Scheme | Feller holds (2.67) | Feller fails (0.08) |
|---|---|---|
| Euler, full truncation | within noise from 16 steps a year | 113 se at 2 steps, 11 se at 64 |
| Andersen's QE | within noise from 2 steps | within noise from 4 steps |

![QE against the exact variance transition](../images/qe-transition.png)

QE matches the first two moments of the exact non-central chi-squared transition. For one
large step near zero it puts 81% of the mass exactly at zero. The exact law has no atom,
but more than half its mass lies within 1e-4 of zero, and QE's mean agrees to 1e-5. The log-price
step's martingale correction keeps `E[S_T] = F` to Monte Carlo noise at any step.

## Reproducing

```bash
volsurf gallery --only 5          # the calibration charts need a snapshot (volsurf chains fetch ^SPX)
pytest tests/stochvol             # against 40 digits, QuantLib, and the exact transition
volsurf validate                  # six Heston rows, none needing market data
```
