# The arbitrage-free surface: SVI under hard constraints, prices between expiries, and the distributions they imply

Day 3 turned the S&P 500 chain into clean implied volatilities. Day 4 turns those quotes
into a surface: a smile at every expiry, joined through time. The surface has to be
free of static arbitrage, which means two things:

- its density must never be negative (no butterfly arbitrage);
- no option may be worth less with more time to run (no calendar arbitrage).

Later days build on it. Heston on Day 5 is calibrated against it, local volatility on
Day 6 is its derivative, and the exotics of Day 7 are priced from its distributions.

The day's finding is that the original surface was not free of arbitrage: not in most of
its slices, and not between them.

---

## 1. Checking a smile on the whole real line

![A smile that fits and still admits arbitrage](../images/vogt-slice.png)

Raw SVI gives total implied variance as
`w(k) = a + b(rho(k - m) + sqrt((k - m)^2 + sigma^2))`. A slice admits no butterfly
arbitrage if and only if Durrleman's function

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2

is non-negative for every `k`. Axel Vogt's slice, the example in Gatheral and Jacquier
(2014), looks like any equity smile. Its `g` reaches -0.0329 at `k = 0.88`, and the check
finds exactly that.

A grid of strikes cannot settle "for every `k`". Two things can:

- **The wings are settled analytically.** As `k -> +-inf`, `g` tends to
  `1/4 - b^2 (1 +- rho)^2 / 16`, so they are safe if and only if the wing slopes obey
  Lee's bound, `b(1 +- rho) <= 2`.
- **The finite part is searched** on a grid uniform in `asinh((k - m)/sigma)`, which puts
  points where the curvature is, across 60 smile widths either side. Every local minimum
  is then refined by Brent's method.

The calendar check works the same way on the gap between two slices. It works: Vogt's
slice moved three units out sits entirely outside a fitting grid on [-1, 1], and the
check still finds its dip.

## 2. The original surface had arbitrage

![The original surface had arbitrage](../images/v1-arbitrage.png)

The v1.0 fitter added penalties for a negative density and for crossing the previous
slice, on 81 strikes. Checked on the whole line, on the 48 expiry dates of 9 October 2026:

- **36 of 48 slices** had a negative density somewhere;
- **31 of 47 pairs** of neighbouring slices crossed;
- **between expiries**, its interpolation reached `g = -3.92`. That interpolation is
  PCHIP in total variance after a running maximum, which is how it hid the crossings.

## 3. Hard constraints, solved by the exchange method

The conditions are now constraints
([ADR 0012](../adr/0012-no-arbitrage-conditions-are-constraints-checked-on-the-whole-line.md)).
"For every `k`" makes the fit a semi-infinite programme, solved by the exchange method:

1. SLSQP fits on a finite set of strikes;
2. the whole-line checks find the worst point;
3. that point joins the set, and the fit runs again.

The constraints carry margins of `1e-6` in `g` and `1e-7 * theta` in total variance,
because SLSQP meets an active constraint only to about 1e-9. Without them the exchange
loop converged to violations of `1e-9` and never stopped. All 48 slices converge, in at
most 9 rounds.

![Fitting the S&P 500 smile without arbitrage](../images/smile-fits.png)

![What the no-arbitrage conditions cost in fit](../images/fit-quality.png)

| Model | Median error | Quotes inside bid-ask | Arbitrage |
|---|---|---|---|
| SSVI, one surface | 2.58 vol pts | 6% | none, by theorem |
| eSSVI, slice by slice | 2.60 | 6% | none, but one pair crosses by 1e-8 |
| v1.0 SVI, penalties | 0.16 | 66% | 36 slices, 31 pairs |
| **SVI, hard constraints** | **0.19** | **68%** | **none** |

**SSVI and eSSVI** are arbitrage-free by conditions on their parameters, but their
three-parameter slice cannot follow the S&P 500 smile. Fitted to a single expiry with no
constraint at all, an SSVI slice misses by 2.6 to 3.1 volatility points, and eSSVI's
curvature condition binds on none of the 48 slices. It is the shape, not the conditions,
that costs. They serve as arbitrage-free starting points for the constrained SVI fit
([ADR 0014](../adr/0014-ssvi-and-essvi-are-kept-as-arbitrage-free-starts.md)).

## 4. Between expiries: interpolate prices

![Between expiries: interpolate prices, not variances](../images/time-interpolation.png)

Gatheral and Jacquier (2014) interpolate prices, not variances
([ADR 0013](../adr/0013-between-expiries-prices-are-interpolated.md)). With `theta` the
ATM total variance:

    c(k, t) = alpha_t c_i(k) + (1 - alpha_t) c_{i+1}(k),
    alpha_t = (sqrt(theta_{i+1}) - sqrt(theta_t)) / (sqrt(theta_{i+1}) - sqrt(theta_i)).

A convex combination of valid call-price functions is valid. The weight falls with time
while the prices rise, so calendar freedom holds too. Of 140 intermediate smiles, 24 of
v1.0's have a negative density; none of these do.

Three details matter:

- **Out-of-the-money prices are interpolated**, puts below the forward. Interpolating
  calls and inverting them deep in the money cost 8e-4 in volatility at `k = -0.8`.
- **Before the first expiry**, the lower slice is expiry itself, with `theta = 0`.
- **Beyond the last expiry**, the distribution is convolved with independent log-normal
  noise. The extended process stays a martingale growing in convex order, so it is free of
  arbitrage by construction. A 60-point Gauss-Hermite rule does this a hundred times
  faster than direct convolution, agrees with it to 1e-9, and joins the last slice to
  1e-10.

## 5. The distributions

![The distributions the S&P 500 options price](../images/risk-neutral-densities.png)

The density of `ln(S_T/F)` follows from `g` in closed form. Over all 48 slices:

| Check | Error |
|---|---|
| mass equals one | 2e-16 |
| mean equals the forward | 2e-16 |
| against Breeden and Litzenberger's second difference of call prices | 2e-6 of the peak |

One thing is visible on the log scale. Where the quotes alone would imply a negative
density, the constrained fit sits on its bound `g = 1e-6`, and the density touches zero.
That is the honest consequence of quotes that are slightly inconsistent in the wings.

## 6. What the wings say

![What the wings say about the moments of the index](../images/lee-wings.png)

Roger Lee's moment formula links a wing's slope to the highest finite moment of `S_T`.
The S&P 500 put wing is the steeper at every maturity, about three times the call wing a
week out, and every slope is far inside the bound of 2.

Beyond the last quoted strike the wings are SVI's extrapolation. These are therefore the
moments the fitted surface implies, not ones the quotes prove.

## Reproducing

```bash
volsurf chains fetch ^SPX            # if no snapshot is stored
volsurf gallery --only 4             # this note's charts
pytest tests/smile                   # parameterisations, checks, fits, densities, interpolation
volsurf validate                     # five surface rows, on a Heston chain, so they run anywhere
```
