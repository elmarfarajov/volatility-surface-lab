# Exotic options: closed forms, Monte Carlo with corrections, and the price under each model of the surface

Days 4 to 6 built one S&P 500 surface and three models that fit it: Black-Scholes at one
volatility, Heston, and Dupire local volatility. Day 7 prices options whose payoff depends
on more than the final index level. These are barriers, Asians, lookbacks and digitals.

The work is in three steps:

1. closed forms where they exist, held to QuantLib;
2. Monte Carlo where they do not, held to the closed forms first;
3. the same exotic under each model of the surface.

The third step is the point. Models that agree on every vanilla need not agree on an
exotic, and the difference is model risk that no calibration to vanillas removes.

---

## 1. Closed forms, checked against QuantLib

The closed forms are in `volsurf.exotics.closed_form`, written in the cost-of-carry form
of Day 1 so that one function covers stocks, indices, futures and currencies:

- **Barriers:** Reiner and Rubinstein's sixteen cases (Haug's terms A to F), with rebates.
  A barrier already reached raises an error rather than returning a price.
- **Geometric Asians:** Kemna and Vorst's, continuous and on `n` equally spaced fixings.
- **Lookbacks:** Goldman, Sosin and Gatto's floating strike; Conze and Viswanathan's fixed
  strike.
- **Digitals:** cash-or-nothing and asset-or-nothing.

| Family | Contracts | Largest difference from QuantLib |
|---|---|---|
| barriers, four kinds, with and without rebate | 400 random | 5e-14 |
| geometric Asians, continuous and discrete | 300 random | 1e-13 |
| floating and fixed lookbacks | 300 random | 1e-12 |
| cash and asset digitals | 300 random | 1e-13 |

Two parities are tested to rounding. Knock-in plus knock-out is the vanilla, and the asset
digital less strike times the cash digital is the call
([ADR 0021](../adr/0021-exotic-closed-forms-are-held-to-quantlib-and-monte-carlo-to-them.md)).

## 2. Barriers: watched continuously or once a day

![Barriers watched continuously or once a day](../images/barrier-monitoring.png)

The closed form assumes the barrier is watched continuously, but contracts watch it at
fixings. Checked once a day, a barrier is crossed less often, so a knock-out is worth more.

**Two tools, used for different questions:**

- **The Brownian bridge.** Between two fixings a log-normal path crosses `H` with
  probability `exp(-2 ln(S_i/H) ln(S_{i+1}/H) / (sigma^2 dt))`. Weighting each path by its
  probability of surviving every interval estimates the *continuous* price from a coarse
  grid. With 12 dates it is within noise of the closed form.
- **Broadie, Glasserman and Kou's correction.** This estimates the *discrete* price as the
  continuous one with the barrier moved away from spot by `exp(0.5826 sigma sqrt(dt))`.

**What the chart shows,** for a call struck at 100 (one year, volatility 25%):

- **Daily monitoring already matters.** The up-and-out with barrier 120 is worth 0.673
  continuously and 0.796 ± 0.006 monitored daily: 18% more.
- **The correction is accurate where the barrier is far from the payoff.** For the
  down-and-out with barrier 85 it is within noise at every grid, even quarterly.
- **It breaks for coarse monitoring near the payoff.** For the up-and-out it overprices by
  2% weekly, 9% monthly and 22% quarterly.

Discrete contracts are therefore priced on their own grid. The correction is used only
where it has been checked
([ADR 0022](../adr/0022-discrete-monitoring-is-priced-on-its-own-grid.md)).

## 3. Arithmetic Asians, and a reference that is not

![Arithmetic Asians: a control variate, and a reference that is not](../images/asian-control-variate.png)

The arithmetic average has no closed form, but the geometric average of the same path
does, and the two move almost one-for-one. Used as a control variate, the geometric price
divides the Monte Carlo variance by:

| Volatility | Strike 80 | Strike 100 | Strike 120 |
|---|---|---|---|
| 10% | 5,860 | 5,242 | 59 |
| 25% | 1,118 | 894 | 210 |
| 50% | 270 | 218 | 131 |

These are for 52 weekly fixings over one year. The control is weakest far out of the
money at low volatility, where the two averages are seldom in the money together.

**Choosing the reference.** With the control, this engine prices an at-the-money call with
52 fixings at 6.3596 ± 0.0005. QuantLib's own Monte Carlo engine, with its own control
variate, gives 6.3580 ± 0.0007: two independent engines agree.

QuantLib's semi-analytic Choi engine, converged in its own parameters, does not agree. It
gives 6.33446:

- within noise at 4 and 7 fixings;
- 0.40% low at 52 fixings, about 48 standard errors.

Turnbull and Wakeman's approximation is 0.3% to 0.4% high throughout. The reference is
therefore the controlled Monte Carlo, and analytic engines are measured against it
([ADR 0023](../adr/0023-the-arithmetic-asian-reference-is-controlled-monte-carlo.md)).

## 4. Lookbacks: the extreme seen once a day

![Lookbacks: the extreme seen once a day](../images/lookback-monitoring.png)

A floating-strike lookback call pays the final price less the lowest price seen. Watched
once a day, the lowest is missed by about `0.58 sigma sqrt(dt)`, so the discrete option is
worth less than the continuous one. The gap closes only as `sqrt(dt)`:

| Monitoring dates | Discrete price | Gap to the closed form (19.30) |
|---|---|---|
| 12 | 16.22 | 3.08 |
| 52 | 17.76 | 1.54 |
| 252 | 18.64 | 0.66 |

A lookback monitored daily is 3% below its continuous closed form, which is more than
any bid-ask spread on it.

## 5. A digital is a call spread, and the smile prices it

![A digital is a call spread, and the smile prices it](../images/digital-smile.png)

A digital call is minus the derivative of the call price in strike. On a smile that
derivative has two terms: the Black-Scholes digital at the strike's own volatility, and a
skew term, `-vega x d sigma / dK`.

On the S&P 500 at one year the skew falls with strike, so the skew term is positive:

- **8 cents per dollar paid** at 105% of the forward;
- **11 cents** near 95%.

Black-Scholes at one volatility, whether at-the-money or the strike's own, misses that
term entirely.

## 6. One surface, three models, three prices

![One surface, three models, three prices](../images/exotic-model-risk.png)

The same seven-contract book is priced on 100,000 paths under each model, monitored daily.
Black-Scholes uses the one-year at-the-money volatility; Heston has the Day 5 calibration;
local volatility comes from the monthly Day 4 surface. Prices are per unit of spot.

| Contract | Black-Scholes | Heston | Local volatility |
|---|---|---|---|
| at-the-money call | 0.0848 | 0.0896 | 0.0898 |
| 90% put | 0.0150 | 0.0278 | 0.0266 |
| digital call at 105% | 0.426 | 0.538 | 0.527 |
| down-and-out call, barrier 80% | 0.0847 | 0.0859 | 0.0872 |
| up-and-out call, barrier 120% | 0.0196 | 0.0459 | 0.0341 |
| one-touch at 85% | 0.236 | 0.260 | 0.272 |
| Asian call, monthly fixings | 0.0499 | 0.0489 | 0.0489 |

The surface itself prices the at-the-money call at 0.0893 and the 90% put at 0.0269.

- **The two smile models agree on the vanillas.** Local volatility reprices them by
  construction; Heston is within its 2.5-point calibration error.
- **They part on the barrier.** The up-and-out call is worth 35% more under Heston than
  under local volatility, and the one-touch 4% less. Vanillas fix the distribution at each
  expiry; the barrier depends on the path between them, where the models differ.
- **Black-Scholes misses the skew.** It is 44% low on the 90% put and 19% low on the
  digital.
- **The Asian hardly moves.** An average depends mainly on the at-the-money variance,
  which every model matches.

This spread is a model risk that more calibration to vanillas cannot remove
([ADR 0024](../adr/0024-exotics-are-priced-under-every-model-of-the-surface.md)). Day 8
turns to hedging, where the same disagreement appears as a delta.

## Validation

Five rows in `volsurf validate`:

- barriers against QuantLib;
- geometric Asians and lookbacks against QuantLib;
- the Brownian bridge against the continuous closed form;
- the BGK correction against daily-monitored Monte Carlo;
- the arithmetic Asian against QuantLib's Monte Carlo.

## References

- Broadie, M., Glasserman, P. and Kou, S. (1997). A continuity correction for discrete
  barrier options. *Mathematical Finance* 7(4).
- Conze, A. and Viswanathan (1991). Path dependent options: the case of lookback options.
  *Journal of Finance* 46(5).
- Goldman, M. B., Sosin, H. B. and Gatto, M. A. (1979). Path dependent options: buy at the
  low, sell at the high. *Journal of Finance* 34(5).
- Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed. McGraw-Hill.
- Kemna, A. G. Z. and Vorst, A. C. F. (1990). A pricing method for options based on average
  asset values. *Journal of Banking and Finance* 14(1).
- Reiner, E. and Rubinstein, M. (1991). Breaking down the barriers. *Risk* 4(8).
- Turnbull, S. M. and Wakeman, L. M. (1991). A quick algorithm for pricing European average
  options. *Journal of Financial and Quantitative Analysis* 26(3).
