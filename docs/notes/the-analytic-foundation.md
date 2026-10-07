# The analytic foundation: one model, seventeen Greeks, three references

Everything else in the lab rests on the Black-Scholes formula. The surface is quoted in
its implied volatility, calibration weights quotes by its vega, and the hedging
laboratory rebalances with its delta. Day 1 makes that foundation something a desk would
sign off on. It is one model for every underlying, with every Greek a desk uses, each
checked against references that share no code with it.

---

## 1. One model for every underlying

![One model, five underlyings](../images/carry-conventions.png)

A stock option, an index option, a futures option and a currency option are not
different models. They differ in what it costs to carry the underlying until expiry.
Holding a stock earns its dividends, holding a currency earns the foreign interest rate,
and a future costs nothing to hold. Generalised Black-Scholes-Merton (Haug, 2007)
captures this with one parameter. `volsurf.analytic` uses the yield `q` of the
underlying, which is the number a desk quotes:

| Underlying | Model | Yield q | `Carry` |
|---|---|---|---|
| stock without dividends | Black-Scholes (1973) | 0 | `Carry.stock(r)` |
| stock or index | Merton (1973) | dividend yield | `Carry.stock(r, q)` |
| future | Black (1976) | r | `Carry.future(r)` |
| currency | Garman-Kohlhagen (1983) | foreign rate | `Carry.currency(r_d, r_f)` |
| margined future | Asay (1982) | r = q = 0 | `Carry.margined_future()` |

With `phi = +1` for a call and `-1` for a put, the price is

    V = phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)],
    d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt T),   d2 = d1 - sigma sqrt T.

## 2. Seventeen Greeks, and what each is a derivative of

![Sixteen Greeks of a European call](../images/greek-atlas.png)

| Order | Greek | Derivative |
|---|---|---|
| first | delta, vega, theta, rho, psi, dual delta | dV/dS, dV/dsigma, dV/dt, dV/dr, dV/dq, dV/dK |
| second | gamma, vanna, volga, charm, veta, dual gamma | d2V/dS2, d2V/dS dsigma, d2V/dsigma2, d delta/dt, d vega/dt, d2V/dK2 |
| third | speed, zomma, color, ultima | d gamma/dS, d gamma/dsigma, d gamma/dt, d volga/dsigma |

**The conventions are stated once, because desks differ** (ADR 0002):

- vega-type Greeks are per unit of volatility, and rho-type per unit of rate;
- every time Greek is the derivative in **calendar time passing**, per year;
- rho holds the yield fixed, and psi moves it. For a futures option, whose forward does
  not depend on the rate, the rate sensitivity is `rho + psi = -T V`.

Dual gamma, `d2V/dK2`, deserves a word. Discounted, it is the risk-neutral density of
the underlying at the strike (Breeden and Litzenberger, 1978). Day 4 recovers that
density from the market's surface.

**At expiry the formulas divide by zero.** The lab's original `bsm_greeks` returned
infinities there. Each Greek is now its limit:

- the price is the discounted intrinsic value on the forward;
- delta is a step;
- the curvature Greeks are zero;
- theta is the decay of the discounted intrinsic value. For an in-the-money put this is
  `r K`, positive, because the strike it will receive is discounted less as time passes.

Exactly at the money no limit exists, and the curvature Greeks are NaN.

![The Greeks at expiry](../images/expiry-limits.png)

## 3. Three references that share no code with the engine

![Every Greek against an independent reference](../images/greek-references.png)

A test written from the same formula as the code checks the transcription, not the
formula. Every result here is held to references that share nothing with it but the
definition of the model (ADR 0003):

**1. Arbitrary precision.** `analytic.reference` evaluates the price in 50-digit
arithmetic with mpmath and differentiates it numerically in that precision, to third
order. The reference knows nothing of the analytic Greeks. Across random contracts every
Greek agrees to 1e-14 relative or better.

The reference found two errors on its first run. **Veta and color came out with their
signs turned.** As they are usually printed, they are derivatives in time *to expiry*;
in the convention stated here, time *passes*. A finite-difference test written from the
same printed formulas would have agreed with them.

**2. An independent library.** On 500 random contracts, QuantLib's
`AnalyticEuropeanEngine` agrees with the engine to 1e-10. The contracts range over:

- maturities from two days to five years;
- negative rates;
- volatilities from 3% to 120%.

The comparison covers price, delta, gamma, vega, theta, rho and the dividend rho.

**3. Published worked examples.** Haug's *Complete Guide to Option Pricing Formulas*
(2007) works each carry convention and each first-order Greek through a numerical
example. All nine used here reproduce to the fourth decimal as printed:

| Example | Printed | Engine |
|---|---|---|
| Black-Scholes call | 2.1334 | 2.1334 |
| Merton put on an index paying 5% | 4.0870 | 4.0870 |
| Black-76 call on a future | 1.7011 | 1.7011 |
| Garman-Kohlhagen call on a currency | 0.0291 | 0.0291 |
| delta of a futures call / put | 0.5946 / -0.3566 | 0.5946 / -0.3566 |
| gamma | 0.0278 | 0.0278 |
| theta of an index put, per year | -31.1924 | -31.1924 |
| rho of a call | 38.7325 | 38.7325 |

A vega example recalled from the same book did not reproduce. Since its inputs could
not be confirmed, it was left out rather than adjusted.

## 4. What must hold for any contract

![Every price solves the Black-Scholes equation](../images/pde-residual.png)

Hypothesis generates contracts across:

- spots from 5 to 500 and strikes 1.5 log-units either side;
- maturities from four days to ten years;
- rates from -3% to 15% and volatilities from 2% to 200%.

On each it checks what no correct engine can break:

- **the Black-Scholes PDE**,
  `dV/dt + (r - q) S dV/dS + sigma^2 S^2/2 d2V/dS2 - r V = 0`, from the analytic
  Greeks, to 1e-10 relative to its largest term (1e-15 on the grid above);
- **put-call parity in price and in every Greek**: calls and puts differ by a forward
  contract, so their curvature Greeks are equal. Their charms differ by `q e^{-qT}`,
  the decay of the forward's delta;
- **homogeneity**: `V = S dV/dS + K dV/dK`, and doubling spot and strike doubles the
  price;
- **Merton's model-free bounds**: a call lies between `max(S e^{-qT} - K e^{-rT}, 0)`
  and `S e^{-qT}`;
- **the strike conditions**: call prices fall with the strike, by no more than the
  discount factor, and are convex. Day 4's surface must meet these too, and
  `analytic.strike_violations` will check market quotes against them.

## 5. Prices far in the wings

![Prices far in the wings](../images/tail-precision.png)

An out-of-the-money option far in the wings is the difference of two tiny numbers. The
original documentation said such prices were "accurate to machine precision". Against
60-digit arithmetic, the relative error is below 5e-11 through 270 orders of magnitude.
At 37 standard deviations the price falls below the smallest double. The claim was close,
and is now measured rather than asserted.

## References

- Black, F. and Scholes, M. (1973), "The pricing of options and corporate liabilities", *Journal of Political Economy*.
- Merton, R. C. (1973), "Theory of rational option pricing", *Bell Journal of Economics and Management Science*.
- Black, F. (1976), "The pricing of commodity contracts", *Journal of Financial Economics*.
- Garman, M. and Kohlhagen, S. (1983), "Foreign currency option values", *Journal of International Money and Finance*.
- Asay, M. (1982), "A note on the design of commodity option contracts", *Journal of Futures Markets*.
- Breeden, D. and Litzenberger, R. (1978), "Prices of state-contingent claims implicit in option prices", *Journal of Business*.
- Haug, E. G. (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed., McGraw-Hill.
- QuantLib, `AnalyticEuropeanEngine`; mpmath (Johansson et al.); Hypothesis (MacIver et al.).
