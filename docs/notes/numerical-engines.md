# Numerical engines: lattices, finite differences and Monte Carlo, against references that are right

Most options have no closed form. An American put can be exercised at any time, and the
Heston model of Day 5 and the exotics of Day 7 need the same tools. Day 2 builds the
three families of numerical engine a desk uses. Each is held to references whose own
error is far below the engine's, and its convergence is measured rather than assumed.

The most useful finding is about one of those references.

---

## 1. Longstaff and Schwartz's "American" prices are Bermudan prices

![Longstaff and Schwartz's 'American' prices are Bermudan prices](../images/longstaff-schwartz-table.png)

Longstaff and Schwartz's (2001) Table 1 is the standard test of least-squares Monte Carlo.
It gives 20 American puts (strike 40, rate 6%) with a finite-difference value for each.
The lab checked its American tree against one of them, 4.478. The tree was 0.008 off, and
the check passed at a tolerance of 0.01.

Measured against Andersen, Lake and Offengenden's (2016) high-precision method (QuantLib,
`QdFpAmericanEngine`), the table's column is not the American price:

| Case (S, sigma, T) | Table | American | Bermudan, 50 a year |
|---|---|---|---|
| 36, 0.2, 1 | 4.478 | 4.486674 | 4.477792 |
| 38, 0.2, 1 | 3.250 | 3.257197 | 3.250102 |
| 40, 0.2, 1 | 2.314 | 2.319574 | 2.314046 |
| 42, 0.2, 1 | 1.617 | 1.621155 | 1.616955 |
| 44, 0.2, 1 | 1.110 | 1.112962 | 1.109850 |

In 15 of the 20 cases (every one-year case, and every two-year case at 20% volatility)
the table agrees to within 0.0005 with a **Bermudan** put exercisable fifty times a year,
the exercise schedule of their own Monte Carlo. It lies 0.003 to 0.009 below the American
put. In the other five (two years, 40% volatility) it lies between the two. This is
stated as measured; what the authors computed is not claimed.

The reference values are now computed, and five independent methods converge to the
American ones: two QuantLib finite-difference grids, two Leisen-Reimer trees, and the
fixed-point method itself. They are stored with their provenance in `numerics.references`
(ADR 0005).

## 2. Lattices

![How fast each lattice converges](../images/lattice-convergence.png)

`numerics.lattice` builds five trees:

| Tree | Construction | European convergence |
|---|---|---|
| Cox-Ross-Rubinstein (1979) | u = e^{sigma sqrt dt}, the exact martingale probability | first order, oscillating |
| Jarrow-Rudd (1983) | the drift in the moves, p close to 1/2 | first order, oscillating |
| Tian (1993) | three moments matched | first order, oscillating |
| Leisen-Reimer (1996) | Peizer-Pratt inversion, centred on the strike, odd n | **second order**, smooth |
| trinomial (Boyle, 1986) | dx = sigma sqrt(3 dt) | first order, smoother |

The oscillation comes from where the strike falls between the final nodes, which changes
with every step added. Between 400 and 440 steps the CRR error gets worse about as often
as it gets better. Two remedies apply to any tree:

- **BBS** (Broadie and Detemple, 1996) replaces the last step with the Black-Scholes value;
- **Richardson extrapolation** of BBS at n and 2n steps (**BBSR**) cancels the leading
  error term.

For the American put, BBSR on CRR is good to 1e-5 at 1,000 steps.

**The same trees as QuantLib.** Tian and Leisen-Reimer reproduce QuantLib's
`BinomialVanillaEngine` to 1e-11, European and American, so they are the same trees.
QuantLib's "CRR" and "Jarrow-Rudd" are log-space variants with a first-order probability.
The trees here follow the 1979 and 1983 papers, and the two differ by about 5e-5 at 301
steps (ADR 0008).

**Properties for any contract** (Hypothesis):

- European <= Bermudan <= American;
- early exercise is worth at least the intrinsic value;
- an American call on an underlying without dividends equals the European call (Merton,
  1973).

## 3. Finite differences

![Crank-Nicolson rings at the strike](../images/rannacher.png)

`numerics.pde` steps the Black-Scholes PDE in log-spot with the theta-scheme, the strike
on a node and spot read off a cubic spline:

| Scheme | Order in time, measured | Its weakness |
|---|---|---|
| implicit Euler | 1 (error halves as the grid doubles) | slow |
| Crank-Nicolson | 2 (error quarters) | the payoff's kink excites a mode it barely damps |
| **Rannacher**: 4 implicit half-steps, then Crank-Nicolson | 2 | none here |

On a coarse time grid (800 x 25), plain Crank-Nicolson's gamma near the strike is wrong by
0.85, twenty times gamma itself, and its at-the-money price by 0.03. Four implicit
half-steps bring these to 1.5e-5 and 7e-4. Rannacher is the default (ADR 0006).

Early exercise is applied by projection, `V = max(V, payoff)`, at every step (American)
or at the steps of a list of dates (Bermudan). The 1,600 x 1,600 grid is good to 3e-4 on
the American reference. The Bermudan reference values in the table above were computed
on that grid, and agree with QuantLib's finite differences to 1e-4.

![Where an American put should be exercised](../images/exercise-boundary.png)

The projection also gives the **early-exercise boundary**, the spot below which the put
is exercised. Near expiry it approaches the strike only slowly, like
`K (1 - sigma sqrt(tau |ln tau|))` (Barles, Burdeau, Romano and Samsoen, 1995). One time
step before expiry (tau = 1/800) that gives 39.3, not 40. The test asserts this
asymptote rather than the naive limit.

## 4. Monte Carlo

![Monte Carlo error against the number of samples](../images/monte-carlo-convergence.png)

`numerics.montecarlo.european` prices by four kinds of sampling, each with a standard
error:

- **pseudo-random**: error falls like N^(-1/2);
- **antithetic**: about 1.4x less variance for this call;
- **control variate** on the discounted terminal spot: about 25x less variance;
- **scrambled Sobol'** (Owen scrambling, randomised sixteen times so its error can be
  measured): error falls close to N^(-1). At 2^18 samples its variance is about 25,000
  times smaller.

## 5. Early exercise by simulation, bracketed from both sides

![Least-squares Monte Carlo, bracketed from both sides](../images/early-exercise-bounds.png)

Longstaff-Schwartz estimates when to exercise by regressing continuation values on
functions of the spot. As usually reported, the rule is fitted and priced on the same
paths. The estimate then borrows foresight, and has no statement of how good the fitted
rule is. `numerics.montecarlo.longstaff_schwartz` reports two bounds (ADR 0007):

- **a lower bound.** The fitted rule followed on independent paths is a feasible policy,
  so it cannot be worth more than the optimum.
- **a dual upper bound.** For any martingale M starting at zero, the option is worth at
  most `E[max_k (Z_k - M_k)]` (Rogers, 2002; Haugh and Kogan, 2004). The martingale is
  built from the fitted value function, its increments estimated by inner simulations one
  exercise date ahead.

For the Bermudan put (S 36, fifty dates):

| Regression on | Lower bound | Upper bound | Gap |
|---|---|---|---|
| powers of S/K | 4.480 +- 0.006 | 4.652 +- 0.005 | 0.17 |
| powers of S/K and the European price of the remaining option | 4.478 +- 0.006 | 4.481 +- 0.001 | **0.003** |

The Bermudan value, 4.47779, lies inside both. The upper bound is only as good as the
value function behind its martingale. One extra regressor, the European price, closes the
gap by a factor of sixty.

## References

- Cox, J., Ross, S. and Rubinstein, M. (1979), "Option pricing: a simplified approach", *Journal of Financial Economics*.
- Jarrow, R. and Rudd, A. (1983), *Option Pricing*, Irwin.
- Tian, Y. (1993), "A modified lattice approach to option pricing", *Journal of Futures Markets*.
- Leisen, D. and Reimer, M. (1996), "Binomial models for option valuation - examining and improving convergence", *Applied Mathematical Finance*.
- Boyle, P. (1986), "Option valuation using a three-jump process", *International Options Journal*.
- Broadie, M. and Detemple, J. (1996), "American option valuation: new bounds, approximations, and a comparison of existing methods", *Review of Financial Studies*.
- Rannacher, R. (1984), "Finite element solution of diffusion problems with irregular data", *Numerische Mathematik*.
- Barles, G., Burdeau, J., Romano, M. and Samsoen, N. (1995), "Critical stock price near expiration", *Mathematical Finance*.
- Owen, A. (1995), "Randomly permuted (t,m,s)-nets and (t,s)-sequences"; Joe, S. and Kuo, F. (2008), Sobol' direction numbers.
- Longstaff, F. and Schwartz, E. (2001), "Valuing American options by simulation: a simple least-squares approach", *Review of Financial Studies*.
- Rogers, L. C. G. (2002), "Monte Carlo valuation of American options", *Mathematical Finance*.
- Haugh, M. and Kogan, L. (2004), "Pricing American options: a duality approach", *Operations Research*.
- Andersen, L. and Broadie, M. (2004), "Primal-dual simulation algorithm for pricing multidimensional American options", *Management Science*.
- Andersen, L., Lake, M. and Offengenden, D. (2016), "High-performance American option pricing", *Journal of Computational Finance*.
