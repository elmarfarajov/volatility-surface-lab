# Methodology

This document derives every model implemented in `src/volsurf/`, explains the numerical
choices, and states the limitations. Section numbers match the pipeline order.

- [1. Market data and forward extraction](#1-market-data-and-forward-extraction)
- [2. Black-76 and implied volatility](#2-black-76-and-implied-volatility)
- [3. Lattice and Monte Carlo engines](#3-lattice-and-monte-carlo-engines)
- [4. SVI smiles and static arbitrage](#4-svi-smiles-and-static-arbitrage)
- [5. The surface and Dupire local volatility](#5-the-surface-and-dupire-local-volatility)
- [6. The Heston model](#6-the-heston-model)
- [7. Heston calibration](#7-heston-calibration)
- [8. Hedging laboratory](#8-hedging-laboratory)
- [9. Limitations](#9-limitations)
- [References](#references)

---

## 1. Market data and forward extraction

**Quote hygiene.** A quote is kept only if $0 < \text{bid} < \text{ask}$ and the relative spread
$(\text{ask}-\text{bid})/\text{mid}$ is below 40%. For S&P 500 index options, Yahoo Finance mixes
AM-settled monthly (`SPX`) and PM-settled weekly (`SPXW`) contracts under one expiry date; the
dominant root is kept, duplicates are resolved by open interest, and time to expiry uses the
correct settlement time (09:30 or 16:00 New York).

**Model-free forwards.** For European options put-call parity holds exactly,

$$C(K) - P(K) = D\,(F - K),$$

so regressing $C - P$ on $K$ across near-the-money strikes gives slope $-D$ and intercept $DF$.
The regression is weighted by inverse bid-ask width and iterated with a 3-MAD outlier filter.
This recovers the discount factor $D$ and forward $F$ for every expiry **without assuming an
interest rate or a dividend yield**: discrete dividends, borrow costs and funding spreads are all
embedded in what the market charges. The implied rate $-\ln D / T$ and carry $\ln(F/S)/T$ are
reported as diagnostics.

**Out-of-the-money selection.** Puts are used below the forward and calls above it. Their prices
consist entirely of time value, which makes implied volatility inversion well conditioned.

## 2. Black-76 and implied volatility

With total volatility $\nu = \sigma\sqrt{T}$ and $d_{1,2} = \ln(F/K)/\nu \pm \nu/2$,

$$C = D\,[F\,N(d_1) - K\,N(d_2)], \qquad P = D\,[K\,N(-d_2) - F\,N(-d_1)].$$

Both are evaluated directly (never via parity subtraction) so a deep out-of-the-money price of
$10^{-18}$ is still accurate. Spot Black-Scholes-Merton is the special case $F = Se^{(r-q)T}$,
$D=e^{-rT}$. Delta, gamma, vega, theta, rho, vanna and volga are analytic and tested against
finite differences.

**Implied volatility inversion.** In normalised units $c = C/(DF)$, $m = K/F$, the OTM price is
convex in $\sigma$ below the inflection point

$$\sigma^* = \sqrt{2\,|\ln m| / T}$$

and concave above it (Manaster & Koehler, 1982). Newton's method started at $\sigma^*$ therefore
converges monotonically. The implementation vectorises Newton over the whole chain and keeps a
bisection bracket per quote: a Newton step that leaves the bracket, or a vanishing vega, falls
back to bisection. Prices outside the static no-arbitrage bounds return `NaN` rather than a
fabricated volatility. 50,000 quotes invert in well under a second with errors below $10^{-8}$.

## 3. Lattice and Monte Carlo engines

**Leisen-Reimer tree.** Up/down probabilities come from the Peizer-Pratt inversion of the normal
CDF,

$$h(z, n) = \tfrac12 + \tfrac12\,\text{sgn}(z)\sqrt{1 - \exp\!\left[-\left(\frac{z}{n + 1/3 + 0.1/(n+1)}\right)^{2}\left(n + \tfrac16\right)\right]},$$

with $p = h(d_2)$, $\bar p = h(d_1)$, $u = e^{(r-q)\Delta t}\,\bar p/p$. Because the tree is centred on
the strike, European prices converge at second order without the odd/even oscillation of
Cox-Ross-Rubinstein; the validation suite shows 201 LR steps beating 2,001 CRR steps. American
exercise is handled by backward induction with an early-exercise check at every node.

**Variance reduction.** GBM Monte Carlo uses antithetic pairs and the discounted terminal price
as a control variate (its expectation $Se^{-qT}$ is known). The optimal coefficient
$\beta = \text{Cov}(Y, X)/\text{Var}(X)$ is estimated from the same sample, and standard errors
are computed from independent antithetic pairs.

**Longstaff-Schwartz.** Continuation values are estimated by regressing discounted future cash
flows on a cubic polynomial in $S/K$, using in-the-money paths only, and exercising where the
immediate payoff exceeds the fitted continuation value.

## 4. SVI smiles and static arbitrage

Raw SVI (Gatheral, 2004) models total implied variance $w(k) = \sigma_{BS}^2(k)\,T$ in log
forward-moneyness $k = \ln(K/F)$:

$$w(k) = a + b\left[\rho\,(k-m) + \sqrt{(k-m)^2 + s^2}\right].$$

**Butterfly arbitrage.** A smile admits a non-negative risk-neutral density if and only if
Durrleman's function is non-negative:

$$g(k) = \left(1 - \frac{k\,w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac1w + \frac14\right) + \frac{w''}{2} \;\ge\; 0,$$

and the density of $\ln(S_T/F)$ is then

$$p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}\exp\!\left(-\frac{d_-(k)^2}{2}\right), \qquad d_-(k) = -\frac{k}{\sqrt{w}} - \frac{\sqrt w}{2}.$$

**Calendar arbitrage.** Call prices must increase with maturity at fixed forward-moneyness, which
is equivalent to $w(k, T_2) \ge w(k, T_1)$ for $T_2 > T_1$.

**Wings.** Roger Lee's moment formula bounds the asymptotic slope of total variance by 2, so
$b(1 + |\rho|) \le 2$.

**Fitting.** For fixed $(m, s)$ the model is linear in $(a,\, b s,\, \rho b s)$ (Zeliade Systems,
2009), so an $(m, s)$ grid is scanned with ordinary least squares to find good starting points.
Each start is then refined by bounded least squares on implied-volatility errors, augmented with
penalty residuals evaluated on a dense strike grid for negative variance, $g(k) < 0$, the Lee
bound, and any crossing of the previous expiry's total variance. Slices are fitted from short to
long maturity so each respects the one before it. The lower bound on $s$ is the median strike
spacing: curvature finer than the quote grid is not identified by the data and otherwise shows up
as spurious spikes in the implied density.

## 5. The surface and Dupire local volatility

**Interpolation.** At fixed $k$ the slice variances are joined in $T$ with a monotone
piecewise-cubic Hermite interpolant (PCHIP) through the origin $w(k, 0) = 0$. PCHIP preserves
monotonicity, so calendar-arbitrage-free slices give a calendar-arbitrage-free surface, and its
$T$-derivative is continuous, which keeps local volatility free of saw-tooth artefacts.

**Local volatility.** Dupire's equation written in total implied variance (Gatheral, 2006) is

$$\sigma_{\text{loc}}^2(k, T) = \frac{\partial w / \partial T}{g(k, T)},$$

with the same $g$ as above. $\partial_T w$ comes from the PCHIP derivative; $\partial_k w$ and
$\partial_{kk} w$ from central differences. Points where $g$ is not strictly positive are
masked instead of being reported as a meaningless number.

**Plotting window.** Figures use standardised moneyness $k/\sqrt{T}$ restricted to the window
actually quoted by most expiries, so that what is shown is the market and not extrapolated wings.

## 6. The Heston model

Under the risk-neutral measure,

$$dS_t = (r-q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t, \qquad dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^v_t, \qquad d\langle W^S, W^v\rangle_t = \rho\,dt.$$

**Characteristic function.** For $x_T = \ln(S_T/F_T)$, $\phi(u) = \exp\big(A(u) + B(u)\,v_0\big)$ with

$$\beta = \kappa - \rho\sigma i u,\quad d = \sqrt{\beta^2 + \sigma^2(iu + u^2)},\quad g = \frac{\beta - d}{\beta + d},$$

$$A = \frac{\kappa\theta}{\sigma^2}\left[(\beta - d)T - 2\ln\frac{1 - g e^{-dT}}{1 - g}\right], \qquad B = \frac{\beta - d}{\sigma^2}\,\frac{1 - e^{-dT}}{1 - g e^{-dT}}.$$

This is the "little Heston trap" form (Albrecher et al., 2007), which keeps the complex logarithm
on its principal branch for long maturities. Separating $A$ and $B$ is a deliberate design
choice: one set of exponents serves thousands of simulated paths that differ only in $v_0$.

**COS pricing (Fang & Oosterlee, 2008).** On a truncation interval $[a, b]$ for $y = \ln(S_T/K)$,

$$P(K) = D\,K \sum_{k=0}^{N-1}{}' \; \text{Re}\!\left[\phi\!\left(\tfrac{k\pi}{b-a}\right) e^{\,i k\pi (x - a)/(b-a)}\right] U_k, \qquad x = \ln(F/K),$$

with payoff coefficients $U_k = \frac{2}{b-a}\big[\psi_k(a, 0) - \chi_k(a, 0)\big]$ and calls from
parity. Implementation details that matter for accuracy and speed:

- *Interval sizing.* The width is $2L\sqrt{c_2}$ with $c_2 = -\big(A''(0) + B''(0)\,v_0\big)$ the exact
  variance of $x_T$, obtained from a central difference of the closed-form exponents. An earlier
  version sized the interval with an approximate closed-form cumulant; at high vol-of-vol it
  underestimated $c_2$ by ~30% and produced pricing errors of $3\times10^{-4}$. The exact variance
  with $L = 20$ brings the error on the published benchmark to $3\times10^{-8}$.
- *Adaptive term count.* $N$ is the smallest power of two with $|\phi(N\pi/(b-a))| < 10^{-14}$.
- *Real arithmetic.* $\text{Re}[\phi\,e^{i\theta}] = |\phi|\cos(\arg\phi + \theta)$, which replaces two
  complex exponentials and a complex product with one real exponential and a cosine.
- *Payoff region clipping.* The put payoff is integrated over $[a, \min(0, b)]$, so deep in- and
  out-of-the-money strikes need no special handling.

**Sensitivities from the same series.** Holding the interval fixed,
$\partial_x e^{iu(x-a)} = iu\,e^{iu(x-a)}$ and $\partial_{v_0}\phi = B(u)\,\phi$, so

$$\frac{\partial V}{\partial S} = \frac{D K}{S}\sum{}'\,\text{Re}\!\left[iu_k\,\phi\,e^{iu_k(x-a)}\right]U_k, \qquad \frac{\partial V}{\partial v} = DK \sum{}'\,\text{Re}\!\left[B(u_k)\,\phi\,e^{iu_k(x-a)}\right]U_k$$

come at almost no extra cost. For hedging simulations, states are grouped into variance quantile
buckets so each bucket's interval and term count match its own variance range; this made
per-path Greeks for 10,000 paths ~20x faster than a single common interval.

**Reference engine.** Heston's original Gil-Pelaez inversion,
$C = D\,(F P_1 - K P_2)$ with $P_j = \tfrac12 + \tfrac1\pi\int_0^\infty \text{Re}\big[e^{-iu\ln(K/F)}\phi_j(u)/(iu)\big]du$,
is evaluated with adaptive quadrature. It is slow but independent, and it validates COS to
$\le 2\times10^{-5}$ across randomly drawn parameter sets.

**Simulation: Andersen's QE scheme.** Given $v_t$, the next variance is drawn from a moment-matched
distribution: a scaled non-central-chi-squared-like quadratic $a(b + Z)^2$ when
$\psi = s^2/m^2 \le 1.5$, and a mixture of a point mass at zero and an exponential otherwise. The
log-price step integrates the correlated part exactly,

$$\ln S_{t+\Delta} = \ln S_t + (r-q)\Delta + K_0^* + K_1 v_t + K_2 v_{t+\Delta} + \sqrt{K_3 v_t + K_4 v_{t+\Delta}}\;Z,$$

where $K_0^*$ is Andersen's martingale correction, chosen so that
$\mathbb{E}[S_{t+\Delta} \mid S_t, v_t] = S_t e^{(r-q)\Delta}$ holds exactly in the discretised model.

## 7. Heston calibration

**Objective.** With market price $C_i$, model price $C_i^H(\Theta)$ and Black vega $\mathcal V_i$,

$$\min_\Theta \sum_i w_i\left(\frac{C_i^H(\Theta) - C_i}{\mathcal V_i}\right)^2 \;\approx\; \min_\Theta \sum_i w_i\left(\sigma_i^H(\Theta) - \sigma_i\right)^2,$$

a first-order implied-volatility error that needs no inversion inside the optimiser. Weights are
inverse bid-ask widths in volatility units, so liquid quotes dominate. Reported errors always use
exact implied volatilities of the final model prices.

**Search.** The default `multistart` method anchors $v_0$ to the shortest expiry's ATM variance and
$\theta$ to the longest's, spans $(\kappa, \sigma, \rho)$ with a small design of four starts, runs
trust-region-reflective least squares from each on a strike subsample, and polishes the best on the
full quote set. `global` replaces the starts with differential evolution. On synthetic surfaces
generated from known parameters, including a Feller-violating high vol-of-vol regime, the
multistart method recovers all five parameters to within a few percent in 3-5 seconds.

**Feller condition.** $2\kappa\theta \ge \sigma^2$ is reported, not imposed. Equity index surfaces
typically require it to be violated, and imposing it would bias the fit.

## 8. Hedging laboratory

A dealer sells one European call at model value $V_0$ and holds $\Delta_t$ shares, rebalancing
$N$ times. The self-financing cash account accrues interest and dividends, and the discounted
terminal P&L is

$$\Pi = e^{-rT}\left[\text{cash}_T + \Delta_{T^-} S_T - (S_T - K)^+\right].$$

**Black-Scholes world.** With the true delta the only error is discretisation, and Derman & Kamal
(1999) show

$$\text{std}(\Pi) \approx \sqrt{\pi/4}\;\frac{\mathcal V\,\sigma}{\sqrt N}.$$

The simulation reproduces this within a few percent, which validates the P&L accounting.

**Heston world.** Paths are simulated with QE and three hedge ratios are compared:

1. Black-Scholes delta at the option's implied volatility (standard desk practice),
2. the Heston delta $\partial V/\partial S$,
3. the minimum-variance delta (Bakshi, Cao & Chen, 1997)

$$\Delta^{MV} = \frac{\text{Cov}(dV, dS)}{\text{Var}(dS)} = \frac{\partial V}{\partial S} + \frac{\rho\,\sigma}{S}\,\frac{\partial V}{\partial v},$$

which also neutralises the part of the variance shock that is correlated with the spot move.
Variance risk orthogonal to spot cannot be hedged with the underlying alone, so hedging error
plateaus as $N$ grows instead of vanishing.

## 9. Limitations

- **Data.** Yahoo Finance quotes are delayed and occasionally stale outside market hours; the
  cleaning step removes obviously bad quotes but cannot certify synchronicity across strikes.
- **American exercise.** Single-stock and ETF options are American; using OTM quotes limits, but
  does not remove, the early-exercise premium's effect on implied volatility. Index options such
  as SPX are European and are the intended use case.
- **Heston itself.** A single-factor diffusive stochastic volatility model cannot produce the
  steep short-dated skew of equity indices; the calibration trades short-maturity wing error for
  long-maturity fit. Jumps (Bates) or rough volatility would address this.
- **SVI.** Raw SVI slices are fitted with arbitrage penalties rather than the fully arbitrage-free
  SSVI parameterisation of Gatheral & Jacquier (2014); diagnostics are reported so any residual
  violation is visible.
- **Hedging lab.** Transaction costs, discrete dividends and model misspecification beyond Heston
  are not modelled; the experiment isolates volatility risk by construction.

## References

- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance* 11(3).
- Bakshi, G., Cao, C. & Chen, Z. (1997). Empirical performance of alternative option pricing models. *Journal of Finance* 52(5).
- Black, F. (1976). The pricing of commodity contracts. *Journal of Financial Economics* 3.
- Derman, E. & Kamal, M. (1999). When you cannot hedge continuously: the corrections to Black-Scholes. *Risk* 12.
- Dupire, B. (1994). Pricing with a smile. *Risk* 7(1).
- Durrleman, V. (2005). From implied to spot volatilities. PhD thesis, Princeton University.
- Fang, F. & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing* 31(2).
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization. Global Derivatives & Risk Management.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1).
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies* 6(2).
- Lee, R. W. (2004). The moment formula for implied volatility at extreme strikes. *Mathematical Finance* 14(3).
- Leisen, D. & Reimer, M. (1996). Binomial models for option valuation: examining and improving convergence. *Applied Mathematical Finance* 3(4).
- Longstaff, F. A. & Schwartz, E. S. (2001). Valuing American options by simulation. *Review of Financial Studies* 14(1).
- Manaster, S. & Koehler, G. (1982). The calculation of implied variances from the Black-Scholes model. *Journal of Finance* 37(1).
- Zeliade Systems (2009). Quasi-explicit calibration of Gatheral's SVI model. White paper.
