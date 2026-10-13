# Gallery

Every chart is rebuilt from source with `volsurf gallery`. The figures of the live S&P 500 analysis (the surface, the smiles, the Heston fit, the hedging laboratory) come from `volsurf analyze` and are shown in the README.

## Day 1: The analytic foundation

**Sixteen Greeks of a European call** - Every Greek against spot, three months, one month and one week from expiry.

![Sixteen Greeks of a European call](images/greek-atlas.png)

**Every Greek against an independent reference** - Arbitrary-precision differentiation and Haug's worked examples.

![Every Greek against an independent reference](images/greek-references.png)

**Prices far in the wings** - The engine against 60-digit arithmetic, through 270 orders of magnitude.

![Prices far in the wings](images/tail-precision.png)

**Every price solves the Black-Scholes equation** - The PDE residual from the analytic Greeks, over moneyness and maturity.

![Every price solves the Black-Scholes equation](images/pde-residual.png)

**One model, five underlyings** - A stock, an index, a future, a currency and a margined future, through the cost of carry.

![One model, five underlyings](images/carry-conventions.png)

**The Greeks at expiry** - Price, delta and theta reaching their limits as time to expiry falls to zero.

![The Greeks at expiry](images/expiry-limits.png)

## Day 2: Numerical engines

**How fast each lattice converges** - Five lattices and BBSR against the exact price, European and American.

![How fast each lattice converges](images/lattice-convergence.png)

**Longstaff and Schwartz's 'American' prices are Bermudan prices** - Their Table 1 against the American and the fifty-date Bermudan put.

![Longstaff and Schwartz's 'American' prices are Bermudan prices](images/longstaff-schwartz-table.png)

**Crank-Nicolson rings at the strike** - Gamma from the finite-difference grid, with and without Rannacher's start.

![Crank-Nicolson rings at the strike](images/rannacher.png)

**Where an American put should be exercised** - The early-exercise boundary through the option's life, and today's value.

![Where an American put should be exercised](images/exercise-boundary.png)

**Monte Carlo error against the number of samples** - Pseudo-random, antithetic, control-variate and scrambled Sobol' sampling.

![Monte Carlo error against the number of samples](images/monte-carlo-convergence.png)

**Least-squares Monte Carlo, bracketed from both sides** - The out-of-sample lower bound and the dual upper bound around the Bermudan value.

![Least-squares Monte Carlo, bracketed from both sides](images/early-exercise-bounds.png)

## Day 3: Implied volatility and the market

**The textbook Black formula cancels in the wings** - Out-of-the-money prices against 60-digit arithmetic: the literal formula and the Mills-ratio form.

![The textbook Black formula cancels in the wings](images/black-cancellation.png)

**Implied volatility to the last digit** - The v1.0 Newton inversion and the Jaeckel-style one, against exact prices from 1e-300 to the forward.

![Implied volatility to the last digit](images/iv-accuracy.png)

**How many steps the inversion takes** - Householder iterations over strike and volatility, and their distribution over random quotes.

![How many steps the inversion takes](images/inversion-iterations.png)

**Forwards and discount factors from put-call parity** - S&P 500 options: the synthetic forward, its residuals, the implied rate curve and dividend yield.

![Forwards and discount factors from put-call parity](images/parity-forward.png)

**Parity is an equality for European options only** - SPX against SPY and Apple: early exercise bends the synthetic forward.

![Parity is an equality for European options only](images/american-parity.png)

**From a raw chain to clean quotes** - What each cleaning rule removes, and the largest executable arbitrage in the chain.

![From a raw chain to clean quotes](images/cleaning-funnel.png)

**The S&P 500 smile across the term structure** - Bid, mid and ask implied volatility from a week to two years, with the vendor's figures.

![The S&P 500 smile across the term structure](images/market-smiles.png)

**Yahoo Finance's implied volatilities assume zero rates** - The vendor's figures against parity forwards, and the assumption that reproduces them.

![Yahoo Finance's implied volatilities assume zero rates](images/vendor-iv.png)

## Day 4: The arbitrage-free surface

**A smile that fits and still admits arbitrage** - Axel Vogt's SVI slice: its negative density, and a refit under hard constraints.

![A smile that fits and still admits arbitrage](images/vogt-slice.png)

**The original surface had arbitrage** - The v1.0 fit checked on the whole line: negative densities, crossing slices, and between expiries.

![The original surface had arbitrage](images/v1-arbitrage.png)

**Fitting the S&P 500 smile without arbitrage** - v1.0 SVI, eSSVI and constrained SVI against bid-ask bands, a week to two years.

![Fitting the S&P 500 smile without arbitrage](images/smile-fits.png)

**What the no-arbitrage conditions cost in fit** - Error and share inside the spread for SSVI, eSSVI, v1.0 and constrained SVI.

![What the no-arbitrage conditions cost in fit](images/fit-quality.png)

**The distributions the S&P 500 options price** - Risk-neutral densities with their mass, mean and Breeden-Litzenberger checks.

![The distributions the S&P 500 options price](images/risk-neutral-densities.png)

**Between expiries: interpolate prices, not variances** - Total variance through time, and the worst density of every intermediate smile.

![Between expiries: interpolate prices, not variances](images/time-interpolation.png)

**What the wings say about the moments** - Wing slopes against Lee's bound, and the moments of the index they imply.

![What the wings say about the moments](images/lee-wings.png)

## Day 5: Heston

**Heston prices to the last digit** - Five engines against a 40-digit reference, far into both wings.

![Heston prices to the last digit](images/heston-engines.png)

**Where the contour runs** - The integrand by damping, and why the quadrature is graded towards the origin.

![Where the contour runs](images/contour-damping.png)

**Simulating Heston across the Feller boundary** - Euler with full truncation and Andersen's QE against exact prices.

![Simulating Heston across the Feller boundary](images/simulation-bias.png)

**QE against the exact variance transition** - One step near zero: exact non-central chi-squared, QE and Euler.

![QE against the exact variance transition](images/qe-transition.png)

**The moment strip and the wings it implies** - Moment explosion through time, and Lee's wing slopes.

![The moment strip and the wings it implies](images/moment-explosion.png)

**Heston calibrated to the S&P 500** - Smiles from a month to two years against the bid-ask, and where five parameters fall short.

![Heston calibrated to the S&P 500](images/heston-calibration.png)

**What the S&P 500 surface pins down** - Multistart, the profile in kappa, and the least determined direction of the Jacobian.

![What the S&P 500 surface pins down](images/heston-identifiability.png)

## Day 6: Local volatility and SABR

**S&P 500 local volatility in closed form** - Local against implied volatility, the term structure, and the noise daily weeklies bring to the wings.

![S&P 500 local volatility in closed form](images/local-vol-surface.png)

**Repricing the surface from its own local volatility** - The forward Dupire equation against the surface, as the grid is refined.

![Repricing the surface from its own local volatility](images/dupire-round-trip.png)

**Hagan's SABR formula against the model** - The expansion against an exact conditional reference, by strike and by nu^2 T.

![Hagan's SABR formula against the model](images/sabr-accuracy.png)

**Where Hagan's formula stops being a model** - Negative densities and the model's own prices from QuantLib's SABR PDE.

![Where Hagan's formula stops being a model](images/sabr-density.png)

**SABR fitted to each S&P 500 expiry** - Parameters by maturity, and the fit against SVI.

![SABR fitted to each S&P 500 expiry](images/sabr-spx.png)

**Same smile today, different smiles tomorrow** - The smile after the index moves, under local volatility and SABR, and the deltas.

![Same smile today, different smiles tomorrow](images/smile-dynamics.png)

## Day 7: Exotic options

**Barriers watched continuously or once a day** - Discrete monitoring by Monte Carlo, the BGK shifted barrier, and the Brownian bridge, against the closed form.

![Barriers watched continuously or once a day](images/barrier-monitoring.png)

**Arithmetic Asians: a control variate, and a reference that is not** - Variance removed by the geometric control, and QuantLib's Choi and Turnbull-Wakeman engines against it.

![Arithmetic Asians: a control variate, and a reference that is not](images/asian-control-variate.png)

**Lookbacks: the extreme seen once a day** - Discrete against continuous lookbacks as monitoring is refined, and the closed forms against volatility.

![Lookbacks: the extreme seen once a day](images/lookback-monitoring.png)

**A digital is a call spread, and the smile prices it** - S&P 500 digital calls from the smile against Black-Scholes at one volatility.

![A digital is a call spread, and the smile prices it](images/digital-smile.png)

**One surface, three models, three prices** - A book of vanillas and exotics under Black-Scholes, Heston and local volatility.

![One surface, three models, three prices](images/exotic-model-risk.png)

