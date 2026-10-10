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
