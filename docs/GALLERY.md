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
