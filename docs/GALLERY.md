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
