# Implied volatility and the market: an exact inverse, and what a real option chain looks like once cleaned

Every later day of the build starts from implied volatilities:

- the arbitrage-free surface of Day 4;
- the Heston calibration of Day 5;
- the local volatility of Day 6.

Day 3 makes sure those numbers are right. It rebuilds the inversion from price to
volatility so that it is exact, and turns a raw S&P 500 option chain into clean quotes
with a stated reason for every one it drops.

Four findings came out of the day:

- the v1.0 inversion was wrong for small prices;
- the textbook Black formula loses digits in the wings;
- a cleaning rule that looks sensible ate whole runs of good quotes;
- Yahoo Finance's implied volatilities assume rates and dividends are zero.

---

## 1. The Black function, computed so that it never cancels

![The textbook Black formula loses digits in the wings](../images/black-cancellation.png)

An out-of-the-money call costs `F N(d1) - K N(d2)`. When the total volatility is small,
the two terms are nearly equal. At `ln(K/F) = 0.6` and total volatility 0.02, the
literal formula keeps nine digits of sixteen, and fewer further out.

The fix is an identity from Jaeckel (2015). With `h = x/s` and `t = s/2`, the Gaussian
factors of the two terms are the same number, so

    b = exp(-(h^2 + t^2)/2) / sqrt(2 pi) * [Y(h + t) - Y(h - t)],

where `Y` is the Mills ratio. The first factor is the vega, so `ln b` is available even
where the price underflows.

The difference of Mills ratios is the only part that can still cancel. Where it would,
it is computed as an integral, `int Y'(z) dz`, by 16-point Gauss-Legendre. `Y'` itself
comes from Laplace's continued fraction, which subtracts nothing
([ADR 0009](../adr/0009-the-black-function-is-computed-through-the-mills-ratio.md)).

Against 60-digit arithmetic, `ln b` is right to a few roundings of its own size, down to
`ln b = -6.8 million`. `black76_price`, which every original module calls, now goes
through this function.

## 2. The inversion: v1.0 stopped early, and how "exact" is measured

![Implied volatility to the last digit](../images/iv-accuracy.png)

The v1.0 inversion was Newton's method on the price, stopped when the price residual was
below `1e-12` times the price plus `1e-15`. The absolute term ends the iteration early
once the price is small. Against exact prices:

| Normalised price | Quotes | v1.0, worst | v1.0, median | Now, worst |
|---|---|---|---|---|
| above 1e-5 | 3,092 | 3.7e-12 | 3.6e-16 | 9.3e-16 |
| 1e-10 to 1e-5 | 678 | 6.4e-8 | 1.0e-13 | 2.9e-16 |
| 1e-20 to 1e-10 | 606 | 18% | 2.1e-4 | 2.2e-16 |
| 1e-50 to 1e-20 | 742 | 102% | 48% | 2.5e-16 |
| 1e-300 to 1e-200 | 310 | 404% | 330% | 2.8e-16 |

How much this mattered for market data was limited. The cheapest SPX quote, 0.05 on an
index at 7,800, is a normalised price of about 6e-6, so real quotes were inverted to
better than 1e-7. Model prices are a different matter: a Heston price in the wing, or a
density read from a fitted surface, reaches the failing range.

The new inversion follows Jaeckel's *Let's be rational*
([ADR 0010](../adr/0010-implied-volatility-is-inverted-on-log-price-objectives.md)):

- third-order Householder steps;
- on `ln b` below half the price bound, and on `ln(b_max - b)` above it;
- every derivative in closed form.

Its accuracy is measured against the problem's own conditioning. One rounding of the
price moves the volatility by `eps * beta / (s * vega)`, and near the bound that can be
large. On 1,816 well-posed quotes the worst error is **4.0 condition numbers**.
Jaeckel's own reference implementation gets 4.1 on the same quotes, and the two agree
to a median of 2e-16.

![How many steps the inversion takes](../images/inversion-iterations.png)

Two million random quotes take at most 7 iterations, 4.7 on average. Jaeckel's
rational-cubic starting points take two. The asymptotic starts here cost speed, not
accuracy.

Stress tests found two faults in the stopping rule, both fixed:

- a bracket check that replaced a converged answer by a bisection;
- a one-ulp limit cycle, in 4 quotes of 200,000.

## 3. Forwards and discount factors from parity

![Forwards and discount factors from put-call parity](../images/parity-forward.png)

For European options, `C - P = D (F - K)` at every strike. Fitting that line across
strikes gives the forward and the discount factor with no rate curve or dividend
forecast assumed. Three choices make the fit robust:

- **A robust start.** A weighted fit pulled by one stale, tightly quoted pair trims the
  good quotes instead of the bad. In the December AM expiry it kept 12 of 199 pairs and
  implied a 45% interest rate. The fit now starts from Siegel's repeated-median line,
  which tolerates up to half the points being wrong wherever they sit.
- **Trimming at three bid-ask half-widths**, refitted until the set settles. On a
  synthetic chain a block of stale pairs seven half-widths off is excluded and the
  forward is right to 0.001. The limit is stated, and tests pin both sides of it: a
  block under four half-widths off is admitted and moves the forward by two points.
- **One curve across expiries.** A week out, a single expiry pins the rate only to
  several per cent. A three-factor Nelson-Siegel curve through the 52 reliable expiries
  gives 4.25% at the short end, 4.92% at one year and 5.33% at three. The forwards are
  refitted on it, with a median standard error of 0.04 index points.

The fitted line passes through 99% of the bid-ask boxes it used. Around the curve,
however, the expiries scatter with a median |z| of 2.1, about three times what the quote
widths predict. The most likely cause is that the quotes are not synchronous.

The forwards also show a timing effect. Against the 16:00 closing print (7,811.54) the
short-dated carry is negative. SPX options quote until 16:15, and the forwards'
intercept is an implied spot of 7,815.39. Against that, the dividend yield settles at
0.8% beyond three months.

## 4. Parity is an equality only for European options

![Parity is an equality for European options only](../images/american-parity.png)

SPY and Apple options are American. Early exercise makes an in-the-money put worth more
than its European twin, so `C - P` bends below the European line. In-the-money calls,
exercised before dividends, lift it above.

Fitted freely, the bend gives a discount factor above one, as if rates were negative.
American chains are therefore fitted only within 5% of spot, with the discount factor
taken from the SPX curve. How to price early exercise properly belongs to Day 8.

## 5. Cleaning, one stated reason at a time

![From a raw chain to clean quotes](../images/cleaning-funnel.png)

Every quote is either kept or removed by a named rule, and the counts add up to the raw
chain ([ADR 0011](../adr/0011-market-data-is-snapshotted-locally-and-cleaned-for-stated-reasons.md)).
For the 17,461 SPX quotes of 9 October 2026:

| Rule | Removed |
|---|---|
| no bid | 800 |
| crossed or locked | 66 |
| no open interest | 536 |
| wide spread (over 50% of mid) | 564 |
| in the money (the OTM twin is kept) | 6,092 |
| strike arbitrage, at bid and ask | 218 |
| **kept** | **9,185** |

Strike arbitrage is tested at the bid and the ask, so only a pattern that could be
traded counts. The worst one in the chain was an 8025 put quoted 208 points above its
neighbours.

The first version of the rule removed, each round, the quote involved in the most
violations. A stale quote and its honest neighbour share every butterfly, so on a tie
the neighbour went, and then the next one. In one expiry the rule removed 364 of 391
quotes, nearly all of them good.

Each round now tries every candidate and removes the one whose loss leaves the least
arbitrage behind. That expiry loses 25. In the expiry that still loses the most, every
removed quote sits at least 51 half-spreads off its neighbours' line, with a median of
1,249, and last traded a median of 16 days ago.

Violations at the mid price that vanish within the spread are left in, 1,706 of them.
They are noise for the Day 4 surface fit to smooth, not arbitrage.

## 6. The smile, and the vendor's numbers

![The S&P 500 smile across the term structure](../images/market-smiles.png)

The cleaned SPX smile:

| Expiry | At-the-money volatility | Median bid-ask width |
|---|---|---|
| a week | 9.4% | 0.50 vol points |
| a year | 16.2% | 0.09 vol points |
| two years | 17.4% | 0.71 vol points |

Puts and calls meet at the money because the forward is the market's own.

![Yahoo Finance's implied volatilities assume zero rates](../images/vendor-iv.png)

Yahoo Finance's figures do not meet there. They are reproduced to within 0.1 vol
point for 63% of quotes, and a median gap of 0.07, by inverting the mid price with the
forward set to spot and no discounting. Against the market's forwards, only 4% are
within 0.1 point. The error grows with maturity, in opposite directions for the two
sides:

| Years to expiry | Puts | Calls |
|---|---|---|
| under 0.1 | -0.30 | +0.70 |
| 0.1 to 0.5 | -1.08 | +1.29 |
| 0.5 to 1 | -2.29 | +2.48 |
| over 1 | -3.86 | +3.81 |

(Yahoo minus the lab's figure, in volatility points.)

## Reproducing

```bash
pip install -e ".[dev]"
volsurf chains fetch ^SPX SPY AAPL   # stores today's snapshot under data/chains/ (not committed)
volsurf gallery --only 3             # redraws this note's charts from the latest snapshot
pytest tests/market                  # against mpmath, Jaeckel's implementation and QuantLib
volsurf validate
```

The charts here come from the close of 9 October 2026. A later snapshot gives other
numbers, and the same checks.
