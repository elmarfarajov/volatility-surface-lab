# Local volatility and SABR: Dupire from the arbitrage-free surface, Hagan against the model, and smile dynamics

Day 4 built a surface free of static arbitrage, and Day 5 fitted Heston to it. Day 6 asks
two other models the same question. Local volatility reproduces any such surface exactly.
SABR describes one smile at a time with four numbers.

Fitted to the same smile today, the two models disagree about the smile after the index
moves, so they disagree about delta. This note measures that disagreement, and where each
model breaks.

---

## 1. Local volatility, in closed form

![S&P 500 local volatility](../images/local-vol-surface.png)

Dupire's local variance is twice the time derivative of the call over its density. The v1.0
code took both by finite differences of a surface interpolated in total variance, and
returned NaN on 4% of the S&P 500 grid.

On the Day 4 surface both quantities are exact. Between expiries the call is the price
combination `alpha c_lo + (1 - alpha) c_hi`, so its time derivative is
`-alpha'(c_hi - c_lo)`. That is non-negative because the slices are calendar-free. The
density is the same combination of the slices' closed-form densities. Local variance is
therefore positive by construction
([ADR 0018](../adr/0018-local-volatility-in-closed-form-from-the-price-interpolation.md)).

Three properties of that interpolation came to light only here.

1. **An atom before the first expiry.** The interpolation mixes the first slice with expiry
   itself, so the distribution keeps an atom at the forward. No diffusion can carry an
   atom: local variance there grows like `1/t`. The short end is now self-similar,
   `w(k, t) = (t/t_1) w_1(k)`.
2. **Noise from daily weeklies.** Any disagreement between neighbouring slices is divided by
   a small time step. Local volatility at `ln(K/F) = -0.1` swings from 8% to over 200% in the
   first two weeks. Dynamics are therefore computed on the monthly expiries.
3. **A smile from mixing.** The price combination of two log-normal laws is not
   log-normal. Between the expiries of a flat 20% surface it puts a smile of 2.8 volatility
   points at `ln(K/F) = -0.4`. The interpolation is free of arbitrage, not minimal.

## 2. The surface repriced from its own local volatility

![Repricing the surface from its own local volatility](../images/dupire-round-trip.png)

Solving the forward Dupire equation with this local volatility must give back the surface.

| Surface | Largest error |
|---|---|
| monthly expiries | within 3.6 bp of volatility at every expiry; under 1 bp beyond three months |
| Heston surface (validation suite) | 2.5 bp |
| every weekly, self-similar short end | stalls at 20 to 40 bp however fine the grid |
| every weekly, interpolation's short end (atom) | stops improving, then drifts back up as the grid is refined |

## 3. SABR: Hagan against the model

![Hagan's SABR formula against the model](../images/sabr-accuracy.png)

Hagan's formula agrees with QuantLib's `sabrVolatility` to 2e-15. That checks the algebra,
not the approximation
([ADR 0019](../adr/0019-sabr-is-held-to-an-exact-reference.md)).

For `beta = 1` an exact reference exists. Given the volatility path, the log-forward is
normal, so a price is a Black price averaged over simulated paths. The reference agrees
with QuantLib's SABR PDE to within its noise.

With `alpha 0.2, rho -0.7, nu 1`, Hagan's error in the put wing is:

| Maturity | Error |
|---|---|
| five weeks | 4 bp |
| six months | 55 bp |
| one year | 150 bp |
| two years | over 400 bp |

At the money it stays within a few basis points while `nu^2 T < 0.3`.

![Where Hagan's formula stops being a model](../images/sabr-density.png)

For a rates-style smile (`beta 0.5`, forward 3%), the density implied by Hagan's
volatilities is negative below 0.75% at five years and below 1.6% at ten: the smile is
not a distribution. At a 0.5% strike its volatilities part from the model's own, from
QuantLib's PDE, by 14 points at five years and 28 at ten.

## 4. SABR on the S&P 500

![SABR fitted to each S&P 500 expiry](../images/sabr-spx.png)

With `beta = 1`, fitted expiry by expiry, SABR matches SVI's error beyond two months. It
misses the shortest expiries by about a point, where SVI misses by 0.2 to 0.4.

Its parameters move with maturity:

| Parameter | Shortest expiry | Three years |
|---|---|---|
| vol-of-vol | over 3 | 0.5 |
| correlation | -0.35 | -0.67 |

A single SABR model has one vol-of-vol. It is a smile model, not a surface model.

## 5. Same smile today, different smiles tomorrow

![Same smile today, different smiles tomorrow](../images/smile-dynamics.png)

Local volatility is fixed in absolute strike. When the index rises, the at-the-money point
slides down the local-volatility skew, which is about twice the implied skew, because
implied volatility averages local volatility between spot and strike (Hagan et al.,
2002). The claim is measured here by re-solving the forward equation after the move
([ADR 0020](../adr/0020-smile-dynamics-are-computed-in-each-model.md)).

- **On a Heston surface**, the at-the-money volatility moves by **1.98 times** the implied
  skew per log-move.
- **On the S&P 500 three-month expiry** it moves by 2.2 times: from 24.6% after a 6% fall
  to 11.5% after a 3% rise. Beyond that, the local-volatility smile's convexity turns it
  back up.
- **SABR with `beta = 1`** barely moves the at-the-money volatility; the smile travels with
  the forward.

So the deltas of the at-the-money call differ:

| Delta | dC/dF |
|---|---|
| Black (sticky strike) | 0.514 |
| local volatility | 0.388 |
| SABR, alpha held | 0.633 |
| SABR, Bartlett (alpha regressed on F) | 0.396 |

Local volatility and Bartlett's SABR delta nearly agree, and both are well below Black's.
On a falling market a local-volatility hedge sells more of the index than a sticky-strike
one.

## Reproducing

```bash
volsurf gallery --only 6          # snapshot charts need volsurf chains fetch ^SPX
pytest tests/localvol             # flat surface, round trip, QuantLib, the exact reference, dynamics
volsurf validate                  # four rows, on a Heston surface and QuantLib
```
