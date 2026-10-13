# 18. Local volatility comes in closed form from the price interpolation, with a self-similar short end

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Dupire's local variance is twice the time derivative of the call over its density. The
v1.0 `VolSurface.local_vol` took both by finite differences of a surface interpolated in
total variance. It returned NaN on 4% of the S&P 500 grid, wherever the density came out
non-positive.

The Day 4 surface interpolates prices instead, which makes both quantities exact. Three
consequences of that interpolation appeared only when its local volatility was used.

- **Before the first expiry**, the interpolation mixes the first slice with expiry itself.
  The distribution then keeps an atom at the forward, which no diffusion can carry. Local
  variance grows like `1/t` at the forward, and a forward Dupire solve from `t = 0`
  stopped converging as its grid was refined.
- **Between daily weeklies**, any disagreement between neighbouring slices is divided by a
  small time step. On 9 October 2026, local volatility at `ln(K/F) = -0.1` swung from 8%
  to over 200% within two weeks.
- **A mixture of two log-normal laws is not log-normal.** Between the expiries of a flat
  20% surface, the interpolation puts a smile of 2.8 volatility points at
  `ln(K/F) = -0.4`.

## Decision

- **Local variance comes from the price interpolation in closed form:**

      sigma_loc^2 = -2 alpha'(t) (c_hi - c_lo) / (e^k (alpha p_lo + (1 - alpha) p_hi)),

  with the out-of-the-money prices and the slices' own densities. It is positive wherever
  the density is, by construction.
- **Before the first expiry, the short end is self-similar**: `w(k, t) = (t/t_1) w_1(k)`,
  with Dupire in total variance. It reaches the first slice exactly and is a diffusion's.
  The interpolation's own short end remains available, for comparison.
- **Beyond the last expiry, local variance is the convolution's rate**, `theta_n / t_n`.
- **Dynamics are computed on monthly expiries**, from 25 days, each at least 25% longer
  than the last.
- **Every local-volatility result is checked** by solving the forward Dupire equation and
  comparing with the surface.

## Consequences

- On the monthly S&P 500 surface, the forward equation reprices every expiry to within
  3.6 basis points of volatility, and to within 1 basis point beyond three months. On a
  Heston surface the worst error is 2.5 basis points at a coarse grid.
- With every weekly expiry included, the error stays at 20 to 40 basis points however fine
  the grid. That is the noise, not the solver.
- Local volatility is piecewise smooth in time, with a jump at each expiry where `theta'`
  jumps. It is shown, not smoothed.
