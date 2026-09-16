"""Discrete delta-hedging laboratory: how much of an option's risk can a hedge remove?

A dealer sells one European call at its model value and hedges with the underlying,
rebalancing on a discrete schedule. The terminal, discounted P&L of the hedged book is
recorded path by path.

Two worlds are simulated:

* Black-Scholes world (geometric Brownian motion). Hedging with the true delta leaves
  only discretisation error, which shrinks like 1/sqrt(N) (Derman & Kamal, 1999):
      std(P&L) ~ sqrt(pi/4) * vega * sigma / sqrt(N)
* Heston world. Spot moves are hedgeable but variance shocks are not, so hedging error
  plateaus no matter how often one rebalances. Three hedge ratios are compared:
    - Black-Scholes delta at the option's implied volatility (market practice),
    - Heston delta dC/dS,
    - Heston minimum-variance delta dC/dS + (rho * sigma / S) * dC/dv, which also
      hedges the part of the variance shock that is correlated with spot
      (Bakshi, Cao & Chen, 1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import bsm_greeks, bsm_price
from .heston import HestonParams, heston_cos_sensitivities, heston_price_cos_spot
from .implied_vol import implied_vol_bsm
from .monte_carlo import simulate_gbm_paths, simulate_heston_paths


@dataclass(frozen=True)
class HedgeRun:
    world: str
    strategy: str
    rebalances: int
    premium: float
    pnl: np.ndarray


@dataclass
class HedgingExperiment:
    runs: list[HedgeRun]
    strike: float
    maturity: float
    notes: dict[str, float]

    def summary(self) -> pd.DataFrame:
        rows = []
        for run in self.runs:
            pnl = run.pnl
            tail = np.sort(pnl)[: max(1, int(0.05 * pnl.size))]
            rows.append(
                {
                    "world": run.world,
                    "strategy": run.strategy,
                    "rebalances": run.rebalances,
                    "premium": run.premium,
                    "mean_pnl": float(pnl.mean()),
                    "std_pnl": float(pnl.std(ddof=1)),
                    "std_pct_premium": float(100.0 * pnl.std(ddof=1) / run.premium),
                    "q05": float(np.quantile(pnl, 0.05)),
                    "expected_shortfall_05": float(tail.mean()),
                    "paths": int(pnl.size),
                }
            )
        return pd.DataFrame(rows)


def _hedge_pnl(
    paths: np.ndarray,
    deltas: np.ndarray,
    rebalance_every: int,
    maturity: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    premium: float,
) -> np.ndarray:
    """Self-financing short call + delta hedge. `deltas[:, j]` is the hedge ratio at fine step j."""
    n_steps = paths.shape[1] - 1
    dt = maturity / n_steps
    growth = np.exp(rate * dt)

    held = deltas[:, 0].copy()
    cash = premium - held * paths[:, 0]
    for j in range(1, n_steps):
        cash = cash * growth + held * paths[:, j - 1] * dividend_yield * dt
        if j % rebalance_every == 0:
            new = deltas[:, j]
            cash -= (new - held) * paths[:, j]
            held = new
    cash = cash * growth + held * paths[:, -2] * dividend_yield * dt
    terminal = cash + held * paths[:, -1] - np.maximum(paths[:, -1] - strike, 0.0)
    return terminal * np.exp(-rate * maturity)


def run_black_scholes_world(
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 0.25,
    rate: float = 0.03,
    dividend_yield: float = 0.0,
    vol: float = 0.2,
    rebalances: tuple[int, ...] = (4, 16, 64, 256),
    n_paths: int = 20_000,
    seed: int = 1,
) -> HedgingExperiment:
    n_steps = max(rebalances)
    if any(n_steps % r for r in rebalances):
        raise ValueError("Every rebalance count must divide the finest one")
    paths = simulate_gbm_paths(spot, maturity, rate, dividend_yield, vol, n_steps, n_paths, seed)
    tau = maturity - np.linspace(0.0, maturity, n_steps + 1)[:-1]
    deltas = bsm_greeks(paths[:, :-1], strike, tau[None, :], rate, dividend_yield, vol, True).delta
    premium = float(bsm_price(spot, strike, maturity, rate, dividend_yield, vol, True))
    vega = float(bsm_greeks(spot, strike, maturity, rate, dividend_yield, vol, True).vega)

    runs = [
        HedgeRun(
            "Black-Scholes",
            "Black-Scholes delta",
            r,
            premium,
            _hedge_pnl(paths, deltas, n_steps // r, maturity, strike, rate, dividend_yield, premium),
        )
        for r in rebalances
    ]
    notes = {f"derman_kamal_std_{r}": float(np.sqrt(np.pi / 4.0) * vega * vol / np.sqrt(r)) for r in rebalances}
    return HedgingExperiment(runs, strike, maturity, notes)


def run_heston_world(
    params: HestonParams,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 0.25,
    rate: float = 0.03,
    dividend_yield: float = 0.0,
    rebalances: tuple[int, ...] = (3, 9, 21, 63),
    n_paths: int = 8_000,
    seed: int = 2,
) -> HedgingExperiment:
    n_steps = max(rebalances)
    if any(n_steps % r for r in rebalances):
        raise ValueError("Every rebalance count must divide the finest one")
    sim = simulate_heston_paths(spot, maturity, rate, dividend_yield, params, n_steps, n_paths, seed)
    paths, variance = sim.spot, sim.variance
    tau = maturity - sim.times[:-1]

    premium = float(heston_price_cos_spot(spot, [strike], maturity, rate, dividend_yield, params, True)[0])
    implied = float(implied_vol_bsm(premium, spot, strike, maturity, rate, dividend_yield, True))

    bs_delta = bsm_greeks(paths[:, :-1], strike, tau[None, :], rate, dividend_yield, implied, True).delta
    heston_delta = np.empty_like(bs_delta)
    mv_delta = np.empty_like(bs_delta)
    for j in range(n_steps):
        sens = heston_cos_sensitivities(
            paths[:, j],
            variance[:, j],
            strike,
            float(tau[j]),
            rate,
            dividend_yield,
            params.kappa,
            params.theta,
            params.sigma,
            params.rho,
            True,
        )
        heston_delta[:, j] = sens.delta
        mv_delta[:, j] = sens.delta + params.rho * params.sigma / paths[:, j] * sens.dprice_dv

    strategies = {
        "Black-Scholes delta @ implied vol": bs_delta,
        "Heston delta": np.clip(heston_delta, 0.0, 1.0),
        "Heston minimum-variance delta": np.clip(mv_delta, -0.5, 1.5),
    }
    runs = [
        HedgeRun(
            "Heston", name, r, premium, _hedge_pnl(paths, deltas, n_steps // r, maturity, strike, rate, dividend_yield, premium)
        )
        for name, deltas in strategies.items()
        for r in rebalances
    ]
    unhedged = np.exp(-rate * maturity) * (-np.maximum(paths[:, -1] - strike, 0.0)) + premium
    runs.append(HedgeRun("Heston", "Unhedged", 0, premium, unhedged))
    return HedgingExperiment(runs, strike, maturity, {"implied_vol": implied})
