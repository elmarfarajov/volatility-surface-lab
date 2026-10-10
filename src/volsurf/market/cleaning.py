"""From a raw chain to clean implied volatilities: every quote kept or dropped for a stated reason.

Each expiry (an expiry date and contract root - SPX's AM-settled monthlies and its
PM-settled weeklies on the same date are different contracts) goes through:

1. **parity**: the forward and discount factor from :func:`~volsurf.market.parity.fit_forward`,
   then one discount curve across expiries and each forward refitted on it;
2. **quote rules**, in order, each recording what it removes:

   ======================  =================================================================
   no bid                  nobody is buying; the ask alone is not a price
   crossed or locked       ask at or below bid - a stale or erroneous quote
   no open interest        never traded and held by nobody: an auto-quote
   wide spread             ask - bid more than ``max_spread`` of the mid
   in the money            the out-of-the-money twin carries the same information, better
   outside price bounds    the mid at or beyond the static bounds has no implied volatility
   strike arbitrage        a quote that, at bid and ask, lets a call spread or butterfly be traded for a credit
   no implied volatility   the inversion found none at the mid
   ======================  =================================================================

3. **implied volatility** at the bid, mid and ask with :func:`~volsurf.market.implied.implied_vol`.

Strike arbitrage is tested on the call-equivalent curve - calls as quoted, puts
through parity - at the bid and the ask (:func:`executable_violations`): only a
pattern that can be traded at the quotes is arbitrage; a mid-price wiggle inside
the spread is noise, left for the surface fit (:func:`mid_violations` counts it,
with :func:`volsurf.analytic.strike_violations`). Quotes are removed one at a time,
each round the one whose removal leaves the least arbitrage, so one stale quote
does not take its honest neighbours with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analytic.bounds import strike_violations
from .chains import Chain
from .implied import implied_vol
from .parity import DiscountCurve, ParityFit, fit_curve, fit_forward

RULES = (
    "no bid",
    "crossed or locked",
    "no open interest",
    "wide spread",
    "in the money",
    "outside price bounds",
    "strike arbitrage",
    "no implied volatility",
)


@dataclass(frozen=True)
class Slice:
    """One expiry after cleaning: the parity fits, the quotes kept, and why the rest went."""

    expiry: str
    root: str
    maturity: float
    free_fit: ParityFit  # forward and discount both fitted
    fit: ParityFit  # forward refitted with the discount from the curve
    quotes: pd.DataFrame = field(repr=False)  # strike, k, is_call, bid, ask, mid, iv_bid, iv_mid, iv_ask, vendor_iv, ...
    funnel: dict[str, int] = field(repr=False)  # quotes removed by each rule, in order; "raw" and "kept" bracket them

    @property
    def forward(self) -> float:
        return self.fit.forward

    @property
    def discount(self) -> float:
        return self.fit.discount


@dataclass(frozen=True)
class Market:
    """A cleaned chain: the slices, the discount curve, and the expiries that could not be used."""

    chain: Chain
    curve: DiscountCurve
    slices: list[Slice]
    skipped: dict[str, str]  # "expiry root" -> why

    def funnel(self) -> pd.Series:
        """Quotes removed by each rule, summed over the slices."""
        total = pd.DataFrame([s.funnel for s in self.slices]).sum()
        return total.astype(int)

    def term_structure(self) -> pd.DataFrame:
        rows = []
        for s in self.slices:
            rows.append(
                {
                    "expiry": s.expiry,
                    "root": s.root,
                    "maturity": s.maturity,
                    "forward": s.forward,
                    "forward_se": s.fit.forward_se,
                    "discount": s.discount,
                    "free_rate": s.free_fit.rate,
                    "free_rate_se": s.free_fit.discount_se / (s.free_fit.discount * s.maturity),
                    "curve_rate": float(self.curve.rate(s.maturity)),
                    "carry": s.fit.carry,
                    "pairs": s.free_fit.pairs,
                    "inside": s.free_fit.inside,
                    "quotes": len(s.quotes),
                }
            )
        return pd.DataFrame(rows)


def _pairs(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    calls = group[group["is_call"]].drop_duplicates("strike").set_index("strike")
    puts = group[~group["is_call"]].drop_duplicates("strike").set_index("strike")
    return calls, puts, calls.index.intersection(puts.index).sort_values()


def _parity(group: pd.DataFrame, spot: float, maturity: float, discount: float | None, band: float | None) -> ParityFit:
    calls, puts, common = _pairs(group)
    return fit_forward(
        common.to_numpy(float),
        calls.loc[common, "bid"].to_numpy(float),
        calls.loc[common, "ask"].to_numpy(float),
        puts.loc[common, "bid"].to_numpy(float),
        puts.loc[common, "ask"].to_numpy(float),
        spot,
        maturity,
        band=band,
        discount=discount,
    )


def executable_violations(
    strikes: np.ndarray, bid: np.ndarray, ask: np.ndarray, discount: float
) -> list[tuple[tuple[int, ...], float]]:
    """Strike arbitrages that can be traded at the quotes, as (indices involved, profit per unit strike step).

    With call-equivalent quotes sorted by strike:

    * a call spread bought at ``ask_i`` and sold at ``bid_{i+1}`` for a credit (prices rising in strike);
    * a call spread sold at ``bid_i``, bought at ``ask_{i+1}``, for more than ``D (K_{i+1} - K_i)`` (too steep);
    * a butterfly - wings bought at the ask, body sold at the bid - for a credit (not convex).

    Violations at mid prices that vanish within the spread are noise, not arbitrage, and are not reported.
    """
    found: list[tuple[tuple[int, ...], float]] = []
    gap = np.diff(strikes)
    rising = bid[1:] - ask[:-1]
    steep = bid[:-1] - ask[1:] - discount * gap
    for i in np.flatnonzero(rising > 0):
        found.append(((int(i), int(i) + 1), float(rising[i])))
    for i in np.flatnonzero(steep > 0):
        found.append(((int(i), int(i) + 1), float(steep[i])))
    left, right = gap[:-1], gap[1:]
    butterfly = ask[:-2] * right - bid[1:-1] * (left + right) + ask[2:] * left
    for i in np.flatnonzero(butterfly < 0):
        found.append(((int(i), int(i) + 1, int(i) + 2), float(-butterfly[i] / (left[i] + right[i]))))
    return found


def strike_arbitrage(strikes: np.ndarray, bid: np.ndarray, ask: np.ndarray, discount: float, max_rounds: int = 500) -> np.ndarray:
    """Indices to drop until no executable arbitrage is left.

    Each round tries removing every quote involved in a violation and drops the one that
    leaves the least arbitrage behind. Blaming the quote that appears in the most
    violations is not enough: a stale quote and its honest neighbour share every
    butterfly, and removing the neighbour only makes the next one the wing.
    """
    alive = np.ones(strikes.size, dtype=bool)

    def remaining(mask: np.ndarray) -> float:
        idx = np.flatnonzero(mask)
        return sum(size for _, size in executable_violations(strikes[idx], bid[idx], ask[idx], discount))

    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        found = executable_violations(strikes[idx], bid[idx], ask[idx], discount)
        if not found:
            break
        suspects = sorted({idx[i] for members, _ in found for i in members})
        best, best_left = suspects[0], np.inf
        for suspect in suspects:
            trial = alive.copy()
            trial[suspect] = False
            left = remaining(trial)
            if left < best_left:
                best, best_left = suspect, left
        alive[best] = False
    return np.flatnonzero(~alive)


def mid_violations(quotes: pd.DataFrame, forward: float, discount: float) -> int:
    """Strike-arbitrage violations left at the mid prices of cleaned quotes (all within the spread)."""
    k = quotes["strike"].to_numpy(float)
    calls = np.where(quotes["is_call"], quotes["mid"], quotes["mid"] + discount * (forward - k))
    return len(strike_violations(k, calls.astype(float), discount, tolerance=1e-9 * max(float(np.max(calls)), 1.0)))


def clean_slice(
    group: pd.DataFrame,
    fit: ParityFit,
    max_spread: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the quote rules to one expiry's raw quotes, given its forward and discount factor."""
    q = group.copy()
    funnel: dict[str, int] = {"raw": len(q)}

    def drop(rule: str, mask: pd.Series | np.ndarray) -> None:
        nonlocal q
        mask = np.asarray(mask, dtype=bool)
        funnel[rule] = int(mask.sum())
        q = q[~mask]

    drop("no bid", q["bid"] <= 0)
    drop("crossed or locked", q["ask"] <= q["bid"])
    drop("no open interest", (q["open_interest"] <= 0) & (q["volume"] <= 0))
    mid = 0.5 * (q["bid"] + q["ask"])
    drop("wide spread", (q["ask"] - q["bid"]) > max_spread * mid)
    f, d = fit.forward, fit.discount
    drop("in the money", np.where(q["is_call"], q["strike"] < f, q["strike"] >= f))
    q = q.assign(mid=0.5 * (q["bid"] + q["ask"]))
    upper = np.where(q["is_call"], d * f, d * q["strike"])
    drop("outside price bounds", (q["mid"] <= 0) | (q["mid"] >= upper))

    q = q.sort_values("strike").reset_index(drop=True)
    k = q["strike"].to_numpy(float)
    shift = np.where(q["is_call"], 0.0, d * (f - k))  # puts to call equivalents through parity
    bad = np.zeros(len(q), dtype=bool)
    bad[strike_arbitrage(k, (q["bid"] + shift).to_numpy(float), (q["ask"] + shift).to_numpy(float), d)] = True
    drop("strike arbitrage", bad)

    k, call = q["strike"].to_numpy(float), q["is_call"].to_numpy(bool)
    t = float(q["maturity"].iloc[0]) if len(q) else 1.0
    q = q.assign(
        k=np.log(k / f),
        iv_bid=implied_vol(q["bid"].to_numpy(float), f, k, t, d, call),
        iv_mid=implied_vol(q["mid"].to_numpy(float), f, k, t, d, call),
        iv_ask=implied_vol(q["ask"].to_numpy(float), f, k, t, d, call),
    )
    drop("no implied volatility", ~np.isfinite(q["iv_mid"]))
    funnel["kept"] = len(q)
    return q.reset_index(drop=True), funnel


def prepare(
    chain: Chain,
    american: bool = False,
    band: float = 0.05,
    max_spread: float = 0.5,
    min_quotes: int = 8,
    curve: DiscountCurve | None = None,
) -> Market:
    """Clean every expiry of a chain.

    European chains (index options) fit forward and discount per expiry and one curve
    across them. For ``american`` chains parity holds only near the money, so the free
    fit is restricted to ``|ln(K/S)| <= band``, and the curve - best taken from a European
    market on the same day, via ``curve`` - supplies the discount factors.
    """
    groups = list(chain.quotes.groupby(["expiry", "root"], sort=False))
    free: dict[tuple[str, str], ParityFit] = {}
    skipped: dict[str, str] = {}
    for (expiry, root), group in groups:
        try:
            free[(expiry, root)] = _parity(group, chain.spot, float(group["maturity"].iloc[0]), None, band if american else None)
        except ValueError as error:
            skipped[f"{expiry} {root}"] = str(error)
    curve = curve or fit_curve(list(free.values()))
    slices: list[Slice] = []
    for (expiry, root), group in groups:
        if (expiry, root) not in free:
            continue
        maturity = float(group["maturity"].iloc[0])
        fit = _parity(group, chain.spot, maturity, float(curve.discount(maturity)), band if american else None)
        quotes, funnel = clean_slice(group, fit, max_spread)
        if len(quotes) < min_quotes:
            skipped[f"{expiry} {root}"] = f"{len(quotes)} quotes survived cleaning; need {min_quotes}"
            continue
        slices.append(Slice(str(expiry), str(root), maturity, free[(expiry, root)], fit, quotes, funnel))
    slices.sort(key=lambda s: (s.maturity, s.root))
    return Market(chain, curve, slices, skipped)
