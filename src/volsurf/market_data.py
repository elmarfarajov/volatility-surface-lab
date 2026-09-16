"""Option-chain ingestion, cleaning and model-free forward extraction.

Raw chains (live from Yahoo Finance or synthetic) share one schema. ``build_snapshot``
turns them into analysis-ready expiries:

1. Quote hygiene: positive bids, crossed/locked markets removed, relative-spread filter.
2. Forward and discount factor per expiry from put-call parity,
       C(K) - P(K) = D * (F - K),
   estimated by weighted least squares of (C - P) on K with iterative outlier removal.
   This needs no assumption about interest rates or dividends: the market's own
   prices reveal the carry, including discrete dividends.
3. Out-of-the-money selection (puts below the forward, calls above) and implied
   volatilities at bid, mid and ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .black_scholes import black76_price, black76_vega
from .heston import HestonParams, heston_price_cos
from .implied_vol import implied_vol_black76

RAW_COLUMNS = ["expiry", "maturity", "strike", "option_type", "bid", "ask", "last", "volume", "open_interest"]
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class ExpiryData:
    expiry: str
    maturity: float
    forward: float
    discount: float
    parity_pairs: int
    forward_method: str
    quotes: pd.DataFrame

    @property
    def implied_rate(self) -> float:
        return -np.log(self.discount) / self.maturity

    def atm_vol(self) -> float:
        q = self.quotes
        idx = np.argsort(np.abs(q["k"].to_numpy()))[:4]
        return float(q["iv_mid"].to_numpy()[idx].mean())


@dataclass
class MarketSnapshot:
    ticker: str
    spot: float
    as_of: datetime
    source: str
    expiries: list[ExpiryData]

    def quotes(self) -> pd.DataFrame:
        frames = []
        for e in self.expiries:
            frame = e.quotes.copy()
            frame["expiry"] = e.expiry
            frame["maturity"] = e.maturity
            frame["forward"] = e.forward
            frame["discount"] = e.discount
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def term_structure(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "expiry": [e.expiry for e in self.expiries],
                "maturity": [e.maturity for e in self.expiries],
                "forward": [e.forward for e in self.expiries],
                "discount": [e.discount for e in self.expiries],
                "implied_rate": [e.implied_rate for e in self.expiries],
                "implied_carry": [np.log(e.forward / self.spot) / e.maturity for e in self.expiries],
                "atm_vol": [e.atm_vol() for e in self.expiries],
                "n_quotes": [len(e.quotes) for e in self.expiries],
                "parity_pairs": [e.parity_pairs for e in self.expiries],
                "forward_method": [e.forward_method for e in self.expiries],
            }
        )


def estimate_forward_discount(
    strikes: np.ndarray,
    call_mid: np.ndarray,
    put_mid: np.ndarray,
    spreads: np.ndarray,
    spot: float,
    maturity: float,
    fallback_rate: float = 0.04,
) -> tuple[float, float, int, str]:
    """Weighted least squares of C - P on K near the money, with 3-MAD outlier rejection."""
    fallback = (spot * np.exp(fallback_rate * maturity), np.exp(-fallback_rate * maturity), 0, "fallback")
    for band in (0.1, 0.2, 0.35):
        sel = np.abs(strikes / spot - 1.0) <= band
        if sel.sum() >= 5:
            break
    else:
        return fallback

    k, y = strikes[sel], (call_mid - put_mid)[sel]
    weights = 1.0 / np.maximum(spreads[sel], 1e-4)
    keep = np.ones(k.size, dtype=bool)
    for _ in range(3):
        design = np.column_stack([np.ones(keep.sum()), k[keep]])
        sw = np.sqrt(weights[keep])
        (intercept, slope), *_ = np.linalg.lstsq(design * sw[:, None], y[keep] * sw, rcond=None)
        resid = y - (intercept + slope * k)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-12
        new_keep = np.abs(resid - np.median(resid[keep])) <= 3.0 * 1.4826 * mad
        if new_keep.sum() < 5 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    discount = -slope
    if not 0.5 < discount < 1.05:
        return fallback
    forward = intercept / discount
    if not 0.7 < forward / spot < 1.3:
        return fallback
    return float(forward), float(discount), int(keep.sum()), "put-call parity"


def build_snapshot(
    raw: pd.DataFrame,
    spot: float,
    as_of: datetime,
    ticker: str,
    source: str,
    max_relative_spread: float = 0.4,
    moneyness_bounds: tuple[float, float] = (0.55, 1.6),
    max_standardised_moneyness: float = 1.6,
    min_quotes: int = 8,
    fallback_rate: float = 0.04,
) -> MarketSnapshot:
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw chain is missing columns: {sorted(missing)}")

    df = raw.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_relative_spread]

    expiries: list[ExpiryData] = []
    for expiry, group in df.groupby("expiry", sort=True):
        maturity = float(group["maturity"].iloc[0])
        calls = group[group["option_type"] == "C"].set_index("strike")
        puts = group[group["option_type"] == "P"].set_index("strike")
        common = calls.index.intersection(puts.index)
        if len(common) < 5:
            continue
        spreads = (calls.loc[common, "ask"] - calls.loc[common, "bid"]) + (puts.loc[common, "ask"] - puts.loc[common, "bid"])
        forward, discount, n_pairs, method = estimate_forward_discount(
            common.to_numpy(float),
            calls.loc[common, "mid"].to_numpy(float),
            puts.loc[common, "mid"].to_numpy(float),
            spreads.to_numpy(float),
            spot,
            maturity,
            fallback_rate,
        )

        otm = pd.concat([puts[puts.index < forward], calls[calls.index >= forward]]).reset_index()
        strikes = otm["strike"].to_numpy(float)
        is_call = (otm["option_type"] == "C").to_numpy()
        k = np.log(strikes / forward)
        iv_mid = implied_vol_black76(otm["mid"].to_numpy(float), forward, strikes, maturity, discount, is_call)
        iv_bid = implied_vol_black76(otm["bid"].to_numpy(float), forward, strikes, maturity, discount, is_call)
        iv_ask = implied_vol_black76(otm["ask"].to_numpy(float), forward, strikes, maturity, discount, is_call)

        quotes = pd.DataFrame(
            {
                "strike": strikes,
                "k": k,
                "option_type": otm["option_type"].to_numpy(),
                "bid": otm["bid"].to_numpy(float),
                "ask": otm["ask"].to_numpy(float),
                "mid": otm["mid"].to_numpy(float),
                "volume": otm["volume"].to_numpy(float),
                "open_interest": otm["open_interest"].to_numpy(float),
                "iv_mid": iv_mid,
                "iv_bid": iv_bid,
                "iv_ask": iv_ask,
            }
        )
        m = strikes / forward
        keep = (
            np.isfinite(iv_mid)
            & (m >= moneyness_bounds[0])
            & (m <= moneyness_bounds[1])
            & (np.abs(k) / np.sqrt(maturity) <= max_standardised_moneyness)
            & (iv_mid > 0.01)
            & (iv_mid < 3.0)
        )
        quotes = quotes[keep].sort_values("strike").reset_index(drop=True)
        if len(quotes) < min_quotes:
            continue
        iv_spread = (quotes["iv_ask"] - quotes["iv_bid"]).fillna(0.05).clip(lower=0.002)
        quotes["weight"] = 1.0 / iv_spread
        quotes["vega"] = black76_vega(forward, quotes["strike"].to_numpy(), maturity, discount, quotes["iv_mid"].to_numpy())
        expiries.append(ExpiryData(str(expiry), maturity, forward, discount, n_pairs, method, quotes))

    if not expiries:
        raise ValueError("No expiry survived cleaning; the chain may be stale or the market closed without quotes")
    expiries.sort(key=lambda e: e.maturity)
    return MarketSnapshot(ticker=ticker, spot=float(spot), as_of=as_of, source=source, expiries=expiries)


def synthetic_chain(
    params: HestonParams,
    spot: float = 5000.0,
    rate: float = 0.04,
    dividend_yield: float = 0.013,
    maturities_days: tuple[int, ...] = (9, 23, 44, 72, 107, 170, 261, 352, 534, 716),
    strikes_per_expiry: int = 41,
    strike_step: float = 5.0,
    iv_noise: float = 0.002,
    half_spread_vol: float = 0.004,
    seed: int = 7,
) -> pd.DataFrame:
    """A realistic option chain generated from a known Heston model: strikes on an exchange
    grid, bid/ask built in volatility space, and small idiosyncratic quote noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for days in maturities_days:
        maturity = days / 365.0
        forward = spot * np.exp((rate - dividend_yield) * maturity)
        discount = np.exp(-rate * maturity)
        width = 2.2 * np.sqrt(params.theta) * np.sqrt(maturity) + 0.04
        strikes = np.unique(
            np.round(forward * np.exp(np.linspace(-1.6 * width, 0.9 * width, strikes_per_expiry)) / strike_step) * strike_step
        )
        call = heston_price_cos(forward, strikes, maturity, discount, params, True)
        true_iv = implied_vol_black76(call, forward, strikes, maturity, discount, True)
        noisy_iv = true_iv + iv_noise * rng.standard_normal(strikes.size)
        expiry = f"T+{days:03d}d"
        for is_call, flag in ((True, "C"), (False, "P")):
            bid = black76_price(forward, strikes, maturity, discount, np.maximum(noisy_iv - half_spread_vol, 1e-4), is_call)
            ask = black76_price(forward, strikes, maturity, discount, noisy_iv + half_spread_vol, is_call)
            bid = np.floor(bid * 20.0) / 20.0
            ask = np.ceil(ask * 20.0) / 20.0
            for i, strike in enumerate(strikes):
                rows.append(
                    {
                        "expiry": expiry,
                        "maturity": maturity,
                        "strike": float(strike),
                        "option_type": flag,
                        "bid": float(bid[i]),
                        "ask": float(ask[i]),
                        "last": float(0.5 * (bid[i] + ask[i])),
                        "volume": float(rng.integers(0, 5000)),
                        "open_interest": float(rng.integers(100, 50000)),
                    }
                )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


_ROOT_PATTERN = re.compile(r"^([A-Z]+)\d")
_TARGET_DAYS = (7, 14, 30, 45, 60, 91, 122, 182, 273, 365, 547, 730)


def _expiry_timestamp(expiry: str, root: str) -> datetime:
    settle = time(9, 30) if root == "SPX" else time(16, 0)
    local = datetime.combine(datetime.strptime(expiry, "%Y-%m-%d").date(), settle, tzinfo=_NEW_YORK)
    return local.astimezone(timezone.utc)


def select_expiries(expiries: list[str], as_of: datetime, max_expiries: int, min_days: int = 5, max_days: int = 800) -> list[str]:
    days = {e: (datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) - as_of).days for e in expiries}
    eligible = [e for e in expiries if min_days <= days[e] <= max_days]
    chosen: list[str] = []
    for target in _TARGET_DAYS:
        if not eligible or len(chosen) >= max_expiries:
            break
        best = min(eligible, key=lambda e: abs(days[e] - target))
        if best not in chosen:
            chosen.append(best)
    return sorted(chosen)


def normalise_yfinance_chain(calls: pd.DataFrame, puts: pd.DataFrame, expiry: str, as_of: datetime) -> pd.DataFrame:
    frames = []
    for frame, flag in ((calls, "C"), (puts, "P")):
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        f["root"] = f["contractSymbol"].str.extract(_ROOT_PATTERN, expand=False)
        f["option_type"] = flag
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    chain = pd.concat(frames, ignore_index=True)

    root = chain["root"].mode().iloc[0]
    chain = chain[chain["root"] == root]
    chain = chain.sort_values(["openInterest", "volume"], ascending=False, na_position="last")
    chain = chain.drop_duplicates(subset=["strike", "option_type"], keep="first")

    maturity = (_expiry_timestamp(expiry, root) - as_of).total_seconds() / _SECONDS_PER_YEAR
    return pd.DataFrame(
        {
            "expiry": expiry,
            "maturity": max(maturity, 1.0 / 365.0),
            "strike": chain["strike"].astype(float),
            "option_type": chain["option_type"],
            "bid": chain["bid"].astype(float).fillna(0.0),
            "ask": chain["ask"].astype(float).fillna(0.0),
            "last": chain["lastPrice"].astype(float),
            "volume": chain["volume"].astype(float).fillna(0.0),
            "open_interest": chain["openInterest"].astype(float).fillna(0.0),
        }
    ).reset_index(drop=True)


def fetch_yfinance_chain(ticker: str, max_expiries: int = 8, ticker_factory=None) -> tuple[pd.DataFrame, float, datetime]:
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker
    handle = ticker_factory(ticker)
    as_of = datetime.now(timezone.utc)
    history = handle.history(period="5d")
    if history is None or history.empty:
        raise RuntimeError(f"No price history returned for {ticker}")
    spot = float(history["Close"].iloc[-1])

    frames = []
    for expiry in select_expiries(list(handle.options), as_of, max_expiries):
        chain = handle.option_chain(expiry)
        frames.append(normalise_yfinance_chain(chain.calls, chain.puts, expiry, as_of))
    if not frames:
        raise RuntimeError(f"No listed expiries found for {ticker}")
    return pd.concat(frames, ignore_index=True), spot, as_of
