"""Thirty-six years of the S&P 500 and the VIX: what selling and hedging options actually earned, "
"and how volatility moves with the index.

**Data.** Daily closes of the S&P 500 index (``^GSPC``), the VIX and VIX3M (30- and
91-day implied volatility, Cboe), and the 13-week Treasury bill (``^IRX``), from Yahoo
Finance. They are stored locally (``volsurf history fetch``) and never committed.

**The backtest.** On every trading day a 30-day at-the-money straddle on the index is sold
at the VIX and delta-hedged at the close each day, at the volatility it was sold at, until
expiry. The VIX is a variance-swap rate, not an at-the-money volatility; on the S&P 500 it
sits above the at-the-money volatility by the value of the skew, so the premium here is an
upper bound on what such a straddle fetched, and the P&L is a variance-swap P&L more than a
straddle's. Dividends are a constant 1.5% yield. The P&L is checked against the gamma
formula of :mod:`volsurf.risk.hedging` on the real paths.

**Spot-volatility dynamics.** The VIX moves against the index. Regressing its daily change
on the index's log-return gives ``d sigma / d ln S``. Compared with the slope of the smile,
it says which smile dynamics the market follows: sticky moneyness (ratio 0), sticky strike
(1), or local volatility (about 2, Day 6).

**The minimum-variance delta.** Hull and White (2017) add to the Black-Scholes delta the
vega times the expected volatility move per unit spot move. Here that move is the regression
slope, estimated only on data before each trade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..analytic.generalized import greeks
from ..market.chains import data_root

Array = NDArray[np.float64]
TICKERS = {"spot": "^GSPC", "vix": "^VIX", "vix3m": "^VIX3M", "bill": "^IRX"}
DIVIDEND_YIELD = 0.015


def history_root() -> Path:
    return data_root().parent / "history"


def available(root: Path | None = None) -> bool:
    base = root or history_root()
    return all((base / f"{name}.csv").exists() for name in TICKERS)


def fetch(root: Path | None = None, start: str = "1990-01-01") -> Path:  # pragma: no cover - network
    """Download the daily closes, up to the last complete day, and store them locally."""
    import yfinance as yf

    base = root or history_root()
    base.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    for name, ticker in TICKERS.items():
        frame = yf.Ticker(ticker).history(start=start, end=today, auto_adjust=False)
        if frame is None or frame.empty:
            raise RuntimeError(f"no history for {ticker}")
        out = pd.DataFrame({"date": frame.index.date, "close": frame["Close"].to_numpy(float)})
        out.to_csv(base / f"{name}.csv", index=False)
    (base / "meta.json").write_text(
        json.dumps({"fetched": today, "source": "Yahoo Finance", "tickers": TICKERS}, indent=2), encoding="utf-8"
    )
    return base


def load(root: Path | None = None) -> pd.DataFrame:
    """Aligned daily series: index level, VIX and VIX3M as decimals, and the bill rate continuously compounded."""
    base = root or history_root()
    series = {}
    for name in TICKERS:
        frame = pd.read_csv(base / f"{name}.csv", parse_dates=["date"]).set_index("date")["close"]
        series[name] = frame[~frame.index.duplicated()]
    out = pd.DataFrame(series)
    out = out.dropna(subset=["spot", "vix", "bill"])
    out["vix"] /= 100.0
    out["vix3m"] /= 100.0
    bill = out["bill"] / 100.0
    out["rate"] = -np.log(1.0 - bill * 91.0 / 360.0) / (91.0 / 365.0)  # the bill's discount yield, continuously compounded
    return out.drop(columns="bill")


# ---------------------------------------------------------------- the backtest


@dataclass(frozen=True)
class Trades:
    table: pd.DataFrame  # start, vix, premium, pnl, predicted, realised, worst move ... one row per trade
    days: int


def _straddle(spot: Array, strike: float, tau: Array, r: float, sigma: float) -> tuple[Array, Array, Array]:
    call = greeks(spot, strike, tau, r, DIVIDEND_YIELD, sigma, True)
    put = greeks(spot, strike, tau, r, DIVIDEND_YIELD, sigma, False)
    return call.price + put.price, call.delta + put.delta, call.gamma + put.gamma


def backtest(history: pd.DataFrame, calendar_days: int = 30, every: int = 1) -> Trades:
    """Sell a ``calendar_days`` at-the-money straddle at the VIX each ``every`` trading days and delta-hedge it daily.

    P&L is per unit of the index level at the sale, discounted to the sale date.
    """
    dates = history.index.to_numpy()
    spot_all = history["spot"].to_numpy(float)
    vix_all = history["vix"].to_numpy(float)
    rate_all = history["rate"].to_numpy(float)
    rows = []
    q = DIVIDEND_YIELD
    for i in range(0, len(dates), every):
        expiry = dates[i] + np.timedelta64(calendar_days, "D")
        j = int(np.searchsorted(dates, expiry, side="right")) - 1  # the last close on or before expiry
        if j <= i + 2 or j >= len(dates) or dates[-1] < expiry:
            continue
        s = spot_all[i : j + 1] / spot_all[i]
        t = (dates[i : j + 1] - dates[i]) / np.timedelta64(1, "D") / 365.0
        big_t = float(t[-1])  # the straddle expires at the last close on or before the 30th day
        r, sigma = float(rate_all[i]), float(vix_all[i])
        tau = big_t - t
        value, delta, gamma = _straddle(s[:-1], 1.0, tau[:-1], r, sigma)
        premium = float(value[0])
        cash = premium - delta[0] * s[0]
        for k in range(1, len(s) - 1):
            dt = t[k] - t[k - 1]
            cash = cash * np.exp(r * dt) + delta[k - 1] * s[k] * (np.exp(q * dt) - 1.0)
            cash -= (delta[k] - delta[k - 1]) * s[k]
        dt_last = big_t - t[-2]
        cash = cash * np.exp(r * dt_last) + delta[-1] * s[-1] * (np.exp(q * dt_last) - 1.0)
        pnl = (cash + delta[-1] * s[-1] - abs(s[-1] - 1.0)) * np.exp(-r * big_t)
        ret = np.diff(s) / s[:-1] - (r - q) * np.diff(t)
        steps = np.diff(t)
        predicted = float(np.sum(0.5 * np.exp(-r * t[:-1]) * gamma * s[:-1] ** 2 * (sigma**2 * steps - ret**2)))
        realised = float(np.sqrt(np.sum(np.diff(np.log(s)) ** 2) / big_t))
        rows.append(
            {
                "start": pd.Timestamp(dates[i]),
                "vix": sigma,
                "premium": premium,
                "pnl": float(pnl),
                "predicted": predicted,
                "realised": realised,
                "move": float(s[-1] - 1.0),
            }
        )
    return Trades(pd.DataFrame(rows), calendar_days)


# ---------------------------------------------------------------- spot-volatility dynamics


@dataclass(frozen=True)
class SpotVol:
    beta: float  # d sigma / d ln S, VIX on daily log-returns
    beta3m: float  # the same for VIX3M (since 2006)
    r2: float
    term_ratio: float  # regression of daily VIX3M changes on VIX changes
    rolling: pd.Series  # one-year rolling beta


def spot_vol(history: pd.DataFrame, window: int = 252) -> SpotVol:
    ret = np.log(history["spot"]).diff()
    dv = history["vix"].diff()
    frame = pd.DataFrame({"ret": ret, "dv": dv, "dv3": history["vix3m"].diff()}).dropna(subset=["ret", "dv"])
    beta, alpha = np.polyfit(frame["ret"], frame["dv"], 1)
    r2 = float(np.corrcoef(frame["ret"], frame["dv"])[0, 1] ** 2)
    both = frame.dropna()
    beta3 = float(np.polyfit(both["ret"], both["dv3"], 1)[0])
    term = float(np.polyfit(both["dv"], both["dv3"], 1)[0])
    cov = frame["ret"].rolling(window).cov(frame["dv"])
    var = frame["ret"].rolling(window).var()
    return SpotVol(float(beta), beta3, r2, term, (cov / var).dropna())


def hedge_errors(history: pd.DataFrame, calendar_days: int = 30, warmup: int = 504) -> pd.DataFrame:
    """Daily hedging errors of a fixed-strike at-the-money call marked at the VIX, with the "
    "Black-Scholes and minimum-variance deltas.

    Each 30-day call is struck at the money on its first day and marked daily at the VIX
    (an at-the-money proxy). The minimum-variance delta adds ``vega beta / S``, with
    ``beta`` the spot-volatility slope estimated on all data before that day.
    """
    s = history["spot"].to_numpy(float)
    v = history["vix"].to_numpy(float)
    r = history["rate"].to_numpy(float)
    dates = history.index
    ret = np.diff(np.log(s))
    dv = np.diff(v)
    cum_xy = np.concatenate([[0.0], np.cumsum(ret * dv)])
    cum_xx = np.concatenate([[0.0], np.cumsum(ret * ret)])
    cum_x = np.concatenate([[0.0], np.cumsum(ret)])
    cum_y = np.concatenate([[0.0], np.cumsum(dv)])
    rows = []
    for i in range(warmup, len(s) - 1, 21):
        strike = s[i]
        end = dates[i] + pd.Timedelta(days=calendar_days)
        j = i
        while j + 1 < len(s) and dates[j + 1] <= end:
            n = j  # data up to day j (changes 0..j-1) are known when hedging on day j
            mean_x, mean_y = cum_x[n] / n, cum_y[n] / n
            beta = (cum_xy[n] / n - mean_x * mean_y) / (cum_xx[n] / n - mean_x**2)
            tau = (end - dates[j]).days / 365.0
            tau_next = (end - dates[j + 1]).days / 365.0
            g = greeks(s[j], strike, tau, r[j], DIVIDEND_YIELD, v[j], True)
            after = greeks(s[j + 1], strike, max(tau_next, 1e-6), r[j], DIVIDEND_YIELD, v[j + 1], True).price
            change = float(after - g.price)
            ds = s[j + 1] - s[j]
            rows.append(
                {
                    "date": dates[j],
                    "error_bs": change - float(g.delta) * ds,
                    "error_mv": change - (float(g.delta) + float(g.vega) * beta / s[j]) * ds,
                    "beta": beta,
                    "premium": float(g.price),
                }
            )
            j += 1
    return pd.DataFrame(rows)
