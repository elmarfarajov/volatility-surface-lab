"""Interactive dashboard: `streamlit run app/streamlit_app.py`."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsurf.black_scholes import bsm_greeks, bsm_price
from volsurf.calibration import calibrate_heston
from volsurf.figures import surface_grid
from volsurf.hedging import run_black_scholes_world, run_heston_world
from volsurf.heston import HestonParams, heston_price_cos_spot, heston_price_integration
from volsurf.implied_vol import implied_vol_bsm
from volsurf.lattice import binomial_price
from volsurf.market_data import build_snapshot, fetch_yfinance_chain, synthetic_chain
from volsurf.monte_carlo import mc_european_gbm
from volsurf.pipeline import fit_surface

st.set_page_config(page_title="Volatility Surface Lab", layout="wide")

SYNTHETIC_PARAMS = HestonParams(v0=0.028, kappa=2.5, theta=0.042, sigma=0.8, rho=-0.74)


@st.cache_data(show_spinner="Building option market ...")
def load_snapshot(source: str, ticker: str, expiries: int):
    if source == "Synthetic Heston market":
        raw = synthetic_chain(SYNTHETIC_PARAMS, maturities_days=(16, 44, 107, 261, 534))
        return build_snapshot(raw, 5000.0, datetime.now(timezone.utc), "SYNTHETIC", "synthetic")
    raw, spot, as_of = fetch_yfinance_chain(ticker, max_expiries=expiries)
    return build_snapshot(raw, spot, as_of, ticker, "Yahoo Finance (delayed)")


@st.cache_resource(show_spinner="Fitting arbitrage-controlled SVI surface ...")
def load_surface(source: str, ticker: str, expiries: int):
    return fit_surface(load_snapshot(source, ticker, expiries))


st.title("Volatility Surface Lab")
st.caption("Option pricing, arbitrage-free SVI surfaces, Heston calibration and a delta-hedging model-risk lab.")

with st.sidebar:
    st.header("Market data")
    source = st.radio("Source", ["Synthetic Heston market", "Live (Yahoo Finance)"], index=0)
    ticker = st.text_input("Ticker", "^SPX", disabled=source != "Live (Yahoo Finance)")
    n_expiries = st.slider("Expiries", 4, 10, 6, disabled=source != "Live (Yahoo Finance)")

pricing_tab, surface_tab, calibration_tab, hedging_tab = st.tabs(
    ["Pricing lab", "Volatility surface", "Heston calibration", "Hedging lab"]
)

with pricing_tab:
    c1, c2, c3 = st.columns(3)
    spot = c1.number_input("Spot", value=100.0, min_value=0.01)
    strike = c1.number_input("Strike", value=100.0, min_value=0.01)
    maturity = c2.number_input("Maturity (years)", value=0.5, min_value=0.01, max_value=10.0)
    vol = c2.number_input("Black-Scholes vol", value=0.2, min_value=0.01, max_value=3.0)
    rate = c3.number_input("Rate", value=0.04, step=0.005, format="%.3f")
    dividend = c3.number_input("Dividend yield", value=0.01, step=0.005, format="%.3f")
    is_call = st.radio("Type", ["Call", "Put"], horizontal=True) == "Call"

    greeks = bsm_greeks(spot, strike, maturity, rate, dividend, vol, is_call)
    mc = mc_european_gbm(spot, strike, maturity, rate, dividend, vol, is_call, 100_000, seed=1)
    engines = pd.DataFrame(
        [
            ("Black-Scholes analytic", float(bsm_price(spot, strike, maturity, rate, dividend, vol, is_call)), None),
            ("Leisen-Reimer tree (301)", binomial_price(spot, strike, maturity, rate, dividend, vol, is_call, steps=301), None),
            (
                "CRR tree (1000)",
                binomial_price(spot, strike, maturity, rate, dividend, vol, is_call, steps=1000, method="crr"),
                None,
            ),
            ("Monte Carlo + control variate", mc.price, mc.std_error),
            ("American (Leisen-Reimer)", binomial_price(spot, strike, maturity, rate, dividend, vol, is_call, True, 501), None),
        ],
        columns=["engine", "price", "std error"],
    )
    left, right = st.columns([1, 1])
    left.subheader("Black-Scholes engines")
    left.dataframe(engines, hide_index=True, use_container_width=True)
    left.dataframe(
        pd.DataFrame({k: [float(getattr(greeks, k))] for k in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga")}),
        hide_index=True,
        use_container_width=True,
    )

    right.subheader("Heston smile")
    h1, h2 = right.columns(2)
    v0 = h1.slider("v0", 0.005, 0.2, 0.04, 0.005)
    theta = h1.slider("theta", 0.005, 0.2, 0.04, 0.005)
    kappa = h1.slider("kappa", 0.1, 10.0, 2.0, 0.1)
    sigma = h2.slider("sigma (vol of vol)", 0.05, 2.0, 0.6, 0.05)
    rho = h2.slider("rho", -0.95, 0.5, -0.7, 0.05)
    params = HestonParams(v0, kappa, theta, sigma, rho)
    strikes = spot * np.exp(np.linspace(-0.4, 0.3, 60) * max(np.sqrt(maturity), 0.3))
    forward = spot * np.exp((rate - dividend) * maturity)
    prices = heston_price_cos_spot(spot, strikes, maturity, rate, dividend, params, strikes >= forward)
    smile = implied_vol_bsm(prices, spot, strikes, maturity, rate, dividend, strikes >= forward)
    fig = go.Figure(go.Scatter(x=strikes, y=100 * smile, mode="lines", name="Heston implied vol"))
    fig.update_layout(height=320, margin={"t": 10, "b": 10}, xaxis_title="strike", yaxis_title="implied vol (%)")
    right.plotly_chart(fig, use_container_width=True)
    heston_cos = float(heston_price_cos_spot(spot, [strike], maturity, rate, dividend, params, is_call)[0])
    heston_quad = heston_price_integration(forward, strike, maturity, np.exp(-rate * maturity), params, is_call)
    right.metric(
        "Heston price (COS)", f"{heston_cos:.6f}", delta=f"{heston_cos - heston_quad:+.1e} vs quadrature", delta_color="off"
    )
    right.caption(f"Feller ratio 2*kappa*theta/sigma^2 = {params.feller_ratio:.2f}")

try:
    snapshot = load_snapshot(source, ticker, n_expiries)
    fits, surface = load_surface(source, ticker, n_expiries)
    data_error = None
except Exception as exc:  # network or data problems should not take the whole app down
    snapshot = fits = surface = None
    data_error = exc

with surface_tab:
    if data_error is not None:
        st.error(f"Could not load market data: {data_error}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Spot", f"{snapshot.spot:,.2f}")
        m2.metric("Expiries / quotes", f"{len(snapshot.expiries)} / {sum(len(e.quotes) for e in snapshot.expiries)}")
        m3.metric("Worst SVI slice RMSE", f"{100 * max(f.rmse_vol for f in fits):.2f} vol pts")
        z, maturities, vol_grid = surface_grid(surface, snapshot)
        surface_fig = go.Figure(go.Surface(x=z, y=maturities, z=100 * vol_grid, colorscale="Viridis"))
        surface_fig.update_layout(
            height=520,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"xaxis_title": "ln(K/F)/sqrt(T)", "yaxis_title": "T (years)", "zaxis_title": "vol (%)"},
        )
        st.plotly_chart(surface_fig, use_container_width=True)

        labels = [f"{e.expiry} (T={e.maturity:.2f})" for e in snapshot.expiries]
        choice = st.selectbox("Smile", labels)
        idx = labels.index(choice)
        expiry, fit = snapshot.expiries[idx], fits[idx]
        grid = np.linspace(expiry.quotes["k"].min(), expiry.quotes["k"].max(), 150)
        smile_fig = go.Figure()
        smile_fig.add_trace(
            go.Scatter(x=expiry.quotes["k"], y=100 * expiry.quotes["iv_bid"], mode="markers", name="bid", marker={"size": 4})
        )
        smile_fig.add_trace(
            go.Scatter(x=expiry.quotes["k"], y=100 * expiry.quotes["iv_ask"], mode="markers", name="ask", marker={"size": 4})
        )
        smile_fig.add_trace(go.Scatter(x=grid, y=100 * fit.slice.implied_vol(grid), mode="lines", name="SVI"))
        smile_fig.update_layout(height=360, xaxis_title="ln(K/F)", yaxis_title="implied vol (%)", margin={"t": 10})
        st.plotly_chart(smile_fig, use_container_width=True)
        st.dataframe(snapshot.term_structure(), hide_index=True, use_container_width=True)

with calibration_tab:
    if data_error is not None:
        st.error(f"Could not load market data: {data_error}")
    elif st.button("Calibrate Heston to this surface", type="primary"):
        with st.spinner("Calibrating ..."):
            result = calibrate_heston(snapshot)
        st.session_state["calibration"] = result
    if "calibration" in st.session_state and data_error is None:
        result = st.session_state["calibration"]
        st.dataframe(pd.DataFrame([result.summary()]), hide_index=True, use_container_width=True)
        by_expiry = result.error_by_expiry()
        bar = go.Figure(go.Bar(x=by_expiry["expiry"], y=by_expiry["rmse_vol_pts"]))
        bar.update_layout(height=320, yaxis_title="RMSE (vol pts)", margin={"t": 10})
        st.plotly_chart(bar, use_container_width=True)

with hedging_tab:
    st.write(
        "Sell a 3-month at-the-money call at model value and delta-hedge it. Compare hedge ratios in a "
        "Black-Scholes world and in a Heston world."
    )
    paths = st.slider("Monte Carlo paths", 500, 5000, 1500, 500)
    hedge_params = st.session_state["calibration"].params if "calibration" in st.session_state else SYNTHETIC_PARAMS
    st.caption(f"Heston world parameters: {hedge_params}")
    if st.button("Run hedging experiment"):
        with st.spinner("Simulating ..."):
            bs = run_black_scholes_world(n_paths=10_000, rebalances=(4, 16, 64, 256))
            hw = run_heston_world(hedge_params, n_paths=paths, rebalances=(3, 9, 21, 63))
        summary = pd.concat([bs.summary(), hw.summary()], ignore_index=True)
        lines = go.Figure()
        for (world, strategy), group in summary[summary["rebalances"] > 0].groupby(["world", "strategy"], sort=False):
            lines.add_trace(
                go.Scatter(x=group["rebalances"], y=group["std_pct_premium"], mode="lines+markers", name=f"{world}: {strategy}")
            )
        lines.update_layout(
            height=420, xaxis_type="log", yaxis_type="log", xaxis_title="rebalances", yaxis_title="std of P&L (% of premium)"
        )
        st.plotly_chart(lines, use_container_width=True)
        st.dataframe(summary, hide_index=True, use_container_width=True)
