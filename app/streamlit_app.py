"""Interactive dashboard over every engine: `streamlit run app/streamlit_app.py`.

It reads stored option-chain snapshots (`volsurf chains fetch ^SPX`) and never downloads on
its own; without one it starts on a synthetic Heston chain, so it runs offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsurf.analytic.generalized import greeks
from volsurf.exotics import closed_form as cf
from volsurf.exotics.monte_carlo import barrier_mc, bgk_barrier
from volsurf.localvol.dupire import local_vol
from volsurf.localvol.dynamics import monthly
from volsurf.market import chains
from volsurf.market.cleaning import prepare
from volsurf.numerics import lattice, pde
from volsurf.numerics.lattice import Contract
from volsurf.platform.analysis import model_expiries
from volsurf.platform.report import surface_figure
from volsurf.platform.sources import Source, snapshot, synthetic
from volsurf.risk import book as bk
from volsurf.risk.hedging import black_scholes_study, wrong_volatility
from volsurf.smile.build import SurfaceFit, build_surface
from volsurf.stochvol.calibration import model_vols, multistart

st.set_page_config(page_title="Volatility Surface Lab", layout="wide")
SYNTHETIC = "Synthetic Heston chain (offline)"


@st.cache_resource(show_spinner="Loading the option chain ...")
def load_source(choice: str) -> Source:
    if choice == SYNTHETIC:
        return synthetic()
    ticker, day = choice.split(" on ")
    return snapshot(ticker, day)


@st.cache_resource(show_spinner="Cleaning quotes and fitting the arbitrage-free surface ...")
def load_fit(choice: str) -> SurfaceFit:
    return build_surface(prepare(load_source(choice).chain))


st.title("Volatility Surface Lab")
choices = [f"{name} on {day}" for day, name in chains.snapshots()] + [SYNTHETIC]
choice = st.sidebar.selectbox("Option chain", choices, index=len(choices) - 1)
st.sidebar.caption("Store more with `volsurf chains fetch ^SPX`. Every engine below is validated in docs/VALIDATION.md.")
source = load_source(choice)
st.caption(source.describe())

pricer, surface_tab, models, exotics, hedging, risk = st.tabs(["Pricer", "Surface", "Models", "Exotics", "Hedging", "Risk"])

with pricer:
    c1, c2, c3, c4 = st.columns(4)
    s = c1.number_input("Spot", value=100.0, min_value=0.01)
    k = c1.number_input("Strike", value=100.0, min_value=0.01)
    t = c2.number_input("Years to expiry", value=0.5, min_value=0.0)
    vol = c2.number_input("Volatility", value=0.2, min_value=0.0)
    r = c3.number_input("Rate", value=0.04)
    q = c3.number_input("Dividend yield (or foreign rate)", value=0.01)
    call = c4.radio("Type", ["call", "put"]) == "call"
    g = greeks(s, k, t, r, q, vol, call)
    table = pd.DataFrame({"Greek": list(g.as_dict()), "value": [float(v) for v in g.as_dict().values()]})
    st.dataframe(table, hide_index=True, use_container_width=False)
    if t > 0 and vol > 0:
        contract = Contract(s, k, t, r, q, vol, call)
        engines = {
            "analytic": float(g.price),
            "Leisen-Reimer tree (301)": lattice.price(contract, 301).price,
            "Crank-Nicolson PDE": pde.solve(contract, 400, 400).price,
            "American, Leisen-Reimer (1001)": lattice.price(contract, 1001, exercise="american").price,
        }
        st.dataframe(pd.DataFrame({"engine": list(engines), "price": list(engines.values())}), hide_index=True)

with surface_tab:
    fit = load_fit(choice)
    margins = pd.DataFrame(
        {
            "days": [365 * f.maturity for f in fit.svi],
            "RMSE (vol pts)": [100 * f.rmse for f in fit.svi],
            "min g (butterfly)": [f.butterfly_margin for f in fit.svi],
            "min calendar gap": [f.calendar_margin for f in fit.svi],
        }
    )
    st.metric("Median fit error", f"{margins['RMSE (vol pts)'].median():.2f} vol pts")
    st.plotly_chart(surface_figure(fit), use_container_width=True)
    i = st.select_slider(
        "Expiry", options=list(range(len(fit.quotes))), format_func=lambda j: f"{365 * fit.quotes[j].maturity:.0f} days"
    )
    quotes = fit.quotes[i]
    grid = np.linspace(float(quotes.k.min()), float(quotes.k.max()), 200)
    smile = go.Figure()
    smile.add_scatter(x=quotes.k, y=100 * quotes.iv, mode="markers", name="mid quotes")
    smile.add_scatter(x=grid, y=100 * fit.surface.implied_vol(grid, quotes.maturity), name="arbitrage-free SVI")
    smile.update_layout(xaxis_title="ln(K/F)", yaxis_title="implied vol (%)", height=380)
    st.plotly_chart(smile, use_container_width=True)
    st.dataframe(margins, hide_index=True)

with models:
    fit = load_fit(choice)
    starts = st.slider("Heston calibration: Sobol' starts", 1, 16, 2)
    if st.button("Calibrate Heston"):
        expiries = model_expiries(prepare(source.chain))
        with st.spinner("Calibrating ..."):
            st.session_state["calibration"] = multistart(expiries, starts)[0]
            st.session_state["calibration_expiries"] = expiries
    if "calibration" in st.session_state:
        cal = st.session_state["calibration"]
        h = cal.heston
        st.write(f"v0 {h.v0:.4f}, kappa {h.kappa:.2f}, theta {h.theta:.4f}, vol of vol {h.sigma:.2f}, rho {h.rho:.2f}")
        st.metric("RMSE", f"{100 * cal.rmse:.2f} vol pts")
        expiries = st.session_state["calibration_expiries"]
        vols = model_vols(h, expiries)
        fig = go.Figure()
        for e, v in zip(expiries, vols, strict=True):
            fig.add_scatter(x=e.k, y=100 * e.iv, mode="markers", name=f"{365 * e.maturity:.0f}d market")
            fig.add_scatter(x=e.k, y=100 * v, mode="lines", name=f"{365 * e.maturity:.0f}d Heston")
        fig.update_layout(xaxis_title="ln(K/F)", yaxis_title="implied vol (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    surface = monthly(fit.surface)
    times = np.linspace(float(surface.maturities[0]), float(surface.maturities[-1]), 30)
    kk = np.linspace(-0.4, 0.2, 41)
    heat = go.Figure(go.Heatmap(x=kk, y=365 * times, z=100 * np.clip([local_vol(surface, kk, float(x)) for x in times], 0, 1)))
    heat.update_layout(title="Dupire local volatility (%)", xaxis_title="ln(K/F)", yaxis_title="days", height=420)
    st.plotly_chart(heat, use_container_width=True)

with exotics:
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Barrier", ["down-out", "down-in", "up-out", "up-in"])
    h_level = c1.number_input("Barrier level", value=85.0 if kind.startswith("down") else 120.0)
    xs = c2.number_input("Strike ", value=100.0)
    vol_e = c2.number_input("Volatility ", value=0.25)
    monitoring = c3.select_slider("Monitoring dates a year", [4, 12, 52, 252], value=52)
    try:
        exact = cf.barrier(kind, 100.0, xs, h_level, 1.0, 0.05, 0.03, vol_e, True)  # type: ignore[arg-type]
    except ValueError as error:
        st.warning(str(error))
    else:
        mc = barrier_mc(kind, 100.0, xs, h_level, 1.0, 0.05, 0.03, vol_e, True, monitoring, 40_000)
        rows = {
            "continuous, closed form": exact,
            "discrete, BGK correction": bgk_barrier(kind, 100.0, xs, h_level, 1.0, 0.05, 0.03, vol_e, True, monitoring),
            f"discrete, Monte Carlo ({mc.stderr:.3f} se)": mc.value,
        }
        st.dataframe(pd.DataFrame({"price": rows}), use_container_width=False)
    st.write(
        f"Geometric Asian (52 fixings) {cf.geometric_asian(100.0, xs, 1.0, 0.05, 0.03, vol_e, True, 52):.4f} - "
        f"floating lookback call {cf.floating_lookback(100.0, 100.0, 1.0, 0.05, 0.03, vol_e, True):.4f} - "
        f"cash digital {cf.cash_or_nothing(100.0, xs, 1.0, 0.05, 0.03, vol_e, 1.0, True):.4f}"
    )

with hedging:
    paths = st.select_slider("Paths", [2_000, 5_000, 20_000], value=2_000)
    if st.button("Run the Black-Scholes hedging study"):
        study = black_scholes_study(paths=paths, steps=252, rebalances=(12, 63, 252), show=63)
        tab = study.table.assign(
            **{c: 100 * study.table[c] / study.table["premium"] for c in ("delta", "delta_gamma", "derman_kamal")}
        )
        st.dataframe(
            tab.drop(columns="premium").rename(columns=lambda c: f"{c} (% premium)" if c != "rebalances" else c), hide_index=True
        )
    sigma_true = st.slider("Realised volatility when hedging at 20%", 0.1, 0.4, 0.25)
    w = wrong_volatility(sigma_true=sigma_true, paths=paths, steps=63)
    st.metric("Mean hedged P&L", f"{100 * w.pnl.mean() / w.premium:.1f}% of premium")
    st.metric("Correlation with the gamma formula", f"{np.corrcoef(w.pnl, w.predicted)[0, 1]:.3f}")

with risk:
    fit = load_fit(choice)
    book = bk.build(fit)
    c = st.slider("Smile dynamic: 0 sticky moneyness, 1 sticky strike, 2 local volatility", 0.0, 2.5, 1.0, 0.1)
    moves = np.linspace(-0.2, 0.1, 13)
    ladder = bk.stress_ladder(book, moves, {"P&L ($)": c})
    st.line_chart(ladder.rename(index=lambda m: f"{100 * m:+.1f}%"))
    st.caption(", ".join(f"{n:+.0f} {name}" for name, n in zip(book.names, book.contracts, strict=True)) + "; delta-hedged.")
