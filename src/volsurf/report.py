"""Self-contained HTML research report plus PNG exports of every figure."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import __version__, figures
from .pipeline import AnalysisResult

_CSS = """
:root { --ink:#1f2933; --muted:#616e7c; --line:#e4e7eb; --soft:#f5f7fa; --accent:#2f6fdb; --warm:#e0632b; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif;
       color:var(--ink); background:#fff; line-height:1.55; }
main { max-width: 1120px; margin: 0 auto; padding: 36px 20px 80px; }
header h1 { font-size: 30px; margin: 0 0 6px; letter-spacing:-0.01em; }
header p { color: var(--muted); margin: 0; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 12px; border-top: 1px solid var(--line); }
h2 span { color: var(--muted); font-weight: 500; font-size: 14px; margin-left: 8px; }
p.lead { color: var(--muted); max-width: 820px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:12px; margin: 26px 0 8px; }
.kpi { background: var(--soft); border-radius: 10px; padding: 14px 16px; }
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.kpi .value { font-size: 22px; font-weight: 650; margin-top: 2px; }
img { max-width: 100%; height: auto; display:block; margin: 14px 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 12px 0; min-width: 60%; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { background: var(--soft); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
.note { background: #fff8f1; border-left: 3px solid var(--warm); padding: 10px 14px;
        border-radius: 4px; color: #52606d; font-size: 14px; }
footer { margin-top: 60px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 14px; }
"""


def _png(fig: plt.Figure, path: Path) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    data = buffer.getvalue()
    path.write_bytes(data)
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(data).decode()}">'


def _table(df: pd.DataFrame, float_format: str = "{:,.4f}") -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else float_format.format(v))
    return f'<div class="table-wrap">{formatted.to_html(index=False, border=0)}</div>'


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'


def interactive_surface(result: AnalysisResult) -> str:
    z, maturities, vol = figures.surface_grid(result.surface, result.snapshot, n_t=45, n_z=50)
    vol = 100 * vol
    q = result.snapshot.quotes()
    q = q.assign(z=q["k"] / np.sqrt(q["maturity"]))
    q = q[(q["z"] >= z[0]) & (q["z"] <= z[-1])]
    fig = go.Figure(
        data=[
            go.Surface(
                x=z,
                y=maturities,
                z=vol,
                colorscale="Viridis",
                opacity=0.92,
                colorbar={"title": "vol %"},
                hovertemplate="k/sqrt(T)=%{x:.2f}<br>T=%{y:.2f}y<br>vol=%{z:.2f}%<extra>SVI surface</extra>",
            ),
            go.Scatter3d(
                x=q["z"],
                y=q["maturity"],
                z=100 * q["iv_mid"],
                mode="markers",
                marker={"size": 2.2, "color": "#1f2933"},
                name="market mid",
                hovertemplate="K=%{customdata:.0f}<br>vol=%{z:.2f}%<extra>market</extra>",
                customdata=q["strike"],
            ),
        ]
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis_title": "ln(K/F)/sqrt(T)",
            "yaxis_title": "maturity (y)",
            "zaxis_title": "implied vol (%)",
            "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.8}},
        },
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def write_report(result: AnalysisResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    snap, cal = result.snapshot, result.calibration

    images = {
        "surface": _png(figures.surface_3d(result.surface, snap), fig_dir / "vol_surface_3d.png"),
        "smiles": _png(figures.smiles(snap, result.svi_fits, cal), fig_dir / "smiles.png"),
        "skew": _png(figures.skew_term_structure(snap, result.svi_fits, cal), fig_dir / "term_structure.png"),
        "local_vol": _png(figures.local_vol_heatmap(result.surface, snap), fig_dir / "local_vol.png"),
        "densities": _png(figures.densities(result.svi_fits), fig_dir / "densities.png"),
        "residuals": _png(figures.calibration_residuals(cal), fig_dir / "calibration_residuals.png"),
    }
    if result.hedging_heston is not None:
        images["hedging"] = _png(figures.hedging(result.hedging_bs, result.hedging_heston), fig_dir / "hedging_lab.png")

    ts = snap.term_structure()
    ts_table = ts.assign(
        implied_rate=100 * ts["implied_rate"], implied_carry=100 * ts["implied_carry"], atm_vol=100 * ts["atm_vol"]
    ).rename(columns={"implied_rate": "implied rate %", "implied_carry": "carry r-q %", "atm_vol": "ATM vol %"})

    svi_rows = pd.DataFrame(
        [
            {
                **{k: v for k, v in f.slice.as_dict().items()},
                "RMSE vol pts": 100 * f.rmse_vol,
                "min Durrleman g": f.min_durrleman_g,
                "calendar violation": f.calendar_violation,
                "quotes": f.n_quotes,
            }
            for f in result.svi_fits
        ]
    )
    cal_summary = pd.DataFrame([{k: v for k, v in cal.summary().items()}])
    by_expiry = cal.error_by_expiry()

    worst_expiry = by_expiry.iloc[int(np.argmax(by_expiry["rmse_vol_pts"]))]
    feller = cal.params.feller_ratio
    sections = []
    sections.append(f"""
<h2>1. Implied volatility surface <span>interactive</span></h2>
<p class="lead">Each dot is an out-of-the-money mid quote converted to Black-76 implied volatility with the
expiry's own market-implied forward and discount factor. The surface is a stack of arbitrage-controlled SVI
slices joined by monotone interpolation in total variance.</p>
{interactive_surface(result)}
{images["surface"]}
""")
    sections.append(f"""
<h2>2. Smiles: market vs SVI vs Heston</h2>
<p class="lead">Error bars span the bid-ask range in volatility terms. SVI is a per-expiry parametric smile;
Heston is a single five-parameter dynamic model that must explain every expiry at once.</p>
{images["smiles"]}
""")
    sections.append(f"""
<h2>3. Term structure, carry and skew</h2>
<p class="lead">Forwards and discount factors are extracted per expiry from put-call parity,
C - P = D(F - K), so the implied rate and carry below are revealed by option prices themselves.</p>
{_table(ts_table)}
{images["skew"]}
""")
    sections.append(f"""
<h2>4. SVI slices and static-arbitrage diagnostics</h2>
<p class="lead">A slice is free of butterfly arbitrage when Durrleman's g(k) is non-negative everywhere, and the
surface is free of calendar arbitrage when total variance never decreases with maturity.</p>
{_table(svi_rows)}
""")
    sections.append(f"""
<h2>5. Local volatility and risk-neutral densities</h2>
{images["local_vol"]}
{images["densities"]}
""")
    sections.append(f"""
<h2>6. Heston calibration</h2>
<p class="lead">Five parameters fitted to {cal.n_quotes} quotes in {cal.elapsed_seconds:.1f}s
({cal.n_evaluations} model evaluations). Feller ratio 2&kappa;&theta;/&sigma;&sup2; = {feller:.2f}
({"variance stays strictly positive" if feller >= 1 else "variance can touch zero, typical for equity indices"}).
The largest misfit is at the {worst_expiry["maturity"]:.2f}y expiry ({worst_expiry["rmse_vol_pts"]:.2f} vol pts RMSE).</p>
{_table(cal_summary)}
{_table(by_expiry)}
{images["residuals"]}
""")
    if result.hedging_heston is not None:
        hedge_table = pd.concat([result.hedging_bs.summary(), result.hedging_heston.summary()], ignore_index=True)
        sections.append(f"""
<h2>7. Hedging lab: model risk in discrete delta hedging</h2>
<p class="lead">A dealer sells a 3-month at-the-money call at model value and delta-hedges it. In a Black-Scholes
world the error vanishes like N<sup>-1/2</sup>. In the calibrated Heston world it plateaus because variance
shocks cannot be hedged with the underlying alone; the minimum-variance delta, which also offsets the
spot-correlated part of the variance shock, is the most effective hedge.</p>
{_table(hedge_table)}
{images["hedging"]}
""")

    kpis = "".join(
        [
            _kpi("Spot", f"{snap.spot:,.2f}"),
            _kpi("Expiries / quotes", f"{len(snap.expiries)} / {sum(len(e.quotes) for e in snap.expiries)}"),
            _kpi("Short / long ATM vol", f"{100 * snap.expiries[0].atm_vol():.1f}% / {100 * snap.expiries[-1].atm_vol():.1f}%"),
            _kpi("Heston RMSE", f"{100 * cal.rmse_vol:.2f} vol pts"),
            _kpi("Heston rho", f"{cal.params.rho:.2f}"),
            _kpi("Worst SVI slice", f"{100 * max(f.rmse_vol for f in result.svi_fits):.2f} vol pts"),
        ]
    )
    timing = ", ".join(f"{k.replace('_', ' ')} {v:.1f}s" for k, v in result.timings.items())
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(snap.ticker)} Volatility Surface Report</title><style>{_CSS}</style></head>
<body><main>
<header>
  <h1>{html.escape(snap.ticker)} volatility surface report</h1>
  <p>Source: {html.escape(snap.source)} &middot; as of {snap.as_of:%Y-%m-%d %H:%M} UTC &middot; volsurf {__version__}</p>
</header>
<div class="kpis">{kpis}</div>
{"".join(sections)}
<footer>Runtime: {html.escape(timing)}. Research and educational output; not investment advice.
Market data may be delayed or indicative.</footer>
</main></body></html>"""
    path = out_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
