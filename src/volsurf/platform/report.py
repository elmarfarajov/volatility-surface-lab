"""The analysis as one HTML file, with its figures as PNGs, its tables as CSVs, and the manifest beside them.

The report is a single file: figures are embedded, and only the interactive surface loads
plotly.js from its CDN. Every number it quotes comes from :attr:`Analysis.results`, which
the manifest records.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from .. import __version__
from ..gallery.style import save
from ..smile.build import SurfaceFit
from . import figures, manifest
from .analysis import Analysis

_CSS = """
:root { --ink:#1f2933; --muted:#616e7c; --line:#e4e7eb; --soft:#f5f7fa; --accent:#2f6fdb; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--ink);
       background:#fff; line-height:1.55; }
main { max-width: 1120px; margin: 0 auto; padding: 36px 20px 80px; }
header h1 { font-size: 30px; margin: 0 0 6px; letter-spacing:-0.01em; }
header p { color: var(--muted); margin: 0; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 12px; border-top: 1px solid var(--line); }
p.lead { color: var(--muted); max-width: 860px; }
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
code { background: var(--soft); padding: 1px 5px; border-radius: 4px; }
footer { margin-top: 60px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 14px; }
"""


HISTORY_NOTE = " and historical value at risk from every day since 1990"


def _image(fig: Figure, path: Path) -> str:
    save(fig, path)
    data = path.read_bytes()
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(data).decode()}">'


def _table(df: pd.DataFrame, digits: int = 4) -> str:
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}f}")
    return f'<div class="table-wrap">{shown.to_html(index=False, border=0)}</div>'


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'


def surface_figure(fit: SurfaceFit) -> go.Figure:
    """The surface in implied volatility against standardised moneyness and expiry, with the quotes: interactive."""
    maturities = np.array([e.maturity for e in fit.expiries])
    t = np.geomspace(maturities[0], maturities[-1], 45)
    z = np.linspace(-2.5, 1.5, 50)
    vol = np.array([fit.surface.implied_vol(z * np.sqrt(float(fit.surface.theta(x))), float(x)) for x in t])
    quotes = pd.concat([pd.DataFrame({"k": q.k, "iv": q.iv, "t": q.maturity}) for q in fit.quotes])
    quotes["z"] = quotes["k"] / np.sqrt([float(fit.surface.theta(x)) for x in quotes["t"]])
    quotes = quotes[(quotes["z"] >= z[0]) & (quotes["z"] <= z[-1])]
    fig = go.Figure(
        data=[
            go.Surface(x=z, y=t, z=100 * vol, colorscale="Viridis", opacity=0.92, colorbar={"title": "vol %"}),
            go.Scatter3d(
                x=quotes["z"], y=quotes["t"], z=100 * quotes["iv"], mode="markers", marker={"size": 2.2, "color": "#1f2933"}
            ),
        ]
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=False,
        scene={
            "xaxis_title": "ln(K/F) / sqrt(w_ATM)",
            "yaxis_title": "years",
            "zaxis_title": "implied vol (%)",
            "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.8}},
        },
    )
    return fig


def tables(a: Analysis) -> dict[str, pd.DataFrame]:
    out = {"term_structure": a.market.term_structure()}
    out["svi"] = pd.DataFrame(
        [
            {
                "days": 365 * f.maturity,
                **{k: getattr(f.slice, k) for k in ("a", "b", "rho", "m", "sigma")},
                "rmse_vol": f.rmse,
                "butterfly_margin": f.butterfly_margin,
                "calendar_margin": f.calendar_margin,
            }
            for f in a.fit.svi
        ]
    )
    out["sabr"] = pd.DataFrame(
        [
            {
                "days": 365 * t,
                "alpha": f.params.alpha,
                "beta": f.params.beta,
                "rho": f.params.rho,
                "nu": f.params.nu,
                "rmse_vol": f.rmse,
            }
            for t, f in a.sabr
        ]
    )
    if a.exotics is not None:
        rows = [{"contract": c, **{m: a.exotics.prices[(m, c)].value for m in a.exotics.models}} for c in a.exotics.contracts]
        out["exotics"] = pd.DataFrame(rows)
    if a.hedging is not None:
        out["hedging"] = a.hedging.table
    out["stress"] = a.ladder.reset_index(names="index move")
    return out


def write(a: Analysis, directory: Path) -> Path:
    """``report.html``, ``figures/*.png``, ``tables/*.csv`` and ``manifest.json`` in ``directory``."""
    directory = Path(directory)
    (directory / "figures").mkdir(parents=True, exist_ok=True)
    (directory / "tables").mkdir(parents=True, exist_ok=True)
    images = {name: _image(fig, directory / "figures" / f"{name}.png") for name, fig in figures.all_figures(a).items()}
    frames = tables(a)
    for name, frame in frames.items():
        frame.to_csv(directory / "tables" / f"{name}.csv", index=False)
    manifest.write(a, directory)

    r = a.results
    h = a.heston.heston
    sections = [
        f"""<h2>1. The surface</h2>
<p class="lead">{int(r["quotes"]):,} quotes over {int(r["expiries"])} expiries, cleaned for stated reasons and fitted
by raw SVI under hard butterfly and calendar constraints, checked on the whole real line (Days 3 and 4). Median error
{100 * r["surface.median_rmse_vol"]:.2f} vol points, worst slice {100 * r["surface.worst_rmse_vol"]:.2f}.</p>
{surface_figure(a.fit).to_html(full_html=False, include_plotlyjs="cdn")}
{images["surface"]}
{images["smiles"]}
{_table(frames["svi"])}""",
        f"""<h2>2. Term structure</h2>
<p class="lead">Forwards and discount factors from put-call parity on each expiry's own quotes.</p>
{images["term-structure"]}
{_table(frames["term_structure"][["expiry", "maturity", "forward", "discount", "curve_rate", "carry", "quotes"]])}""",
        f"""<h2>3. Densities and local volatility</h2>
<p class="lead">Risk-neutral densities from each slice, and Dupire local volatility in closed form (Day 6).</p>
{images["densities-local-vol"]}""",
        f"""<h2>4. Heston and SABR</h2>
<p class="lead">Heston calibrated from {a.config.heston_starts} Sobol' starts to bid-ask-weighted implied volatilities (Day 5):
v0 {h.v0:.4f}, kappa {h.kappa:.2f}, theta {h.theta:.4f}, vol of vol {h.sigma:.2f}, rho {h.rho:.2f};
RMSE {100 * a.heston.rmse:.2f} vol points. SABR (beta 1) fitted expiry by expiry.</p>
{images["heston-residuals"]}
{_table(frames["sabr"])}""",
    ]
    if a.exotics is not None:
        sections.append(
            f"""<h2>5. Exotics under three models</h2>
<p class="lead">One-year contracts, per unit of spot, under Black-Scholes, Heston and local volatility (Day 7).</p>
{images["exotics"]}
{_table(frames["exotics"])}"""
        )
    if a.hedging is not None:
        sections.append(
            f"""<h2>6. Hedging and model risk</h2>
<p class="lead">A three-month at-the-money call sold at the surface's price, hedged in each model's world with each
model's delta (Day 8).</p>
{images["hedging"]}
{_table(frames["hedging"])}"""
        )
    sections.append(
        f"""<h2>7. A dealer book's risk</h2>
<p class="lead">Stress under each smile dynamic{HISTORY_NOTE if a.historical else ""} (Day 8).</p>
{images["book"]}
{_table(frames["stress"], digits=0)}"""
    )
    kpis = "".join(
        [
            _kpi("Spot", f"{r['spot']:,.2f}"),
            _kpi("Expiries / quotes", f"{int(r['expiries'])} / {int(r['quotes']):,}"),
            _kpi("Surface error", f"{100 * r['surface.median_rmse_vol']:.2f} vol pts"),
            _kpi("Heston RMSE", f"{100 * r['heston.rmse_vol']:.2f} vol pts"),
            _kpi("Heston rho", f"{r['heston.rho']:.2f}"),
            _kpi("10% stress", f"${-r['book.stress_-10%.sticky strike'] / 1e6:.1f}m (sticky strike)"),
        ]
    )
    timing = ", ".join(f"{k} {v:.1f}s" for k, v in a.timings.items())
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(a.source.ticker)} Volatility Report</title><style>{_CSS}</style></head>
<body><main>
<header>
  <h1>{html.escape(a.source.ticker)} volatility report</h1>
  <p>{html.escape(a.source.describe())} &middot; volsurf {__version__}</p>
</header>
<div class="kpis">{kpis}</div>
{"".join(sections)}
<footer>Reproduce with <code>volsurf reproduce {html.escape(directory.name)}</code>: the manifest records the source's SHA-256,
the configuration, the library versions and every number above. Runtime: {html.escape(timing)}.
Research output, not investment advice; market data delayed.</footer>
</main></body></html>"""
    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
