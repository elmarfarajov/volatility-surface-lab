"""Command-line interface.

volsurf chains fetch ^SPX                      store today's option chain
volsurf analyze --ticker ^SPX --out reports/SPX analyse a stored snapshot into a report and manifest
volsurf reproduce reports/SPX                  rerun it and compare every number
volsurf demo --out reports/demo                the same analysis on a synthetic Heston chain, offline
volsurf price --spot 100 --strike 100 --maturity 1 --rate 0.05 --vol 0.2 --heston 0.04,2,0.04,0.5,-0.7
volsurf validate --out docs/VALIDATION.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .validation import run_validation, to_markdown

if TYPE_CHECKING:
    from .platform.analysis import Config
    from .platform.sources import Source


def _config(args: argparse.Namespace) -> Config:
    from .platform.analysis import Config

    overrides = {"exotics": not args.no_exotics, "hedging": not args.no_hedging}
    return Config.fast(**overrides) if args.quick else Config(**overrides)


def _run(source: Source, args: argparse.Namespace) -> None:
    from .platform import analysis, report

    print(f"Source: {source.describe()}")
    result = analysis.run(source, _config(args))
    path = report.write(result, Path(args.out))
    print(f"Report: {path.resolve()}")
    print(f"Manifest: {(path.parent / 'manifest.json').resolve()}  (volsurf reproduce {args.out})")


def cmd_analyze(args: argparse.Namespace) -> None:
    from .platform.sources import snapshot

    _run(snapshot(args.ticker, args.day), args)


def cmd_demo(args: argparse.Namespace) -> None:
    from .platform.sources import synthetic

    _run(synthetic(seed=args.seed), args)


def cmd_reproduce(args: argparse.Namespace) -> None:
    from .platform.manifest import reproduce

    comparisons = reproduce(Path(args.directory), rtol=args.rtol)
    failed = [c for c in comparisons if not c.agrees]
    for c in failed:
        print(f"  differs: {c.name}: recorded {c.recorded:.10g}, rerun {c.rerun:.10g}")
    print(f"{len(comparisons) - len(failed)} of {len(comparisons)} results reproduced to a relative {args.rtol:g}")
    if failed:
        sys.exit(1)


def cmd_price(args: argparse.Namespace) -> None:
    from .analytic.generalized import price as bsm
    from .numerics import lattice, montecarlo, pde
    from .numerics.lattice import Contract
    from .stochvol import pricing
    from .stochvol.characteristic import Heston
    from .stochvol.simulation import otm_estimate, simulate

    s, k, t, r, q, v = args.spot, args.strike, args.maturity, args.rate, args.dividend, args.vol
    call = args.type == "call"
    c = Contract(s, k, t, r, q, v, call)
    rows: list[tuple[str, float, float | None]] = [
        ("Black-Scholes-Merton (analytic)", float(bsm(s, k, t, r, q, v, call)), None),
        ("Leisen-Reimer tree, 301 steps", lattice.price(c, 301).price, None),
        ("CRR tree, 2000 steps", lattice.price(c, 2000, "crr").price, None),
        ("Crank-Nicolson PDE after Rannacher steps", pde.solve(c, 800, 800).price, None),
    ]
    mc = montecarlo.european(c, 2**17, "sobol", seed=1)
    rows.append(("Monte Carlo, scrambled Sobol'", mc.price, mc.std_error))
    rows.append(("American: Leisen-Reimer tree, 1001 steps", lattice.price(c, 1001, exercise="american").price, None))
    rows.append(("American: Crank-Nicolson PDE", pde.solve(c, 800, 800, exercise="american").price, None))
    ls = montecarlo.longstaff_schwartz(c, np.linspace(t / 50, t, 50).tolist(), 20_000, 20_000, 500, 100, seed=2)
    rows.append(("American: Longstaff-Schwartz, lower bound", ls.lower.price, ls.lower.std_error))
    rows.append(("American: Andersen-Broadie, dual upper bound", ls.upper.price, ls.upper.std_error))
    if args.heston:
        h = Heston(*(float(x) for x in args.heston.split(",")))
        forward, discount = s * np.exp((r - q) * t), float(np.exp(-r * t))
        rows.append(("Heston: contour integral", float(pricing.price(forward, [k], t, h, discount, call)[0]), None))
        kk = float(np.log(k / forward))
        cos = float(pricing.cos_otm_price(np.array([kk]), t, h)[0]) * forward
        intrinsic = max(forward - k, 0.0) if call else max(k - forward, 0.0)
        cos += intrinsic if (kk < 0) == call else 0.0
        rows.append(("Heston: COS", discount * cos, None))
        paths = simulate(h, t, 100, 100_000, "qe", seed=3)
        est = otm_estimate(paths, np.array([kk]))
        value = float(est.value[0]) * forward + (intrinsic if (kk < 0) == call else 0.0)
        rows.append(("Heston: QE Monte Carlo", discount * value, discount * forward * float(est.stderr[0])))
    width = max(len(name) for name, _, _ in rows)
    for name, value, se in rows:
        extra = f"  +- {se:.4f}" if se is not None else ""
        print(f"{name:<{width}}  {value:>12.6f}{extra}")


def cmd_validate(args: argparse.Namespace) -> None:
    results = run_validation(fast=args.fast)
    table = to_markdown(results)
    print(table)
    if args.out:
        header = (
            "# Validation report\n\n"
            f"Generated by `volsurf validate` on {datetime.now(timezone.utc):%Y-%m-%d}. "
            "Every numerical engine is checked against a published reference value or an independent method.\n\n"
        )
        Path(args.out).write_text(header + table + "\n", encoding="utf-8")
    if not results["passed"].all():
        sys.exit(1)


def cmd_greeks(args: argparse.Namespace) -> None:
    from .analytic import Carry, greeks

    carry = {
        "stock": Carry.stock(args.rate, args.dividend),
        "future": Carry.future(args.rate),
        "currency": Carry.currency(args.rate, args.dividend),
    }[args.underlying]
    g = greeks(args.spot, args.strike, args.maturity, carry.rate, carry.yield_, args.vol, args.type == "call")
    print(f"{carry.convention} {args.type}: S={args.spot} K={args.strike} T={args.maturity} r={carry.rate} q={carry.yield_}")
    for name, value in g.as_dict().items():
        print(f"  {name:<11} {float(value): .10f}")


def cmd_gallery(args: argparse.Namespace) -> None:
    from pathlib import Path

    from . import gallery

    written = gallery.build(args.out, only=args.only)
    Path("docs").mkdir(exist_ok=True)
    if args.only is None:
        Path("docs/GALLERY.md").write_text(gallery.markdown(), encoding="utf-8")
    print(f"{len(written)} charts written to {args.out}")


def cmd_chains(args: argparse.Namespace) -> None:
    from .market import chains

    if args.action == "fetch":
        for ticker in args.tickers:
            print(f"{ticker}: stored {chains.fetch(ticker)}")
        return
    stored = chains.snapshots()
    if not stored:
        print(f"no snapshots under {chains.data_root()}; run `volsurf chains fetch ^SPX`")
    for day, ticker in stored:
        chain = chains.load(ticker, day)
        expiries, quotes = len(chain.expiries()), len(chain.quotes)
        print(f"{day}  {ticker:<6} spot {chain.spot:>10,.2f}  {expiries:>3} expiries  {quotes:>6,} quotes")


def cmd_history(args: argparse.Namespace) -> None:
    from .risk import history

    if args.action == "fetch":
        print(f"stored {history.fetch()}")
        return
    if not history.available():
        print(f"no history under {history.history_root()}; run `volsurf history fetch`")
        return
    frame = history.load()
    print(f"{len(frame):,} days from {frame.index[0].date()} to {frame.index[-1].date()} under {history.history_root()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volsurf", description="Option pricing, volatility surfaces and Heston calibration.")
    sub = parser.add_subparsers(dest="command", required=True)

    def analysis_flags(p: argparse.ArgumentParser, default_out: str) -> None:
        p.add_argument("--out", default=default_out, help="directory for report.html, figures, tables and manifest.json")
        p.add_argument("--quick", action="store_true", help="fewer starts and paths: a first look in under a minute")
        p.add_argument("--no-exotics", action="store_true", help="skip pricing the exotic book under three models")
        p.add_argument("--no-hedging", action="store_true", help="skip the cross-model hedging simulation")

    analyze = sub.add_parser("analyze", help="analyse a stored option-chain snapshot (see `volsurf chains fetch`)")
    analyze.add_argument("--ticker", default="^SPX")
    analyze.add_argument("--day", default=None, help="the snapshot's day, YYYY-MM-DD (default: the latest stored)")
    analysis_flags(analyze, "reports/SPX")
    analyze.set_defaults(func=cmd_analyze)

    demo = sub.add_parser("demo", help="the same analysis offline, on a synthetic Heston chain")
    demo.add_argument("--seed", type=int, default=7)
    analysis_flags(demo, "reports/demo")
    demo.set_defaults(func=cmd_demo)

    again = sub.add_parser("reproduce", help="rerun the analysis recorded in a report directory and compare every number")
    again.add_argument("directory")
    again.add_argument("--rtol", type=float, default=1e-6)
    again.set_defaults(func=cmd_reproduce)

    validate = sub.add_parser("validate", help="run the numerical validation suite")
    validate.add_argument("--out", default=None, help="write the results as Markdown")
    validate.add_argument("--fast", action="store_true")
    validate.set_defaults(func=cmd_validate)

    price = sub.add_parser("price", help="price one contract with every engine")
    price.add_argument("--spot", type=float, required=True)
    price.add_argument("--strike", type=float, required=True)
    price.add_argument("--maturity", type=float, required=True)
    price.add_argument("--rate", type=float, default=0.0)
    price.add_argument("--dividend", type=float, default=0.0)
    price.add_argument("--vol", type=float, required=True)
    price.add_argument("--type", choices=["call", "put"], default="call")
    price.add_argument("--heston", default=None, help="v0,kappa,theta,sigma,rho")
    price.set_defaults(func=cmd_price)

    greek = sub.add_parser("greeks", help="the price and seventeen Greeks of a European option")
    greek.add_argument("--spot", type=float, required=True)
    greek.add_argument("--strike", type=float, required=True)
    greek.add_argument("--maturity", type=float, required=True)
    greek.add_argument("--rate", type=float, default=0.0)
    greek.add_argument("--dividend", type=float, default=0.0, help="dividend yield, or the foreign rate for a currency")
    greek.add_argument("--vol", type=float, required=True)
    greek.add_argument("--type", choices=["call", "put"], default="call")
    greek.add_argument("--underlying", choices=["stock", "future", "currency"], default="stock")
    greek.set_defaults(func=cmd_greeks)

    charts = sub.add_parser("gallery", help="redraw the documentation's charts")
    charts.add_argument("--out", default="docs/images")
    charts.add_argument("--only", default=None, help="one chart's file name, or a day's number")
    charts.set_defaults(func=cmd_gallery)

    snapshot = sub.add_parser("chains", help="store option-chain snapshots from Yahoo Finance, or list them")
    snapshot.add_argument("action", choices=["fetch", "list"])
    snapshot.add_argument("tickers", nargs="*", default=["^SPX", "SPY", "AAPL"])
    snapshot.set_defaults(func=cmd_chains)

    past = sub.add_parser("history", help="store S&P 500, VIX and bill-rate history from Yahoo Finance, or describe it")
    past.add_argument("action", choices=["fetch", "list"])
    past.set_defaults(func=cmd_history)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
