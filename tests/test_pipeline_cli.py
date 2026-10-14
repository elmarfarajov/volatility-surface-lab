"""The platform end to end: a quick analysis into a report and manifest, its reproduction, and the command line."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from volsurf.cli import build_parser, main
from volsurf.platform import manifest, sources
from volsurf.validation import run_validation, to_markdown


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    main(["demo", "--quick", "--out", str(out)])
    return out


def test_the_demo_writes_a_report_its_figures_tables_and_manifest(demo):
    page = (demo / "report.html").read_text(encoding="utf-8")
    for section in ("The surface", "Heston and SABR", "Exotics under three models", "Hedging and model risk", "dealer book"):
        assert section in page
    figures = {p.stem for p in (demo / "figures").glob("*.png")}
    assert figures == {
        "surface",
        "smiles",
        "term-structure",
        "densities-local-vol",
        "heston-residuals",
        "exotics",
        "hedging",
        "book",
    }
    assert all(p.stat().st_size > 20_000 for p in (demo / "figures").glob("*.png"))
    assert pd.read_csv(demo / "tables" / "svi.csv")["rmse_vol"].max() < 0.01
    record = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    assert record["source"]["kind"] == "synthetic" and len(record["source"]["sha256"]) == 64
    assert record["config"]["quick"] is True
    # the synthetic chain is Heston with known parameters, and the calibration finds them
    assert record["results"]["heston.rho"] == pytest.approx(-0.74, abs=0.03)
    assert record["results"]["heston.sigma"] == pytest.approx(0.8, abs=0.08)


def test_a_rerun_reproduces_every_number(demo):
    comparisons = manifest.reproduce(demo, log=lambda _: None)
    assert len(comparisons) > 40
    assert all(c.agrees for c in comparisons), [c for c in comparisons if not c.agrees]


def test_a_changed_source_is_refused(demo, tmp_path):
    record = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    record["source"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source has changed"):
        manifest.reproduce(tmp_path, log=lambda _: None)


def test_the_synthetic_source_is_deterministic():
    assert sources.synthetic(seed=7).sha256 == sources.synthetic(seed=7).sha256
    assert sources.synthetic(seed=7).sha256 != sources.synthetic(seed=8).sha256


def test_without_a_snapshot_analyze_says_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLSURF_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="volsurf chains fetch"):
        main(["analyze", "--ticker", "^SPX", "--out", str(tmp_path / "out")])


def test_cli_price_prints_every_engine(capsys):
    main(
        [
            "price",
            "--spot",
            "100",
            "--strike",
            "95",
            "--maturity",
            "0.5",
            "--rate",
            "0.03",
            "--vol",
            "0.25",
            "--heston",
            "0.0625,2,0.0625,0.3,-0.5",
        ]
    )
    out = capsys.readouterr().out
    for engine in (
        "Black-Scholes-Merton",
        "Leisen-Reimer",
        "Crank-Nicolson",
        "Longstaff-Schwartz",
        "Heston: COS",
        "Heston: QE Monte Carlo",
    ):
        assert engine in out
    prices = {line.split("  ")[0]: float(line.split()[-1] if "+-" not in line else line.split()[-3]) for line in out.splitlines()}
    assert prices["Heston: contour integral"] == pytest.approx(prices["Heston: COS"], abs=1e-8)


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["unknown"])


@pytest.mark.slow
def test_validation_suite_passes_in_fast_mode():
    results = run_validation(fast=True)
    assert len(results) >= 16
    assert results["passed"].all(), results.loc[~results["passed"], ["check", "result", "error"]].to_string()


def test_validation_markdown_format():
    table = pd.DataFrame(
        [{"area": "A", "check": "c", "reference": "1", "result": "1", "error": "0", "passed": True, "seconds": 0.1}]
    )
    assert "| A | c | 1 | 1 | 0 | PASS |" in to_markdown(table)
