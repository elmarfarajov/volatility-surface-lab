import pandas as pd
import pytest

from volsurf.cli import build_parser, main
from volsurf.pipeline import AnalysisConfig, run_analysis
from volsurf.report import write_report
from volsurf.validation import run_validation, to_markdown


def test_end_to_end_report(tmp_path, noisy_snapshot):
    config = AnalysisConfig(heston_hedge_paths=400, heston_rebalances=(3, 9), bs_hedge_paths=2_000, bs_rebalances=(4, 16))
    result = run_analysis(noisy_snapshot, config, log=lambda _: None)
    path = write_report(result, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "Hedging lab" in html and "Heston calibration" in html
    for name in ("vol_surface_3d", "smiles", "term_structure", "local_vol", "densities", "calibration_residuals", "hedging_lab"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 10_000
    assert len(result.svi_fits) == len(noisy_snapshot.expiries)


def test_cli_demo_without_hedging(tmp_path, capsys):
    main(["demo", "--out", str(tmp_path), "--no-hedging"])
    assert (tmp_path / "report.html").exists()
    residuals = pd.read_csv(tmp_path / "heston_residuals.csv")
    assert residuals["error"].abs().mean() < 0.01
    assert "Report:" in capsys.readouterr().out


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
    for engine in ("Black-Scholes", "Leisen-Reimer", "Longstaff-Schwartz", "Heston: COS", "Heston: QE Monte Carlo"):
        assert engine in out


def test_parser_rejects_unknown_command():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["unknown"])


@pytest.mark.slow
def test_validation_suite_passes_in_fast_mode():
    results = run_validation(fast=True)
    assert len(results) >= 16
    assert results["passed"].all(), results.loc[~results["passed"], ["check", "result", "error"]].to_string()


def test_replay_saved_chain(tmp_path, true_params, capsys):
    import json

    from volsurf.market_data import synthetic_chain

    raw = synthetic_chain(true_params, maturities_days=(30, 90, 180, 365, 540), strikes_per_expiry=21)
    raw.to_csv(tmp_path / "raw_chain.csv", index=False)
    meta = {"ticker": "REPLAY", "spot": 5000.0, "as_of": "2026-09-15T20:00:00+00:00"}
    (tmp_path / "raw_chain.json").write_text(json.dumps(meta), encoding="utf-8")
    main(["analyze", "--raw-csv", str(tmp_path / "raw_chain.csv"), "--out", str(tmp_path / "out"), "--no-hedging"])
    assert "REPLAY" in (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "Replaying saved REPLAY chain" in capsys.readouterr().out


def test_validation_markdown_format():
    table = pd.DataFrame(
        [{"area": "A", "check": "c", "reference": "1", "result": "1", "error": "0", "passed": True, "seconds": 0.1}]
    )
    assert "| A | c | 1 | 1 | 0 | PASS |" in to_markdown(table)
