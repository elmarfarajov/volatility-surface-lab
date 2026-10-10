"""The gallery: every chart draws, the page lists exactly the charts there are, and the CLI rebuilds them."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from volsurf import gallery
from volsurf.cli import main


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.mark.parametrize("item", gallery.items(), ids=lambda item: item.filename)
def test_every_chart_draws(item):
    if not item.available():
        pytest.skip("drawn from a local option-chain snapshot, which this machine does not have")
    figure = item.build()
    assert figure.axes and item.day in gallery.DAYS


def test_file_names_are_unique_and_the_page_lists_each_once():
    names = [item.filename for item in gallery.items()]
    assert len(names) == len(set(names))
    page = gallery.markdown()
    assert all(page.count(f"({name})") == 1 for name in [f"images/{n}" for n in names])


def test_the_cli_draws_one_day_into_a_directory(tmp_path, capsys):
    main(["gallery", "--out", str(tmp_path), "--only", "expiry-limits.png"])
    assert (tmp_path / "expiry-limits.png").stat().st_size > 10_000
    assert "1 charts written" in capsys.readouterr().out


def test_the_cli_prints_seventeen_greeks(capsys):
    main(
        [
            "greeks",
            "--spot",
            "105",
            "--strike",
            "100",
            "--maturity",
            "0.5",
            "--rate",
            "0.1",
            "--vol",
            "0.36",
            "--underlying",
            "future",
        ]
    )
    out = capsys.readouterr().out
    assert "Black-76 call" in out and "0.5946286597" in out  # Haug's futures delta, 0.5946
    assert len([line for line in out.splitlines() if line.startswith("  ")]) == 17
