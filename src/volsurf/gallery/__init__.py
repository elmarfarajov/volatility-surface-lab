"""The chart gallery: every figure the documentation shows, rebuilt from source with ``volsurf gallery``.

Each chart is a :class:`Item` - the file it writes, what it shows, the function
that draws it, and the day of the build it belongs to - so the gallery page
(``docs/GALLERY.md``) and the images cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from . import analytic
from .style import save

DAYS: dict[int, str] = {1: "The analytic foundation"}


@dataclass(frozen=True)
class Item:
    filename: str
    title: str
    description: str
    build: Callable[[], Figure]
    day: int


def items() -> tuple[Item, ...]:
    return (
        Item(
            "greek-atlas.png",
            "Sixteen Greeks of a European call",
            "Every Greek against spot, three months, one month and one week from expiry.",
            analytic.greek_atlas,
            1,
        ),
        Item(
            "greek-references.png",
            "Every Greek against an independent reference",
            "Arbitrary-precision differentiation and Haug's worked examples.",
            analytic.reference_agreement,
            1,
        ),
        Item(
            "tail-precision.png",
            "Prices far in the wings",
            "The engine against 60-digit arithmetic, through 270 orders of magnitude.",
            analytic.tail_precision,
            1,
        ),
        Item(
            "pde-residual.png",
            "Every price solves the Black-Scholes equation",
            "The PDE residual from the analytic Greeks, over moneyness and maturity.",
            analytic.pde_residual,
            1,
        ),
        Item(
            "carry-conventions.png",
            "One model, five underlyings",
            "A stock, an index, a future, a currency and a margined future, through the cost of carry.",
            analytic.carry_conventions,
            1,
        ),
        Item(
            "expiry-limits.png",
            "The Greeks at expiry",
            "Price, delta and theta reaching their limits as time to expiry falls to zero.",
            analytic.expiry_limits,
            1,
        ),
    )


def build(out_dir: str | Path = "docs/images", only: str | None = None) -> list[Path]:
    """Draw every chart (or the one named, or one day's) into ``out_dir``."""
    written = []
    for item in items():
        if only and only not in (item.filename, str(item.day)):
            continue
        written.append(save(item.build(), Path(out_dir) / item.filename))
    return written


def markdown(prefix: str = "images") -> str:
    """The gallery page, generated, so it always lists exactly the charts there are."""
    lines = [
        "# Gallery",
        "",
        "Every chart is rebuilt from source with `volsurf gallery`. The figures of the live S&P 500 analysis "
        "(the surface, the smiles, the Heston fit, the hedging laboratory) come from `volsurf analyze` and are shown "
        "in the README.",
        "",
    ]
    for day, title in DAYS.items():
        lines += [f"## Day {day}: {title}", ""]
        for item in items():
            if item.day == day:
                lines += [f"**{item.title}** - {item.description}", "", f"![{item.title}]({prefix}/{item.filename})", ""]
    return "\n".join(lines)
