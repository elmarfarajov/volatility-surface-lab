"""The house style for every chart in the gallery: one palette, one title block, one caption line.

The palette continues the lab's original figures (``volsurf.figures``): ink and
muted greys for text, blue, green and violet for data, and the orange accent kept
for the one thing a chart wants the eye to land on.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (the backend is chosen first)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PALETTE: dict[str, str] = {
    "ink": "#1f2933",
    "muted": "#7b8794",
    "grid": "#e4e7eb",
    "band": "#f0f4f8",
    "blue": "#2f6fdb",
    "green": "#1b9e77",
    "violet": "#7b4fb5",
    "teal": "#0f8a8a",
    "slate": "#52606d",
    "red": "#c23b3b",
    "accent": "#e0632b",
}
SERIES = (PALETTE["blue"], PALETTE["green"], PALETTE["violet"], PALETTE["teal"], PALETTE["slate"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": "#cbd2d9",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def new_figure(width: float, height: float) -> Figure:
    apply_style()
    return plt.figure(figsize=(width, height))


def title_block(fig: Figure, title: str, subtitle: str | None = None) -> None:
    height = fig.get_figheight()
    fig.suptitle(title, x=0.012, y=1 - 0.22 / height, ha="left", va="top", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    if subtitle:
        wrapped = textwrap.fill(subtitle, int(fig.get_figwidth() * 14.5))  # about the characters a line holds
        fig.text(0.012, 1 - 0.52 / height, wrapped, ha="left", va="top", fontsize=9.5, color=PALETTE["muted"])


def caption(fig: Figure, text: str) -> None:
    fig.text(0.012, 0.16 / fig.get_figheight(), text, ha="left", va="bottom", fontsize=7.5, color=PALETTE["muted"])


def tidy(ax: Axes, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> Axes:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def save(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(fig)
    return destination
