"""Shared figure style used by every chart in the project."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from credit_scoring.config import Config, load_config

# Figures are written to disk by scripts and never displayed, so the
# non-interactive backend is selected here. Without it matplotlib looks for a
# display and stalls on a machine that has none, such as a CI runner.
matplotlib.use("Agg")

# Categorical colours, assigned in this fixed order. Checked with a
# colour-vision simulation: the four slots used for model comparison stay
# distinguishable under protanopia and deuteranopia.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Two hues around a neutral grey, for signed quantities such as correlation.
# The grey midpoint is what makes "no relationship" read as nothing.
DIVERGING_STEPS = ("#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f3b3b2", "#e34948", "#8f1f1f")

# One hue from pale to dark, for quantities that only grow, such as counts.
SEQUENTIAL_STEPS = ("#f0efec", "#cde2fb", "#3987e5", "#0d366b")

_style_applied = False


def series_color(index: int) -> str:
    """Return the colour for series ``index``, raising once the palette runs out."""
    if not 0 <= index < len(SERIES_COLORS):
        raise IndexError(
            f"Series index {index} is outside the {len(SERIES_COLORS)}-colour palette. "
            "Group the remaining categories instead of reusing a colour."
        )
    return SERIES_COLORS[index]


def diverging_cmap() -> LinearSegmentedColormap:
    """Return the colormap used for correlation heatmaps."""
    return LinearSegmentedColormap.from_list("credit_diverging", DIVERGING_STEPS)


def sequential_cmap() -> LinearSegmentedColormap:
    """Return the colormap used for unsigned quantities such as counts."""
    return LinearSegmentedColormap.from_list("credit_sequential", SEQUENTIAL_STEPS)


def apply_style(config: Config | None = None) -> None:
    """Apply the project figure style; later calls do nothing."""
    global _style_applied
    if _style_applied:
        return

    cfg = config or load_config()
    mstyle.use(cfg.plotting.style)
    plt.rcParams.update(
        {
            "figure.figsize": cfg.plotting.figsize_default,
            "savefig.dpi": cfg.plotting.dpi,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": GRIDLINE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    _style_applied = True


def figure_path(name: str, config: Config | None = None) -> Path:
    """Return the file path a figure called ``name`` is saved to."""
    cfg = config or load_config()
    directory = cfg.paths.absolute(cfg.paths.figures_dir)
    return directory / f"{name}.{cfg.plotting.format}"


@contextmanager
def figure(
    name: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: Sequence[float] | None = None,
    config: Config | None = None,
) -> Iterator[Axes]:
    """Create a figure, hand over its axes, then save and close it.

    Closing happens in a ``finally`` block, so a failure halfway through a batch
    of figures cannot leave them open and slowly exhaust memory.
    """
    cfg = config or load_config()
    apply_style(cfg)

    fig, ax = plt.subplots(figsize=tuple(figsize or cfg.plotting.figsize_default))
    try:
        yield ax
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        path = figure_path(name, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    finally:
        plt.close(fig)
