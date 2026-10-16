"""Shared figure style: one palette, recessive axes, year ticks."""

from __future__ import annotations

import matplotlib.dates as mdates

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, BLUE_LIGHT, ORANGE, GRAY = "#2a78d6", "#b7d3f6", "#eb6834", "#8a8984"
AQUA, YELLOW = "#1baf7a", "#eda100"


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7)


def year_ticks(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
