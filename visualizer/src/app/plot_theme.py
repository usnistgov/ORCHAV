"""Theme adapters for matplotlib and pyqtgraph surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .theme import ThemeColors, current_theme


def apply_matplotlib_theme(fig: Any, ax: Any, *, grid: bool = True) -> None:
    """Apply the current application theme to a matplotlib figure and axes."""
    theme = current_theme()
    fig.set_facecolor(theme.bg_secondary)
    ax.set_facecolor(theme.bg_secondary)
    ax.xaxis.label.set_color(theme.text_primary)
    ax.yaxis.label.set_color(theme.text_primary)
    ax.title.set_color(theme.text_primary)
    ax.tick_params(colors=theme.text_primary, which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.border_primary)
    if grid:
        ax.grid(True, color=theme.border_primary, linestyle="-", linewidth=0.5, alpha=0.55)


def apply_matplotlib_legend_theme(ax: Any) -> None:
    """Apply the current application theme to a matplotlib legend, if present."""
    legend = ax.get_legend()
    if legend is None:
        return
    theme = current_theme()
    frame = legend.get_frame()
    frame.set_facecolor(theme.bg_secondary)
    frame.set_edgecolor(theme.border_primary)
    for text in legend.get_texts():
        text.set_color(theme.text_primary)


@dataclass(frozen=True)
class PyQtGraphTheme:
    """Resolved pyqtgraph pens and styles for the current Qt theme."""

    background: str
    title_style: dict[str, str]
    label_style: dict[str, str]
    axis_pen: Any
    axis_text_pen: Any
    border_pen: Any
    scatter_pen: Any
    guide_pen: Any


def pyqtgraph_theme(theme: ThemeColors | None = None) -> PyQtGraphTheme:
    """Return pyqtgraph style tokens for a Qt theme."""
    import pyqtgraph as pg

    t = theme or current_theme()
    return PyQtGraphTheme(
        background=t.bg_secondary,
        title_style={"color": t.text_primary, "size": "11pt"},
        label_style={"color": t.text_secondary, "font-size": "9pt"},
        axis_pen=pg.mkPen(color=t.text_secondary, width=1),
        axis_text_pen=pg.mkPen(color=t.text_secondary),
        border_pen=pg.mkPen(color=t.border_primary, width=1),
        scatter_pen=pg.mkPen(t.text_secondary, width=0.3),
        guide_pen=pg.mkPen(t.border_primary, width=0.5),
    )


def apply_pyqtgraph_plot_theme(
    plot: Any,
    *,
    title: str = "",
    tokens: PyQtGraphTheme | None = None,
    show_grid: bool = True,
) -> PyQtGraphTheme:
    """Apply theme styling to a pyqtgraph PlotWidget and return used tokens."""
    style = tokens or pyqtgraph_theme()
    plot.setBackground(style.background)
    plot.setTitle(title, **style.title_style)
    if show_grid:
        plot.showGrid(x=True, y=True, alpha=0.3)
    plot.getPlotItem().getViewBox().setBorder(style.border_pen)
    for axis_name in ("bottom", "left"):
        axis = plot.getAxis(axis_name)
        axis.setPen(style.axis_pen)
        axis.setTextPen(style.axis_text_pen)
        label_text = getattr(axis, "labelText", "")
        label_units = getattr(axis, "labelUnits", "")
        if label_text:
            axis.setLabel(label_text, units=label_units or None, **style.label_style)
    return style
