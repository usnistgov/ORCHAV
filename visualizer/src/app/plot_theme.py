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
    from matplotlib.colors import to_rgba

    for text in (*fig.texts, *ax.texts):
        neutral = getattr(text, "_orchav_neutral_text", None)
        if neutral is None:
            red, green, blue, _alpha = to_rgba(text.get_color())
            neutral = red == green == blue
            text._orchav_neutral_text = neutral
        if neutral:
            text.set_color(theme.text_primary)
    for table in ax.tables:
        for cell in table.get_celld().values():
            cell.get_text().set_color(theme.text_primary)
            cell.set_facecolor(theme.bg_secondary)
            cell.set_edgecolor(theme.border_primary)
    for collection in ax.collections:
        if collection.get_gid() == "orchav-neutral":
            collection.set_color(theme.text_primary)
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


def path_stats_style(theme):
    return f"""
QGroupBox#PathStatistics, QGroupBox#PathStatisticsGraphs {{
    background: {theme.bg_primary}; color: {theme.text_primary}; border: none;
    font-family: 'DejaVu Sans'; font-size: 12px;
}}
QGroupBox#PathStatistics QLabel, QGroupBox#PathStatisticsGraphs QLabel {{
    background: transparent; color: {theme.text_primary};
}}
QLabel#pathEyebrow {{ color: {theme.text_secondary}; font-size: 10px; font-weight: 600; }}
QLabel#pathTitle {{ color: {theme.text_primary}; font-size: 24px; font-weight: 700; }}
QLabel#pathScope {{ color: {theme.text_secondary}; font-size: 12px; }}
QGroupBox#PathStatisticsGraphs QGroupBox {{ border: none; background: transparent; }}
QGroupBox#PathStatisticsGraphs QFrame#pathPlotCard {{
    background: {theme.bg_secondary}; border: 1px solid {theme.border_subtle}; border-radius: 9px;
}}
QLabel#pathPlotTitle {{ font-size: 13px; font-weight: 600; color: {theme.text_primary}; }}
QLabel#pathPlotNote {{ font-size: 11px; color: {theme.text_secondary}; }}
QLabel#statisticsChartMessage {{ color: {theme.text_secondary}; font-size: 12px; padding: 16px; }}
QGroupBox#PathStatisticsGraphs QTabWidget::pane {{ border: none; background: transparent; }}
QGroupBox#PathStatisticsGraphs QTabBar::tab {{
    background: transparent; color: {theme.text_secondary}; border: none;
    border-bottom: 2px solid transparent; padding: 8px 13px; font-size: 12px;
}}
QGroupBox#PathStatisticsGraphs QTabBar::tab:selected {{
    color: {theme.accent}; font-weight: 600; border-bottom: 2px solid {theme.accent};
}}
QGroupBox#PathStatisticsGraphs QTabBar::tab:disabled {{ color: {theme.text_muted}; }}
QGroupBox#PathStatisticsGraphs QScrollArea,
QGroupBox#PathStatisticsGraphs QScrollArea > QWidget > QWidget {{ background: transparent; }}
QGroupBox#PathStatistics QPushButton, QGroupBox#PathStatisticsGraphs QPushButton {{
    background: {theme.bg_secondary}; color: {theme.text_secondary}; border: 1px solid {theme.border_primary};
    border-radius: 6px; padding: 7px 11px;
}}
QGroupBox#PathStatistics QPushButton:hover {{ background: {theme.bg_tertiary}; }}


            QGroupBox#PathStatistics {{ background: {theme.bg_primary}; color: {theme.text_primary}; border: none; font-family: 'DejaVu Sans'; }}
            QGroupBox#PathStatistics QLabel {{ background: transparent; color: {theme.text_primary}; }}
            QGroupBox#PathStatistics QGroupBox {{ color: {theme.text_secondary}; border: 1px solid {theme.border_subtle}; border-radius: 7px; margin-top: 8px; padding-top: 6px; }}
            QGroupBox#PathStatistics QFrame#pathMetricCard {{ background: {theme.bg_secondary}; border: 1px solid {theme.border_subtle}; border-radius: 8px; border-top: 3px solid {theme.accent}; }}
            QGroupBox#PathStatistics QLabel#pathMetricName {{ font-size: 10px; color: {theme.text_secondary}; font-weight: 600; }}
            QGroupBox#PathStatistics QLabel#pathMetricValue {{ font-size: 23px; color: {theme.text_primary}; font-weight: 700; }}
            QGroupBox#PathStatistics QLabel#pathMetricCaption {{ font-size: 10px; color: {theme.text_secondary}; }}
            QGroupBox#PathStatistics QPushButton {{ background: {theme.bg_secondary}; color: {theme.text_secondary}; border: 1px solid {theme.border_primary}; border-radius: 6px; padding: 6px 9px; }}
            QGroupBox#PathStatistics QPushButton:hover {{ background: {theme.bg_tertiary}; }}
        """


def dashboard_series_color(color: str, theme: ThemeColors | None = None) -> str:
    """Preserve dashboard series identities with readable dark-theme contrasts."""
    if not (theme or current_theme()).is_dark:
        return color
    return {
        "#2865b0": "#83b9ff",
        "#00866b": "#4ed4ad",
        "#a66b00": "#eeb958",
        "#b5475a": "#ff8b9e",
        "#7053ad": "#b9a1f3",
        "#078596": "#6ad6d0",
        "#64748b": "#aab9cc",
        "#7552a3": "#c4a0ef",
        "#bf5b13": "#ffb477",
        "#dce4ee": "#4b5563",
        "#e5ebf2": "#394150",
    }.get(color, color)
