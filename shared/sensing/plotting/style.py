"""Shared ORCHAV styling for exported scientific sensing figures.

Opaque light backgrounds remain readable in both light and dark documentation.
The context is local to each figure call; user Matplotlib settings are restored.
"""

from functools import wraps

import matplotlib as mpl
from cycler import cycler

INK = "#17233c"
MUTED = "#5d687b"
BLUE = "#2865b0"
TEAL = "#00866b"
AMBER = "#a66b00"
RED = "#b5475a"
ORANGE = "#bf5b13"
PURPLE = "#6251a3"
PALETTE = (BLUE, ORANGE, TEAL, PURPLE, RED, "#197a78")

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 10,
    "axes.labelpad": 8,
    "axes.edgecolor": "#cbd5e1",
    "axes.facecolor": "#f6f8fc",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=PALETTE),
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": "#cbd5e1",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#e2e8f0",
    "legend.borderpad": 0.6,
    "image.cmap": "viridis",
    "svg.fonttype": "none",
}


def sensing_figure_style(function):
    """Apply the export theme without modifying data or global plotting state."""

    @wraps(function)
    def styled(*args, **kwargs):
        with mpl.rc_context(STYLE):
            return function(*args, **kwargs)

    return styled
