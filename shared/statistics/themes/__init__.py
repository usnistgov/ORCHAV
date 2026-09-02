"""Theme definitions for shared statistics figures.

Themes centralize colors and colormaps for plots produced from statistics and
material analyses. UI widget styling remains owned by the visualizer.
"""

from .base import (
    AVAILABLE_CATEGORICAL_COLORMAPS,
    AVAILABLE_CONTINUOUS_COLORMAPS,
    Theme,
)
from .dark import DARK_THEME
from .default import DEFAULT_THEME
from .manager import ThemeManager, theme_manager

__all__ = [
    # Theme class
    "Theme",
    # Pre-built themes
    "DEFAULT_THEME",
    "DARK_THEME",
    # Manager
    "ThemeManager",
    "theme_manager",
    # Available colormaps
    "AVAILABLE_CONTINUOUS_COLORMAPS",
    "AVAILABLE_CATEGORICAL_COLORMAPS",
]
