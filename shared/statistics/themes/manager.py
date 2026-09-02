"""Runtime theme manager for shared statistics figures.

The singleton ``theme_manager`` lets visualizer panels and analysis tools switch
between registered figure themes and listen for color/colormap changes.
"""

import threading
from dataclasses import replace
from typing import Callable, Dict, List, Optional

from shared.logging import get_logger

from .base import Theme
from .dark import DARK_THEME
from .default import DEFAULT_THEME

logger = get_logger(__name__)


class ThemeManager:
    """Singleton manager for runtime figure-theme switching.

    Thread-safe implementation that notifies all registered listeners
    when the theme changes.
    """

    _instance: Optional["ThemeManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ThemeManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the manager (called only once)."""
        self._current_theme: Theme = DEFAULT_THEME
        self._listeners: List[Callable[[Theme], None]] = []
        self._themes: Dict[str, Theme] = {
            "default": DEFAULT_THEME,
            "dark": DARK_THEME,
        }

    @property
    def current(self) -> Theme:
        """Get the current theme."""
        return self._current_theme

    @property
    def available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self._themes.keys())

    def set_theme(self, name: str) -> None:
        """
        Switch to a different theme and notify all listeners.

        Args:
            name: Theme name ("default" or "dark")

        Raises:
            ValueError: If theme name is not recognized
        """
        if name not in self._themes:
            raise ValueError(f"Unknown theme: {name}. Available: {list(self._themes.keys())}")

        self._current_theme = self._themes[name]
        self._notify_listeners()

    def set_colormap(self, colormap_type: str, name: str) -> None:
        """
        Change colormap without switching entire theme.

        This allows users to customize colormaps independently of the
        overall theme.

        Args:
            colormap_type: "continuous" or "categorical"
            name: Colormap name (e.g., "viridis", "plasma", "tab10")

        Raises:
            ValueError: If colormap_type is not recognized
        """
        if colormap_type == "continuous":
            self._current_theme = replace(self._current_theme, continuous_colormap=name)
        elif colormap_type == "categorical":
            self._current_theme = replace(self._current_theme, categorical_colormap=name)
        else:
            raise ValueError(
                f"Unknown colormap type: {colormap_type}. Use 'continuous' or 'categorical'"
            )

        self._notify_listeners()

    def register_theme(self, theme: Theme) -> None:
        """
        Register a custom theme.

        Args:
            theme: Theme instance to register
        """
        self._themes[theme.name] = theme

    def add_listener(self, callback: Callable[[Theme], None]) -> None:
        """
        Register callback for theme changes.

        The callback will be called with the new Theme instance whenever
        set_theme() or set_colormap() is called.

        Args:
            callback: Function that takes a Theme argument
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Theme], None]) -> None:
        """
        Unregister a theme change callback.

        Args:
            callback: Previously registered callback to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of theme change."""
        for listener in self._listeners:
            try:
                listener(self._current_theme)
            except Exception as exc:  # noqa: BLE001
                # Theme listeners are UI/plugin callbacks; isolate one failure
                # so the manager can notify the remaining listeners.
                logger.warning("Theme listener error: %s", exc)


# Global singleton instance
theme_manager = ThemeManager()
