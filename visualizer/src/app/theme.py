"""Application-wide Qt theme tokens and stylesheet generation.

The visualizer follows Qt 6's platform color scheme by default while still
owning application-specific colors for custom controls and dense panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    """Immutable color palette for the application."""

    # Surfaces
    bg_primary: str = "#f5f6f8"
    bg_secondary: str = "#ffffff"
    bg_tertiary: str = "#ebedf0"

    # Borders
    border_primary: str = "#d0d4da"
    border_subtle: str = "#e2e5ea"

    # Text
    text_primary: str = "#1a1d23"
    text_secondary: str = "#5a6070"
    text_muted: str = "#8c93a3"

    # Accent
    accent: str = "#2563eb"
    accent_hover: str = "#3b82f6"
    accent_active: str = "#dc2626"

    # Semantic
    success: str = "#27ae60"
    success_hover: str = "#2ecc71"
    warning: str = "#f59e0b"
    error: str = "#dc2626"

    # Metadata
    name: str = "light"
    is_dark: bool = False


class ThemeMode(str, Enum):
    """Persisted application theme policy."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


# --- Spacing scale (px) ---
SPACING_XS = 2
SPACING_SM = 4
SPACING_MD = 8
SPACING_LG = 12
SPACING_XL = 16

# --- Font sizes (pt) ---
FONT_SM = 10
FONT_MD = 11
FONT_LG = 13

LIGHT_THEME = ThemeColors()
DARK_THEME = ThemeColors(
    bg_primary="#1f2329",
    bg_secondary="#272c34",
    bg_tertiary="#333a45",
    border_primary="#4b5563",
    border_subtle="#394150",
    text_primary="#f3f6fb",
    text_secondary="#c7cfdb",
    text_muted="#9aa4b2",
    accent="#60a5fa",
    accent_hover="#93c5fd",
    accent_active="#ef4444",
    success="#4ade80",
    success_hover="#86efac",
    warning="#fbbf24",
    error="#f87171",
    name="dark",
    is_dark=True,
)

# Light-token compatibility alias consumed by custom-painted widgets.
DEFAULT_THEME = LIGHT_THEME

SETTINGS_THEME_MODE_KEY = "ui/theme_mode"
_THEME_MANAGER: ThemeManager | None = None


def normalize_theme_mode(value: ThemeMode | str | None) -> ThemeMode:
    """Return a valid theme mode from persisted or UI-provided values."""
    if isinstance(value, ThemeMode):
        return value
    raw = str(value or ThemeMode.SYSTEM.value).strip().lower()
    for mode in ThemeMode:
        if raw == mode.value:
            return mode
    return ThemeMode.SYSTEM


def _is_dark_window_palette(palette: QPalette | None) -> bool:
    """Infer a dark palette when Qt reports an unknown platform scheme."""
    if palette is None:
        return False
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def theme_for_color_scheme(
    scheme: Qt.ColorScheme | None,
    *,
    fallback_palette: QPalette | None = None,
) -> ThemeColors:
    """Return tokens for a Qt color scheme, falling back to palette brightness."""
    if scheme == Qt.ColorScheme.Dark:
        return DARK_THEME
    if scheme == Qt.ColorScheme.Light:
        return LIGHT_THEME
    return DARK_THEME if _is_dark_window_palette(fallback_palette) else LIGHT_THEME


def theme_for_mode(mode: ThemeMode | str, app: QApplication | None = None) -> ThemeColors:
    """Resolve effective tokens for a persisted theme mode."""
    normalized = normalize_theme_mode(mode)
    if normalized == ThemeMode.DARK:
        return DARK_THEME
    if normalized == ThemeMode.LIGHT:
        return LIGHT_THEME

    app = app or QApplication.instance()
    hints = app.styleHints() if app is not None else None
    scheme = hints.colorScheme() if hints is not None and hasattr(hints, "colorScheme") else None
    palette = app.palette() if app is not None else None
    return theme_for_color_scheme(scheme, fallback_palette=palette)


def generate_application_palette(theme: ThemeColors | None = None) -> QPalette:
    """Return a Qt palette matching the supplied ORCHAV theme tokens."""
    t = theme or DEFAULT_THEME
    palette = QPalette()

    def set_color(role: QPalette.ColorRole, color: str) -> None:
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, role, QColor(color))

    set_color(QPalette.ColorRole.Window, t.bg_primary)
    set_color(QPalette.ColorRole.WindowText, t.text_primary)
    set_color(QPalette.ColorRole.Base, t.bg_secondary)
    set_color(QPalette.ColorRole.AlternateBase, t.bg_primary)
    set_color(QPalette.ColorRole.ToolTipBase, t.bg_secondary)
    set_color(QPalette.ColorRole.ToolTipText, t.text_primary)
    set_color(QPalette.ColorRole.Text, t.text_primary)
    set_color(QPalette.ColorRole.Button, t.bg_secondary)
    set_color(QPalette.ColorRole.ButtonText, t.text_primary)
    set_color(QPalette.ColorRole.BrightText, t.error)
    set_color(QPalette.ColorRole.Link, t.accent)
    set_color(QPalette.ColorRole.Highlight, t.accent)
    set_color(QPalette.ColorRole.HighlightedText, "#ffffff")
    set_color(QPalette.ColorRole.PlaceholderText, t.text_muted)

    disabled = QPalette.ColorGroup.Disabled
    palette.setColor(disabled, QPalette.ColorRole.Window, QColor(t.bg_primary))
    palette.setColor(disabled, QPalette.ColorRole.WindowText, QColor(t.text_muted))
    palette.setColor(disabled, QPalette.ColorRole.Base, QColor(t.bg_tertiary))
    palette.setColor(disabled, QPalette.ColorRole.Text, QColor(t.text_muted))
    palette.setColor(disabled, QPalette.ColorRole.Button, QColor(t.bg_tertiary))
    palette.setColor(disabled, QPalette.ColorRole.ButtonText, QColor(t.text_muted))
    palette.setColor(disabled, QPalette.ColorRole.Highlight, QColor(t.border_primary))
    palette.setColor(disabled, QPalette.ColorRole.HighlightedText, QColor(t.text_muted))

    return palette


def generate_application_stylesheet(theme: ThemeColors | None = None) -> str:
    """Return a single QSS string covering all common widget types.

    This stylesheet is applied once at application startup via
    ``QApplication.instance().setStyleSheet()``.  Panel-specific overrides
    (e.g. the export-button green) are still allowed. Qt CSS specificity
    rules let more-specific selectors win.
    """
    t = theme or DEFAULT_THEME

    return f"""
        /* Global */
        QWidget {{
            font-size: {FONT_MD}px;
            color: {t.text_primary};
        }}
        QMainWindow, QDialog {{
            background-color: {t.bg_primary};
            color: {t.text_primary};
        }}
        QToolTip {{
            background-color: {t.bg_secondary};
            color: {t.text_primary};
            border: 1px solid {t.border_primary};
            padding: 4px 6px;
        }}
        QMenuBar {{
            background: {t.bg_primary};
            color: {t.text_primary};
            border-bottom: 1px solid {t.border_subtle};
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 4px 10px;
        }}
        QMenuBar::item:selected {{
            background: {t.bg_tertiary};
        }}
        QMenu {{
            background: {t.bg_secondary};
            color: {t.text_primary};
            border: 1px solid {t.border_primary};
        }}
        QMenu::item {{
            padding: 4px 22px 4px 22px;
        }}
        QMenu::item:selected {{
            background: {t.accent};
            color: #ffffff;
        }}
        QMenu::separator {{
            height: 1px;
            background: {t.border_subtle};
            margin: 4px 8px;
        }}
        QStatusBar {{
            background: {t.bg_primary};
            color: {t.text_secondary};
            border-top: 1px solid {t.border_subtle};
        }}

        /* QGroupBox */
        QGroupBox {{
            font-weight: normal;
            font-size: {FONT_MD}px;
            border: 1px solid {t.border_primary};
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 6px;
            background-color: {t.bg_secondary};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: {t.text_secondary};
            background-color: {t.bg_secondary};
            font-weight: normal;
            font-size: {FONT_MD}px;
        }}

        /* QTabWidget / QTabBar */
        QTabWidget::pane {{
            border: 1px solid {t.border_primary};
            border-top: none;
            background: {t.bg_primary};
        }}
        QTabBar::tab {{
            background: {t.bg_tertiary};
            border: 1px solid {t.border_primary};
            border-bottom: none;
            padding: 6px 14px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            color: {t.text_secondary};
            font-size: {FONT_MD}px;
        }}
        QTabBar::tab:selected {{
            background: {t.bg_secondary};
            color: {t.accent};
            font-weight: bold;
            border-bottom: 2px solid {t.accent};
        }}
        QTabBar::tab:hover:!selected {{
            background: {t.bg_secondary};
            color: {t.text_primary};
        }}

        /* QPushButton */
        QPushButton {{
            background-color: {t.bg_secondary};
            color: {t.text_primary};
            border: 1px solid {t.border_primary};
            padding: 4px 10px;
            border-radius: 4px;
            font-size: {FONT_MD}px;
        }}
        QPushButton:hover {{
            background-color: {t.bg_tertiary};
            border-color: {t.accent};
        }}
        QPushButton:pressed {{
            background-color: {t.border_subtle};
        }}
        QPushButton:checked {{
            background-color: {t.accent};
            color: #ffffff;
            border-color: {t.accent};
        }}
        QPushButton:disabled {{
            background-color: {t.bg_tertiary};
            color: {t.text_muted};
            border-color: {t.border_subtle};
        }}

        /* QSlider (horizontal) */
        QSlider::groove:horizontal {{
            border: 1px solid {t.border_primary};
            height: 6px;
            background: {t.bg_tertiary};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {t.text_primary};
            border: 1px solid {t.text_primary};
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {t.accent};
            border-color: {t.accent};
        }}

        /* QSpinBox / QDoubleSpinBox */
        QSpinBox, QDoubleSpinBox {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            padding: 3px 4px;
            background: {t.bg_secondary};
            color: {t.text_primary};
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            border: none;
            background: {t.bg_tertiary};
            border-radius: 2px;
        }}

        /* QCheckBox */
        QCheckBox {{
            color: {t.text_primary};
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {t.border_primary};
            background: {t.bg_secondary};
            border-radius: 3px;
        }}
        QCheckBox::indicator:checked {{
            border-color: {t.accent};
            background: {t.accent};
        }}
        QCheckBox:disabled {{
            color: {t.text_muted};
        }}

        /* QRadioButton */
        QRadioButton {{
            color: {t.text_primary};
            spacing: 6px;
            font-size: {FONT_MD}px;
        }}
        QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            border: 2px solid {t.border_primary};
            border-radius: 8px;
            background: {t.bg_secondary};
        }}
        QRadioButton::indicator:checked {{
            border-color: {t.accent};
            background: {t.accent};
        }}
        QRadioButton:disabled {{
            color: {t.text_muted};
        }}

        /* QComboBox */
        QComboBox {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            padding: 3px 8px;
            background: {t.bg_secondary};
            color: {t.text_primary};
            min-height: 22px;
        }}
        QComboBox:hover {{
            border-color: {t.accent};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            border: 1px solid {t.border_primary};
            background: {t.bg_secondary};
            selection-background-color: {t.accent};
            selection-color: #ffffff;
        }}
        QComboBox:disabled {{
            background: {t.bg_tertiary};
            color: {t.text_muted};
        }}

        /* QScrollArea */
        QAbstractScrollArea, QScrollArea {{
            border: none;
            background: {t.bg_secondary};
            color: {t.text_primary};
        }}
        QListView, QTableView, QTableWidget, QTreeView {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            background: {t.bg_secondary};
            color: {t.text_primary};
            alternate-background-color: {t.bg_primary};
            selection-background-color: {t.accent};
            selection-color: #ffffff;
        }}
        QListView::item, QTableView::item, QTreeView::item {{
            padding: 3px 4px;
        }}
        QListView::item:hover:!selected,
        QTableView::item:hover:!selected,
        QTreeView::item:hover:!selected {{
            background: {t.bg_tertiary};
        }}
        QListView#materialFilterList {{
            min-height: 72px;
            max-height: 120px;
        }}

        /* QScrollBar (thin overlay style) */
        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {t.border_primary};
            min-height: 30px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t.text_muted};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {t.border_primary};
            min-width: 30px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {t.text_muted};
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}

        /* QLabel */
        QLabel {{
            color: {t.text_primary};
        }}
        QLabel[role="secondary"] {{
            color: {t.text_secondary};
        }}
        QLabel[role="muted"] {{
            color: {t.text_muted};
        }}
        QLabel[role="accent"] {{
            color: {t.accent};
        }}
        QLabel[role="success"] {{
            color: {t.success};
        }}
        QLabel[role="warning"] {{
            color: {t.warning};
        }}
        QLabel[role="error"] {{
            color: {t.error};
        }}
        QLabel#mpcInfoLabel {{
            color: {t.text_secondary};
            background-color: {t.bg_primary};
            padding: 4px 6px;
            border-radius: 4px;
            border: 1px solid {t.border_subtle};
        }}

        /* QToolButton (CollapsibleSection toggle) */
        QToolButton {{
            background: {t.bg_tertiary};
            border: 1px solid {t.border_subtle};
            border-radius: 4px;
            padding: 4px 8px;
            font-weight: bold;
            font-size: {FONT_MD}px;
            color: {t.text_primary};
            text-align: left;
        }}
        QToolButton:hover {{
            background: {t.border_subtle};
        }}
        QToolButton:checked {{
            background: {t.bg_secondary};
            border-color: {t.border_primary};
        }}

        /* QFrame separator */
        QFrame[frameShape="4"] {{
            color: {t.border_subtle};
            max-height: 1px;
        }}

        /* QTreeView */
        QTreeView {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            background: {t.bg_secondary};
            alternate-background-color: {t.bg_primary};
        }}
        QTreeView::item:selected {{
            background: {t.accent};
            color: #ffffff;
        }}
        QTreeView::item:hover:!selected {{
            background: {t.bg_tertiary};
        }}
        QHeaderView::section {{
            background: {t.bg_tertiary};
            color: {t.text_primary};
            border: 1px solid {t.border_primary};
            padding: 4px;
            font-weight: bold;
            font-size: {FONT_SM}px;
        }}

        /* QLineEdit */
        QLineEdit {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            padding: 4px 6px;
            background: {t.bg_secondary};
            color: {t.text_primary};
        }}
        QLineEdit:focus {{
            border-color: {t.accent};
        }}

        /* QProgressBar */
        QProgressBar {{
            border: 1px solid {t.border_primary};
            border-radius: 4px;
            background: {t.bg_tertiary};
            color: {t.text_primary};
            text-align: center;
            font-size: {FONT_SM}px;
        }}
        QProgressBar::chunk {{
            background: {t.accent};
            border-radius: 3px;
        }}
    """


class ThemeManager(QObject):
    """Apply and persist the visualizer's Qt theme mode."""

    theme_changed = Signal(object)

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._mode = ThemeMode.SYSTEM
        self._current_theme = DEFAULT_THEME
        self._installed_app: QApplication | None = None
        self._platform_palette: QPalette | None = None
        self._applied_palette: QPalette | None = None
        self._applied_stylesheet: str | None = None
        self._scheme_connected = False
        self._applying = False

    @property
    def mode(self) -> ThemeMode:
        """Return the active theme policy."""
        return self._mode

    @property
    def current_theme(self) -> ThemeColors:
        """Return currently effective theme tokens."""
        return self._current_theme

    def _qsettings(self) -> QSettings:
        if self._settings is None:
            self._settings = QSettings()
        return self._settings

    def load_mode(self) -> ThemeMode:
        """Load the persisted mode, defaulting to System."""
        self._mode = normalize_theme_mode(
            self._qsettings().value(SETTINGS_THEME_MODE_KEY, ThemeMode.SYSTEM.value)
        )
        return self._mode

    def install(
        self,
        app: QApplication | None = None,
        *,
        mode: ThemeMode | str | None = None,
        persist: bool = False,
    ) -> ThemeColors:
        """Install hooks and apply a theme to the QApplication."""
        app = app or QApplication.instance()
        same_app = app is not None and app is self._installed_app
        previous_mode = self._mode
        if app is not None and app is not self._installed_app:
            self._installed_app = app
            self._platform_palette = QPalette(app.palette())
            self._applied_palette = None
            self._applied_stylesheet = None
            self._scheme_connected = False
            app.installEventFilter(self)
            hints = app.styleHints()
            signal = getattr(hints, "colorSchemeChanged", None)
            connect = getattr(signal, "connect", None)
            if callable(connect) and not self._scheme_connected:
                connect(lambda *_args: self._on_platform_scheme_changed())
                self._scheme_connected = True

        if mode is None:
            self.load_mode()
        else:
            self._mode = normalize_theme_mode(mode)

        if (
            same_app
            and self._mode == previous_mode
            and self._application_matches_applied_theme(app)
        ):
            if persist:
                self._qsettings().setValue(SETTINGS_THEME_MODE_KEY, self._mode.value)
            return self._current_theme
        return self.apply_current_theme(app, persist=persist)

    def set_mode(self, mode: ThemeMode | str, *, persist: bool = True) -> ThemeColors:
        """Set a new theme policy and reapply the app palette/stylesheet."""
        self._mode = normalize_theme_mode(mode)
        return self.apply_current_theme(self._installed_app, persist=persist)

    def apply_current_theme(
        self,
        app: QApplication | None = None,
        *,
        persist: bool = False,
    ) -> ThemeColors:
        """Apply the current mode and return effective tokens."""
        app = app or self._installed_app or QApplication.instance()
        if persist:
            self._qsettings().setValue(SETTINGS_THEME_MODE_KEY, self._mode.value)

        if app is None:
            self._current_theme = theme_for_mode(self._mode, None)
            return self._current_theme

        self._applying = True
        try:
            hints = app.styleHints()
            if hasattr(hints, "setColorScheme"):
                if self._mode == ThemeMode.SYSTEM:
                    hints.setColorScheme(Qt.ColorScheme.Unknown)
                elif self._mode == ThemeMode.DARK:
                    hints.setColorScheme(Qt.ColorScheme.Dark)
                else:
                    hints.setColorScheme(Qt.ColorScheme.Light)

            if self._mode == ThemeMode.SYSTEM:
                scheme = hints.colorScheme() if hasattr(hints, "colorScheme") else None
                fallback_palette = self._platform_palette or app.palette()
                theme = theme_for_color_scheme(scheme, fallback_palette=fallback_palette)
            else:
                theme = theme_for_mode(self._mode, app)

            self._current_theme = theme
            app.setPalette(generate_application_palette(theme))
            app.setStyleSheet(generate_application_stylesheet(theme))
            self._applied_palette = QPalette(app.palette())
            self._applied_stylesheet = app.styleSheet()
        finally:
            self._applying = False

        self.theme_changed.emit(theme)
        return theme

    def _on_platform_scheme_changed(self) -> None:
        """Refresh System mode when Qt reports an OS color-scheme change."""
        if self._mode == ThemeMode.SYSTEM and not self._applying:
            self.apply_current_theme(self._installed_app, persist=False)

    def _application_matches_applied_theme(self, app: QApplication | None) -> bool:
        """Return whether the application still has the palette and QSS we applied."""
        if app is None or self._applied_palette is None or self._applied_stylesheet is None:
            return False
        try:
            return (
                app.palette() == self._applied_palette
                and app.styleSheet() == self._applied_stylesheet
            )
        except (AttributeError, RuntimeError, TypeError):
            return False

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        """Reapply System mode when Qt/native palette changes underneath us."""
        if (
            obj is self._installed_app
            and self._mode == ThemeMode.SYSTEM
            and not self._applying
            and event.type()
            in {
                QEvent.Type.ApplicationPaletteChange,
                QEvent.Type.PaletteChange,
                QEvent.Type.StyleChange,
            }
        ):
            if self._application_matches_applied_theme(self._installed_app):
                return super().eventFilter(obj, event)
            self._platform_palette = QPalette(self._installed_app.palette())
            self.apply_current_theme(self._installed_app, persist=False)
        return super().eventFilter(obj, event)


def get_theme_manager() -> ThemeManager:
    """Return the process-wide theme manager."""
    global _THEME_MANAGER
    if _THEME_MANAGER is None:
        _THEME_MANAGER = ThemeManager()
    return _THEME_MANAGER


def current_theme() -> ThemeColors:
    """Return currently effective application theme tokens."""
    return get_theme_manager().current_theme
