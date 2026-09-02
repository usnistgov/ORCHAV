"""Tests for the central theme module."""

import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLabel as _REAL_QLABEL


class _SignalDouble:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _StyleHintsDouble:
    def __init__(self):
        from PySide6.QtCore import Qt

        self._scheme = Qt.ColorScheme.Unknown
        self.colorSchemeChanged = _SignalDouble()

    def colorScheme(self):  # noqa: N802 - Qt-compatible test double
        return self._scheme

    def setColorScheme(self, scheme):  # noqa: N802 - Qt-compatible test double
        self._scheme = scheme


class _ApplicationDouble(QObject):
    """Minimal application surface used to test ThemeManager without global Qt state."""

    def __init__(self, palette):
        from PySide6.QtGui import QPalette

        super().__init__()
        self._palette = QPalette(palette)
        self._stylesheet = ""
        self._style_hints = _StyleHintsDouble()
        self.event_filters = []
        self.palette_apply_count = 0
        self.stylesheet_apply_count = 0

    def palette(self):
        from PySide6.QtGui import QPalette

        return QPalette(self._palette)

    def setPalette(self, palette):  # noqa: N802 - Qt-compatible test double
        from PySide6.QtGui import QPalette

        self._palette = QPalette(palette)
        self.palette_apply_count += 1

    def styleSheet(self):  # noqa: N802 - Qt-compatible test double
        return self._stylesheet

    def setStyleSheet(self, stylesheet):  # noqa: N802 - Qt-compatible test double
        self._stylesheet = stylesheet
        self.stylesheet_apply_count += 1

    def styleHints(self):  # noqa: N802 - Qt-compatible test double
        return self._style_hints

    def installEventFilter(self, event_filter):  # noqa: N802 - Qt-compatible test double
        self.event_filters.append(event_filter)


def test_theme_colors_is_frozen():
    from visualizer.src.app.theme import ThemeColors

    t = ThemeColors()
    with pytest.raises(AttributeError):
        t.accent = "#000000"


def test_default_theme_has_expected_tokens():
    from visualizer.src.app.theme import DEFAULT_THEME

    assert DEFAULT_THEME.bg_primary.startswith("#")
    assert DEFAULT_THEME.accent.startswith("#")
    assert DEFAULT_THEME.text_primary.startswith("#")


def test_generate_application_stylesheet_returns_nonempty():
    from visualizer.src.app.theme import generate_application_stylesheet

    css = generate_application_stylesheet()
    assert isinstance(css, str)
    assert len(css) > 100
    assert "QGroupBox" in css
    assert "QTabWidget" in css
    assert "QPushButton" in css
    assert "QSlider" in css
    assert "QCheckBox" in css
    assert "QScrollBar" in css
    assert "QMenuBar" in css
    assert "QListView" in css


def test_generate_application_stylesheet_includes_semantic_label_roles():
    from visualizer.src.app.theme import DARK_THEME, generate_application_stylesheet

    css = generate_application_stylesheet(DARK_THEME)
    assert 'QLabel[role="accent"]' in css
    assert 'QLabel[role="success"]' in css
    assert 'QLabel[role="warning"]' in css
    assert 'QLabel[role="error"]' in css
    assert DARK_THEME.success in css
    assert DARK_THEME.warning in css
    assert DARK_THEME.error in css


def test_empty_display_value_is_ascii():
    from visualizer.src.panels.ui_theme import EMPTY_DISPLAY_VALUE

    assert EMPTY_DISPLAY_VALUE == "--"
    assert EMPTY_DISPLAY_VALUE.isascii()


def test_configure_label_uses_platform_fixed_font(qtbot):
    from PySide6.QtGui import QFontDatabase

    from visualizer.src.panels.ui_theme import configure_label

    label = _REAL_QLABEL()
    qtbot.addWidget(label)
    configure_label(label, monospace=True)

    expected = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    assert label.font().family() == expected.family()


def test_compact_text_edit_style_uses_platform_fixed_font(qapp):
    from PySide6.QtGui import QFontDatabase

    from visualizer.src.panels.ui_theme import compact_text_edit_style

    expected = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    style = compact_text_edit_style(monospace=True)

    assert f'font-family: "{expected.family()}";' in style
    assert "Consolas" not in style
    assert "font-family:" not in compact_text_edit_style(monospace=False)


def test_generate_application_stylesheet_with_custom_theme():
    from visualizer.src.app.theme import ThemeColors, generate_application_stylesheet

    custom = ThemeColors(accent="#ff0000")
    css = generate_application_stylesheet(custom)
    assert "#ff0000" in css


def test_dark_theme_generates_dark_stylesheet():
    from visualizer.src.app.theme import DARK_THEME, generate_application_stylesheet

    css = generate_application_stylesheet(DARK_THEME)
    assert DARK_THEME.bg_primary in css
    assert DARK_THEME.text_primary in css


def test_generate_application_palette_uses_theme_roles():
    from PySide6.QtGui import QPalette

    from visualizer.src.app.theme import DARK_THEME, generate_application_palette

    palette = generate_application_palette(DARK_THEME)
    assert palette.color(QPalette.ColorRole.Window).name() == DARK_THEME.bg_primary
    assert palette.color(QPalette.ColorRole.Text).name() == DARK_THEME.text_primary


def test_theme_mode_normalization():
    from visualizer.src.app.theme import ThemeMode, normalize_theme_mode

    assert normalize_theme_mode("dark") == ThemeMode.DARK
    assert normalize_theme_mode("invalid") == ThemeMode.SYSTEM


def test_theme_manager_applies_explicit_dark_mode():
    from visualizer.src.app.theme import (
        DARK_THEME,
        LIGHT_THEME,
        ThemeManager,
        ThemeMode,
        generate_application_palette,
    )

    class Settings:
        def __init__(self):
            self.values = {}

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):  # noqa: N802 - QSettings-compatible test double
            self.values[key] = value

    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    settings = Settings()
    manager = ThemeManager(settings=settings)
    theme = manager.install(app, mode=ThemeMode.DARK, persist=True)

    assert theme == DARK_THEME
    assert manager.mode == ThemeMode.DARK
    assert settings.values["ui/theme_mode"] == "dark"


def test_theme_manager_system_mode_uses_platform_palette_after_override():
    from visualizer.src.app.theme import (
        DARK_THEME,
        LIGHT_THEME,
        ThemeManager,
        ThemeMode,
        generate_application_palette,
    )

    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    manager = ThemeManager()

    assert manager.install(app, mode=ThemeMode.DARK) == DARK_THEME
    assert manager.set_mode(ThemeMode.SYSTEM, persist=False) == LIGHT_THEME


def test_theme_manager_repeated_install_does_not_reapply_unchanged_theme():
    from visualizer.src.app.theme import (
        LIGHT_THEME,
        ThemeManager,
        ThemeMode,
        generate_application_palette,
    )

    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    manager = ThemeManager()

    manager.install(app, mode=ThemeMode.LIGHT)
    manager.install(app, mode=ThemeMode.LIGHT)

    assert app.palette_apply_count == 1
    assert app.stylesheet_apply_count == 1


def test_theme_manager_ignores_its_own_deferred_palette_change():
    from PySide6.QtCore import QEvent

    from visualizer.src.app.theme import (
        LIGHT_THEME,
        ThemeManager,
        ThemeMode,
        generate_application_palette,
    )

    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    manager = ThemeManager()
    manager.install(app, mode=ThemeMode.SYSTEM)

    manager.eventFilter(app, QEvent(QEvent.Type.ApplicationPaletteChange))

    assert app.palette_apply_count == 1
    assert app.stylesheet_apply_count == 1


def test_theme_manager_reapplies_one_external_palette_change_once():
    from PySide6.QtCore import QEvent

    from visualizer.src.app.theme import (
        DARK_THEME,
        LIGHT_THEME,
        ThemeManager,
        ThemeMode,
        generate_application_palette,
    )

    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    manager = ThemeManager()
    manager.install(app, mode=ThemeMode.SYSTEM)

    app.setPalette(generate_application_palette(DARK_THEME))
    event = QEvent(QEvent.Type.ApplicationPaletteChange)
    manager.eventFilter(app, event)

    assert manager.current_theme == DARK_THEME
    assert app.palette_apply_count == 3
    assert app.stylesheet_apply_count == 2

    manager.eventFilter(app, QEvent(QEvent.Type.ApplicationPaletteChange))

    assert app.palette_apply_count == 3
    assert app.stylesheet_apply_count == 2


def test_spacing_and_font_constants():
    from visualizer.src.app.theme import (
        FONT_LG,
        FONT_MD,
        FONT_SM,
        SPACING_LG,
        SPACING_MD,
        SPACING_SM,
        SPACING_XL,
        SPACING_XS,
    )

    assert SPACING_XS < SPACING_SM < SPACING_MD < SPACING_LG < SPACING_XL
    assert FONT_SM < FONT_MD < FONT_LG
