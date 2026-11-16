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

    def allWidgets(self):
        return []

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


@pytest.mark.parametrize("scale", [1.0, 1.3])
def test_theme_filter_ignores_non_event_wrapper(qtbot, scale, monkeypatch):
    from PySide6.QtGui import QStandardItem

    from visualizer.src.app.theme import ThemeManager

    manager = ThemeManager()
    manager._scale = scale
    widget = _REAL_QLABEL("Selection")
    qtbot.addWidget(widget)
    monkeypatch.setattr(manager, "_scale_widget", lambda _: pytest.fail("Not a Qt event"))
    # Reported by PySide on Windows while switching between native plot widgets.
    assert manager.eventFilter(widget, QStandardItem("unexpected wrapper")) is False


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


def test_interface_scale_round_trip_and_late_controls(qapp, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QSpinBox, QWidget

    from visualizer.src.app.theme import ThemeManager

    settings = QSettings(str(tmp_path / "theme.ini"), QSettings.IniFormat)
    manager = ThemeManager(settings)
    host = QWidget()
    spin = QSpinBox(host)
    spin.setFixedHeight(24)
    spin.setStyleSheet("font-size: 10px;")
    manager.install(qapp, mode="light")
    try:
        manager.set_interface_scale(1.3)
        assert spin.height() == 31
        assert "13px" in spin.styleSheet()
        manager.set_mode("dark", persist=False)
        assert spin.height() == 31
        assert "13px" in spin.styleSheet()
        later = QSpinBox(host)
        later.setStyleSheet("font-size: 10px;")
        later.ensurePolished()
        assert "13px" in later.styleSheet()
        manager.set_interface_scale(1.0)
        assert spin.height() == 24
        assert "10px" in spin.styleSheet()
        assert "10px" in later.styleSheet()
    finally:
        qapp.removeEventFilter(manager)
        host.close()


def test_theme_change_survives_widget_deletion_during_scaling(qapp, monkeypatch):
    from PySide6.QtWidgets import QLabel, QWidget
    from shiboken6 import delete, isValid

    from visualizer.src.app.theme import (
        LIGHT_THEME,
        ThemeManager,
        generate_application_palette,
    )

    victim = QWidget()

    class DeletesNextWidget(QLabel):
        def property(self, name):
            if name == "_scaleAppliedStyle" and isValid(victim):
                delete(victim)
            return super().property(name)

    first = DeletesNextWidget()
    last = QLabel()
    first.setStyleSheet("font-size: 10px;")
    last.setStyleSheet("font-size: 10px;")
    app = _ApplicationDouble(generate_application_palette(LIGHT_THEME))
    manager = ThemeManager()
    manager.install(app, mode="light")
    monkeypatch.setattr(app, "allWidgets", lambda: [first, victim, last])
    try:
        manager.set_interface_scale(1.3, persist=False)
        assert not isValid(victim)
        assert "13px" in first.styleSheet()
        assert "13px" in last.styleSheet()
        assert manager.current_theme == LIGHT_THEME
    finally:
        first.close()
        last.close()


def test_scaled_workflow_controls_match_new_controls(qapp, qtbot, tmp_path):
    """Changing scale must not enlarge QSS bounds a second time."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox, QVBoxLayout, QWidget

    from visualizer.src.app.theme import ThemeManager

    manager = ThemeManager(QSettings(str(tmp_path / "workflow.ini"), QSettings.IniFormat))
    previous_style = qapp.styleSheet()
    previous_palette = qapp.palette()

    def controls():
        host = QWidget()
        qtbot.addWidget(host)
        host.setProperty("workflowWorkspace", True)
        layout = QVBoxLayout(host)
        widgets = [QSpinBox(), QComboBox(), QLineEdit()]
        widgets[0].setFixedWidth(72)
        widgets[1].addItem("Choice")
        for widget in widgets:
            layout.addWidget(widget)
        host.show()
        qapp.processEvents()
        return host, widgets

    manager.install(qapp, mode="light")
    host, existing = controls()
    try:
        for scale, mode in ((1.3, "light"), (1.3, "dark"), (1.15, "dark"), (1.0, "light")):
            manager.set_interface_scale(scale, persist=False)
            manager.set_mode(mode, persist=False)
            new_host, fresh = controls()
            for old, new in zip(existing, fresh, strict=True):
                assert old.minimumHeight() == new.minimumHeight()
                assert old.sizeHint().height() == new.sizeHint().height()
                assert old.font().pixelSize() == new.font().pixelSize()
            assert existing[0].minimumWidth() == round(72 * scale)
            assert existing[0].maximumWidth() == round(72 * scale)
            new_host.close()
    finally:
        qapp.removeEventFilter(manager)
        host.close()
        qapp.setStyleSheet(previous_style)
        qapp.setPalette(previous_palette)


def test_local_stylesheet_bounds_scale_once_after_creation(qapp, qtbot, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QSpinBox, QWidget

    from visualizer.src.app.theme import ThemeManager

    manager = ThemeManager(QSettings(str(tmp_path / "local.ini"), QSettings.IniFormat))
    previous_style = qapp.styleSheet()
    previous_palette = qapp.palette()
    host = QWidget()
    qtbot.addWidget(host)
    spin = QSpinBox(host)
    spin.setFixedWidth(80)
    manager.install(qapp, mode="light")
    try:
        manager.set_interface_scale(1.3, persist=False)
        spin.setStyleSheet("min-height: 28px; max-height: 45px; font-size: 12px;")
        spin.ensurePolished()
        qapp.processEvents()
        bounds = spin.minimumHeight(), spin.maximumHeight()
        manager.set_mode("dark", persist=False)
        assert (spin.minimumHeight(), spin.maximumHeight()) == bounds
        assert spin.minimumWidth() == spin.maximumWidth() == 104
        assert "16px" in spin.styleSheet()
        manager.set_interface_scale(1.0, persist=False)
        # QSS bounds include the theme's 3px padding and 1px border on each side.
        assert spin.minimumHeight() == 36
        assert spin.maximumHeight() == 53
        assert spin.minimumWidth() == spin.maximumWidth() == 80
        assert "12px" in spin.styleSheet()
    finally:
        qapp.removeEventFilter(manager)
        host.close()
        qapp.setStyleSheet(previous_style)
        qapp.setPalette(previous_palette)
