"""Statistics controls fit narrow viewports under the application theme."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from visualizer.src.app import theme
from visualizer.src.panels.statistics_panel import StatisticsPanel


@pytest.fixture
def scaled_theme(qapp, monkeypatch, tmp_path):
    original_style = qapp.styleSheet()
    original_palette = qapp.palette()
    manager = theme.ThemeManager(QSettings(str(tmp_path / "statistics.ini"), QSettings.IniFormat))
    monkeypatch.setattr(theme, "_THEME_MANAGER", manager)
    manager.install(qapp, mode="dark")
    manager.set_interface_scale(1.3, persist=False)
    yield manager
    qapp.removeEventFilter(manager)
    qapp.setStyleSheet(original_style)
    qapp.setPalette(original_palette)


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("margin", [0, 22])
def test_statistics_scope_controls_fit_380px_at_130_percent(
    qtbot, qapp, scaled_theme, populated, margin
):
    panel = StatisticsPanel(SimpleNamespace())
    root = panel.create_graphs_panel()
    viewport = QWidget()
    layout = QVBoxLayout(viewport)
    layout.setContentsMargins(margin, 0, margin, 0)
    layout.addWidget(root)
    qtbot.addWidget(viewport)
    bar = panel._scope_bar
    if populated:
        bar.bind(
            {
                "frame_indices": [0, 1],
                "frame_times_s": [0.0, 123.456789],
                "tx_rx_pairs": [(0, 0)],
            },
            enabled=True,
        )
        bar.interval.setChecked(True)
    viewport.resize(380, 900)
    viewport.show()
    qapp.processEvents()
    try:
        assert viewport.width() == 380
        for control in (bar.pair, bar.interval, bar.start, bar.end, bar.apply, bar.reset):
            assert control.width() >= control.minimumSizeHint().width(), control
            assert control.height() >= control.minimumSizeHint().height(), control
            assert bar.rect().contains(control.geometry()), control
        assert panel._graph_tabs.currentWidget().horizontalScrollBar().maximum() == 0
    finally:
        panel.cleanup()


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_statistics_checkbox_captions_follow_actual_application_theme(
    qtbot, qapp, scaled_theme, mode
):
    scaled_theme.set_mode(mode, persist=False)
    panel = StatisticsPanel(SimpleNamespace())
    roots = (panel.create_panel(), panel.create_graphs_panel())
    try:
        for root in roots:
            qtbot.addWidget(root)
            root.resize(380, 900)
            root.show()
        qapp.processEvents()
        captions = {
            checkbox.text(): checkbox for root in roots for checkbox in root.findChildren(QCheckBox)
        }
        for text in ("Compute automatically on cache miss", "Grid"):
            checkbox = captions[text]
            foreground = QColor(scaled_theme.current_theme.text_primary)
            assert checkbox.palette().color(QPalette.WindowText) == foreground
            image = checkbox.grab().toImage()
            # Beyond the indicator, real text pixels must remain visible.
            assert any(
                image.pixelColor(x, y) == foreground
                for x in range(40, image.width())
                for y in range(image.height())
            ), text
    finally:
        panel.cleanup()
