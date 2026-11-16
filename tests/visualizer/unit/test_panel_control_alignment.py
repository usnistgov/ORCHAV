"""Panel controls retain alignment as Qt themes and interface scale change."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel

from visualizer.src.app.theme import ThemeManager
from visualizer.src.panels.camera_panel import CameraControlPanel
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel


@pytest.fixture
def theme_manager(qapp, tmp_path):
    original_style = qapp.styleSheet()
    original_palette = qapp.palette()
    manager = ThemeManager(QSettings(str(tmp_path / "alignment.ini"), QSettings.IniFormat))
    manager.install(qapp, mode="light")
    yield manager
    qapp.removeEventFilter(manager)
    qapp.setStyleSheet(original_style)
    qapp.setPalette(original_palette)


@pytest.mark.parametrize("width", [280, 900])
def test_mpc_headings_align_with_first_line_after_theme_and_scale_changes(
    qtbot, qapp, theme_manager, width
):
    panel = MPCVisualizationPanel(SimpleNamespace())
    widget = panel._create_discrete_filters()
    qtbot.addWidget(widget)
    widget.resize(width, 200)
    widget.show()
    headings = {label.text(): label for label in widget.findChildren(QLabel)}
    for mode, scale in [("light", 1.0), ("dark", 1.3), ("light", 1.0)]:
        theme_manager.set_mode(mode, persist=False)
        theme_manager.set_interface_scale(scale, persist=False)
        qapp.processEvents()
        for title, key in [("Orders", "order_0_cb"), ("Types", "type_0_cb")]:
            heading, checkbox = headings[title], panel.widgets[key]
            label_center = heading.mapTo(widget, heading.rect().center()).y()
            check_center = checkbox.mapTo(widget, checkbox.rect().center()).y()
            assert abs(label_center - check_center) <= 1
        checks = [panel.widgets[f"order_{index}_cb"] for index in range(7)]
        first_y = checks[0].mapTo(widget, checks[0].rect().center()).y()
        last_y = checks[-1].mapTo(widget, checks[-1].rect().center()).y()
        if width == 280:
            assert last_y > first_y
        else:
            assert last_y == first_y


@pytest.mark.parametrize("width", [380, 900])
def test_camera_buttons_fit_themed_text_and_keep_row_centers(qtbot, qapp, theme_manager, width):
    panel = CameraControlPanel(SimpleNamespace())
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    widget.resize(width, 500)
    widget.show()
    quick = [
        panel.widgets[key]
        for key in (
            "view_top_btn",
            "view_side_btn",
            "view_iso_btn",
            "view_front_btn",
            "reset_camera_btn",
        )
    ]
    saved = [panel.widgets["camera_preset_save_btn"], *panel.widgets["camera_preset_buttons"]]
    for mode, scale in [("light", 1.0), ("dark", 1.3), ("light", 1.0)]:
        theme_manager.set_mode(mode, persist=False)
        theme_manager.set_interface_scale(scale, persist=False)
        qapp.processEvents()
        for row in (quick, saved):
            centers = [button.mapTo(widget, button.rect().center()).y() for button in row]
            assert max(centers) - min(centers) <= 1
            for button in row:
                assert button.width() >= button.minimumSizeHint().width()
                assert button.height() >= button.minimumSizeHint().height()
