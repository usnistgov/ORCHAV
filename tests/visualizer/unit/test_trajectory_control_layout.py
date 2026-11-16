"""Trajectory display controls remain usable in a narrow scaled sidebar."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtWidgets import QLabel, QWidget

from visualizer.src.app.theme import ThemeManager
from visualizer.src.panels.trajectory_preview_panel import TrajectoryPreviewPanel
from visualizer.src.services.trajectory_load_service import TrajectoryLoadCoordinator


@pytest.mark.parametrize("width", [380, 1100])
def test_trajectory_controls_wrap_and_keep_color_label_with_selection(
    qtbot, qapp, monkeypatch, tmp_path, width
):
    original_style, original_palette = qapp.styleSheet(), qapp.palette()
    manager = ThemeManager(QSettings(str(tmp_path / "trajectory.ini"), QSettings.IniFormat))
    manager.install(qapp, mode="light")
    monkeypatch.setattr(
        "visualizer.src.panels.trajectory_preview_panel.get_theme_manager", lambda: manager
    )
    coordinator = TrajectoryLoadCoordinator()
    panel = TrajectoryPreviewPanel(
        SimpleNamespace(trajectory_load_coordinator=coordinator, frame_source=None)
    )
    monkeypatch.setattr(panel, "_create_trajectory_canvas", QWidget)
    group = panel.create_panel()
    qtbot.addWidget(group)
    group.resize(width, 400)
    group.show()
    controls = [
        panel.widgets[key]
        for key in (
            "load_btn",
            "show_tx",
            "show_rx",
            "show_targets",
            "show_labels",
            "color_by_combo",
            "status_label",
        )
    ]
    combo = panel.widgets["color_by_combo"]
    color_label = next(label for label in group.findChildren(QLabel) if label.text() == "Color by:")
    combo.setCurrentText("Strongest Single-Path Loss")
    panel.widgets["status_label"].setText("Loaded 10000 frames")
    try:
        for mode, scale in [("light", 1.0), ("dark", 1.3), ("light", 1.0)]:
            manager.set_mode(mode, persist=False)
            manager.set_interface_scale(scale, persist=False)
            qapp.processEvents()
            assert group.width() == width
            rectangles = [
                QRect(control.mapTo(group, QPoint()), control.size()) for control in controls
            ]
            for control, rect in zip(controls, rectangles):
                assert group.rect().contains(rect)
                assert control.height() >= control.minimumSizeHint().height()
            for index, rect in enumerate(rectangles):
                assert not any(rect.intersects(other) for other in rectangles[index + 1 :])
            label_center = color_label.mapTo(group, color_label.rect().center()).y()
            combo_center = combo.mapTo(group, combo.rect().center()).y()
            assert abs(label_center - combo_center) <= 1
            assert color_label.parentWidget() is combo.parentWidget()
            assert combo.currentText() == "Strongest Single-Path Loss"
            first_center = controls[0].mapTo(group, controls[0].rect().center()).y()
            if width == 380:
                assert combo_center > first_center
            else:
                assert abs(combo_center - first_center) <= 1
    finally:
        panel.cleanup()
        coordinator.shutdown()
        group.close()
        qapp.removeEventFilter(manager)
        qapp.setStyleSheet(original_style)
        qapp.setPalette(original_palette)
