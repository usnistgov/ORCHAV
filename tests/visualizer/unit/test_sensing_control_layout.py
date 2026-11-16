"""Sensing controls retain preferences while exposing relevant compact options."""

from PySide6.QtCore import QSignalBlocker

from tests.visualizer.unit.test_sensing_scene_controls import scene_panel as scene_panel
from visualizer.src.sensing.sensing_window import SensingWindow


def test_scene_dependencies_hide_outside_relevant_overlay_mode(scene_panel):
    panel, _, _ = scene_panel(enabled=True)
    widgets = panel.widgets
    dependent = (
        widgets["sensing_scene_full_history"],
        widgets["sensing_scene_show_velocity_arrows"],
        widgets["sensing_scene_show_track_labels"],
    )
    assert all(widget.isHidden() for widget in dependent)
    mode = widgets["sensing_scene_mode"]
    mode.setCurrentIndex(mode.findData("tracks"))
    assert all(widget.isVisible() and widget.isEnabled() for widget in dependent)
    dependent[1].setChecked(True)
    mode.setCurrentIndex(mode.findData("detections"))
    assert dependent[1].isHidden() and dependent[1].isChecked()
    mode.setCurrentIndex(mode.findData("tracks"))
    assert dependent[1].isVisible() and dependent[1].isChecked()
    dependent[0].setChecked(True)
    assert widgets["sensing_scene_history_frames"].isHidden()


def test_display_spans_align_in_narrow_dashboard(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(380, 950)
    window.show()
    window._set_view_mode("micro-doppler")
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    assert window._advanced_widget.isVisible()
    qtbot.waitUntil(lambda: window._compact)
    grid = window._options_grid
    rd_row = grid.getItemPosition(grid.indexOf(window._rd_range_label))[0]
    motion_row = grid.getItemPosition(grid.indexOf(window._motion_range_label))[0]
    assert rd_row != motion_row
    assert grid.getItemPosition(grid.indexOf(window._rd_dynamic_range_box))[:2] == (rd_row, 1)
    assert grid.getItemPosition(grid.indexOf(window._micro_dynamic_range_box))[:2] == (
        motion_row,
        1,
    )
    assert window._variant_box.isHidden()  # No alternate spectrum has been loaded.
    qtbot.waitUntil(lambda: window.evidence_scroll.horizontalScrollBar().maximum() == 0)
