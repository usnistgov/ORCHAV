"""Path presentation controls stay usable in floating and narrow docked panes."""

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QAbstractButton, QComboBox, QDoubleSpinBox, QWidget

from tests.visualizer.unit.test_experimental_integration_review import _settings
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.service import _analyze
from visualizer.src.experiments.window import ExperimentWindow
from visualizer.src.panels.collapsible_section import CollapsibleSection


@pytest.fixture
def window(qtbot):
    window = ExperimentWindow()
    qtbot.addWidget(window)
    data, histogram, end = _analyze(
        _catalog([10, 20], losses=[60, 70], angles=[20, 40]), _settings()
    )
    window.set_analysis(data, histogram, reset_gate=True, clock_end=end)
    window.presentation_options.expand()
    window.show()
    return window


def test_presentation_groups_share_a_row_then_wrap_without_clipping(window, qtbot):
    window.resize(1040, 800)
    qtbot.wait(20)
    groups = list(window.presentation_groups.values())
    assert len({group.geometry().center().y() for group in groups}) == 1
    wide_height = window.presentation_options.height()

    window.resize(480, 1000)
    qtbot.wait(20)
    assert window.width() == 480
    assert len({group.geometry().center().y() for group in groups}) > 1
    assert window.presentation_options.height() > wide_height
    pane = window.widget()
    for group in groups:
        assert group.isVisible()
        for control in group.findChildren(QWidget):
            if not isinstance(control, (QAbstractButton, QComboBox, QDoubleSpinBox)):
                continue
            corner = control.mapTo(pane, QPoint(0, 0))
            assert corner.x() >= 0 and corner.y() >= 0
            assert corner.x() + control.width() <= pane.width()
            assert corner.y() + control.height() <= pane.height()


def test_presentation_controls_have_context_without_a_details_accordion(window):
    sections = window.findChildren(CollapsibleSection)
    assert len(sections) == 1 and sections[0] is window.presentation_options
    for name in (
        "pair",
        "scope",
        "weight",
        "lower_bound",
        "upper_bound",
        "play",
        "cinematic_limit",
        "interaction_mode",
        "presentation_speed",
        "interaction_dwell",
        "match_mpc_colors",
        "interaction_labels",
        "presentation_end",
        "restart",
        "trail_mode",
        "halo_opacity",
        "previous_interaction",
        "next_interaction",
        "clock",
        "angle",
        "angular_bin",
        "delay_bin",
    ):
        assert getattr(window, name).toolTip(), name
    assert "overlap" in window.interaction_labels.toolTip()
    assert "32" in window.interaction_labels.toolTip()
    assert not window.interaction_dwell.isEnabled()
    window.interaction_mode.setCurrentIndex(window.interaction_mode.findData("dwell"))
    assert window.interaction_dwell.isEnabled()


def test_provenance_and_diagnostics_follow_their_controls(window):
    window.set_frame_identity("Accepted frame 4 · source epoch 2")
    assert "Accepted frame 4" in window.pair.toolTip()
    window.set_status("No matching paths", "0 contributing paths; 0 drawn")
    assert window.status.isVisible()
    assert "0 contributing paths" in window.matching_label.toolTip()
    window.set_clock(0)
    assert "clock 0.000 ns" in window.clock.toolTip()
    window.set_interaction_note("Labels 3/8 · Plain markers")
    assert "Labels 3/8" in window.interaction_labels.toolTip()
    assert "Plain markers" in window.match_mpc_colors.toolTip()
    assert "occupied bins" in window.angle_plot.toolTip()
    window.clear_analysis()
    assert not window.matching_label.toolTip()
    assert "occupied bins" not in window.angle_plot.toolTip()
    assert "Labels 3/8" not in window.interaction_labels.toolTip()
    assert window.availability_note.isVisible()
