"""Beam display controls fit the sidebar and expose relevant subordinate fields."""

from types import SimpleNamespace

from visualizer.src.panels.beam_pattern_panel import BeamPatternPanel
from visualizer.src.state import create_initial_state


def test_dynamic_range_is_conditional_and_preserves_value(qtbot):
    panel = BeamPatternPanel(SimpleNamespace(app_state=create_initial_state()))
    group = panel.create_panel()
    qtbot.addWidget(group)
    widgets = panel.widgets
    assert widgets["beam_dynamic_range_controls"].isHidden()
    widgets["beam_dynamic_range"].setValue(55)
    widgets["beam_db_scale_cb"].setChecked(True)
    assert not widgets["beam_dynamic_range_controls"].isHidden()
    assert widgets["beam_dynamic_range"].isEnabled()
    widgets["beam_db_scale_cb"].setChecked(False)
    assert widgets["beam_dynamic_range_controls"].isHidden()
    assert widgets["beam_dynamic_range"].value() == 55


def test_full_beam_controls_fit_380_pixels_with_manual_steering(qtbot, qapp):
    panel = BeamPatternPanel(SimpleNamespace(app_state=create_initial_state()))
    group = panel.create_panel()
    qtbot.addWidget(group)
    panel.widgets["standalone_strategy"].setCurrentText("Manual Steering")
    panel.widgets["beam_db_scale_cb"].setChecked(True)
    panel.widgets["beamforming_cb"].setChecked(True)
    group.resize(380, 1000)
    group.show()
    qapp.processEvents()
    assert group.minimumSizeHint().width() <= 380
    for key in ("standalone_azimuth", "standalone_elevation", "beam_dynamic_range"):
        assert panel.widgets[key].isVisibleTo(group)
