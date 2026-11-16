"""Scenario replacement resets every timeline control, including sparse runs."""

from types import MethodType, SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSlider, QSpinBox

from visualizer.src.app.scenario_workflow import _sync_initial_frame
from visualizer.visualizer import OrchavVisualizer


@pytest.mark.parametrize("frame_ids", [list(range(512)), [63, 127, 191, 255]])
@pytest.mark.parametrize("state_already_reset", [True, False])
def test_scenario_switch_resets_all_navigation_without_emitting_navigation(
    qtbot, frame_ids, state_already_reset
):
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 511)
    slider.setValue(511)
    frame_input = QSpinBox()
    frame_input.setRange(1, 512)
    frame_input.setValue(512)
    label = QLabel("512")
    for widget in (slider, frame_input, label):
        qtbot.addWidget(widget)
    navigation = []
    slider.valueChanged.connect(lambda value: navigation.append(("slider", value)))
    frame_input.valueChanged.connect(lambda value: navigation.append(("input", value)))
    context = []

    def update_bounds(count):
        slider.setMaximum(count - 1)
        frame_input.setMaximum(count)

    viz = SimpleNamespace(
        animation_step=frame_ids[0] if state_already_reset else 511,
        step_slider=slider,
        frame_input=frame_input,
        step_label=label,
        get_animation_step_index=frame_ids.index,
        ui_manager=SimpleNamespace(
            update_total_steps=update_bounds,
            panels={"animation": SimpleNamespace(set_recorded_frame_context=context.append)},
        ),
    )
    viz._sync_animation_navigation_widgets = MethodType(
        OrchavVisualizer._sync_animation_navigation_widgets, viz
    )
    viz.set_state = lambda **values: setattr(viz, "committed_state", values)
    _sync_initial_frame(viz, SimpleNamespace(list_frames=lambda: frame_ids))
    assert viz.animation_step == frame_ids[0]
    assert viz.committed_state == {"step": frame_ids[0]}
    assert slider.value() == 0
    assert frame_input.value() == 1
    assert label.text() == "1"
    assert slider.maximum() == len(frame_ids) - 1
    assert frame_input.maximum() == len(frame_ids)
    assert context == [frame_ids]
    assert navigation == []


def test_navigation_refresh_preserves_preexisting_signal_block(qtbot):
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(100)
    qtbot.addWidget(slider)
    slider.blockSignals(True)
    viz = SimpleNamespace(
        animation_step=0,
        step_slider=slider,
        get_animation_step_index=lambda _source_id: 0,
        set_state=lambda **_values: None,
    )
    viz._sync_animation_navigation_widgets = MethodType(
        OrchavVisualizer._sync_animation_navigation_widgets, viz
    )
    _sync_initial_frame(viz, SimpleNamespace(list_frames=lambda: [0, 1]))
    assert slider.value() == 0
    assert slider.signalsBlocked()
