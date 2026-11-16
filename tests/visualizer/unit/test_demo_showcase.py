"""A frozen geometric tour hands off once to ordinary recorded playback."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_pulses import renderer
from tests.visualizer.unit.test_demo_service import Appearance
from tests.visualizer.unit.test_demo_service import rig as rig
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services.demo_mpc_timeline import DemoMpcTimeline


@pytest.fixture
def showcase(rig):
    viz, service, tools = rig
    native = renderer()
    for key in ("_gfx", "_scene", "_apply_clipping_to_material", "request_redraw"):
        setattr(viz.renderer, key, getattr(native, key))
    catalog = _catalog([1.0, 2.0, 3.0], losses=[20.0, 10.0, 30.0])
    packet = SimpleNamespace(
        canonical_data=replace(
            catalog.canonical_data, segment_path_id=np.arange(3, dtype=np.int32)
        ),
        path_mask=np.ones(3, dtype=bool),
        segment_mask=np.array([True, False, True]),
        mpc_visibility=SimpleNamespace(effective_paths=True),
    )
    experiments = SimpleNamespace(packet=packet, step=5)
    viz.extension_services = {"experiments": experiments}
    viz.get_available_animation_steps = lambda: [5, 11, 17]
    viz._mpc_presented_source_epoch = 1
    viz.play_direction = -1
    viz.appearance_preset_service = Appearance(viz.renderer)
    calls = []

    def toggle(direction=None):
        if direction is not None:
            viz.play_direction = direction
        viz.animation_running = not viz.animation_running
        calls.append(("play", viz.animation_running))

    def reset():
        assert not viz.animation_running
        experiments.step = min(viz.get_available_animation_steps())
        calls.append(("reset", experiments.step))

    viz.animation_controller = SimpleNamespace(toggle_animation=toggle, reset_animation=reset)
    return viz, service, tools, experiments, calls


def finish_tour(service):
    for _ in range(len(service._appearance_cycle) + 2):
        service.tick(service._next_look)


def test_showcase_holds_frame_then_retires_pulses_and_camera_before_forward_playback(showcase):
    viz, service, tools, experiments, calls = showcase
    appearance = viz.appearance_preset_service
    original = deepcopy(appearance.capture())
    viz.animation_running = True
    viz.loop_cb.setChecked(True)
    assert service.start(showcase=True)
    timeline = service._showcase
    assert not viz.animation_running
    assert service._timer.interval() == 33
    assert timeline.geometry.ids.tolist() == [0, 2]  # Hidden strongest path stays hidden.
    assert tools.autorotate is not None

    for _ in range(len(service._appearance_cycle) + 1):
        service.tick(service._next_look)
        assert not viz.animation_running and experiments.step == 5
    assert appearance.capture() == original
    assert [name for action, name in appearance.calls if action == "apply"] == [
        "Detailed",
        "Paper",
        "Plan",
        "Dark",
        "Glow",
    ]
    assert timeline.overlay.layers
    service.tick(service._next_look)
    assert not timeline.overlay.layers
    assert tools.autorotate is None
    assert viz.animation_running and viz.play_direction == 1
    assert not viz.loop_cb.isChecked()
    assert calls[-2:] == [("reset", 5), ("play", True)]
    assert service.status.startswith("Showcase · recorded playback")
    service.tick(service._next_look + 100)
    assert service.active  # The deliberately stopped orbit does not cancel playback.

    viz.animation_running = False  # Ordinary AnimationController reaches the end.
    service.tick()
    assert not service.active and service.status == "Showcase complete."
    assert not viz.animation_running  # Completion must not restart old playback.
    assert viz.loop_cb.isChecked() and viz.play_direction == -1
    assert tools.history == ["original-history"]
    assert appearance.capture() == original


@pytest.mark.parametrize("phase", ["presets", "playback"])
def test_takeover_restores_original_transport_and_removes_only_demo_overlay(showcase, phase):
    viz, service, _tools, _experiments, _calls = showcase
    viz.animation_running = True
    original_objects = tuple(viz.renderer._scene.children)
    assert service.start(showcase=True, orbit=False)
    timeline = service._showcase
    if phase == "playback":
        finish_tour(service)
    service.stop()
    assert viz.animation_running and viz.play_direction == -1
    assert not viz.loop_cb.isChecked()
    assert tuple(viz.renderer._scene.children) == original_objects
    assert not timeline.overlay.layers


@pytest.mark.parametrize("change", ["frame", "source", "renderer", "scene"])
def test_showcase_retires_stale_frozen_paths(showcase, change):
    viz, service, _tools, experiments, _calls = showcase
    assert service.start(showcase=True, orbit=False)
    timeline = service._showcase
    if change == "frame":
        experiments.step = 11
    elif change == "source":
        viz._mpc_presented_source_epoch += 1
    elif change == "renderer":
        viz.renderer = object()
    else:
        viz.scenario = object()
    service.tick()
    assert not service.active
    assert not timeline.overlay.layers
    assert not viz.animation_running


@pytest.mark.parametrize(
    "problem", ["single_frame", "no_paths", "hidden_paths", "analysis_playing"]
)
def test_unavailable_showcase_cannot_leave_a_partial_demo(showcase, problem):
    viz, service, tools, experiments, _calls = showcase
    if problem == "single_frame":
        viz.get_available_animation_steps = lambda: [5]
    elif problem == "no_paths":
        experiments.packet.canonical_data = None
    elif problem == "hidden_paths":
        experiments.packet.mpc_visibility.effective_paths = False
    else:
        experiments._clock = SimpleNamespace(isActive=lambda: True)
    viz.animation_running = True
    assert not service.start(showcase=True)
    assert not service.active and not service._timer.isActive()
    assert tools.autorotate is None
    assert viz.animation_running
    assert not viz.renderer._scene.children


def test_close_during_tour_does_not_resume_previous_playback(showcase):
    viz, service, _tools, _experiments, _calls = showcase
    viz.animation_running = True
    assert service.start(showcase=True, orbit=False)
    service.close()
    assert not viz.animation_running and not viz.renderer._scene.children


def test_playback_source_replacement_does_not_restore_old_transport_into_new_source(showcase):
    viz, service, _tools, _experiments, _calls = showcase
    assert service.start(showcase=True, orbit=False)
    finish_tour(service)
    viz._mpc_presented_source_epoch += 1
    viz.animation_running = False
    viz.play_direction = 1
    service.tick()
    assert not service.active and not viz.animation_running
    assert viz.play_direction == 1


def test_texture_locked_showcase_finishes_after_its_available_presets(showcase, monkeypatch):
    viz, service, _tools, _experiments, _calls = showcase
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    assert service.start(showcase=True, orbit=False)
    assert "Detailed" not in service._appearance_cycle
    finish_tour(service)
    assert viz.animation_running and service._showcase_phase == "playback"


def test_showcase_pulses_follow_recorded_geometry_without_mutating_it(showcase):
    viz, _service, _tools, experiments, _calls = showcase
    points = experiments.packet.canonical_data.points
    original = points.copy()
    timeline = DemoMpcTimeline(viz)
    try:
        timeline.restart(10.0, 5.0)
        timeline.sample(12.0)
        leading = timeline.overlay.layers["cinematic_head"].geometry.positions.data
        assert np.all((leading[:, 0] > 0) & (leading[:, 0] < 1))
        np.testing.assert_array_equal(leading[:, 1:], 0)
        np.testing.assert_array_equal(points, original)
        timeline.sample(15.0)
        assert not timeline.overlay.layers["cinematic_head"].geometry.colors.data[:, 3].any()
    finally:
        timeline.close()


def test_panel_exposes_sequence_and_restores_independent_options(showcase, qtbot):
    viz, _service, _tools, _experiments, _calls = showcase
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        panel.widgets["demo_showcase"].setChecked(True)
        assert not panel.widgets["demo_playback"].isEnabled()
        assert not panel.widgets["demo_look"].isEnabled()
        assert not panel.widgets["demo_appearance"].isEnabled()
        assert panel.widgets["demo_interval"].isEnabled()
        panel.start()
        assert panel.service.active and not viz.animation_running
        panel.service.stop()
        panel.widgets["demo_showcase"].setChecked(False)
        assert panel.widgets["demo_playback"].isEnabled()
        assert panel.widgets["demo_look"].isEnabled()
    finally:
        panel.cleanup()
