"""Camera timing and real pygfx controller regressions (no graphics device needed)."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx")

from visualizer.src.controllers.camera_controller import CameraController
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.advanced_camera_panel import AdvancedCameraPanel
from visualizer.src.panels.camera_panel import CameraControlPanel
from visualizer.src.panels.object_panel import ObjectManagementPanel
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
from visualizer.src.services.camera_motion import interpolate, orbit
from visualizer.src.types.camera_state import CameraState

INITIAL = CameraState((0, -10, 3), (0, 0, 0), (0, 0, 1), 60)


@pytest.fixture
def rig(qapp, monkeypatch):
    # Tests advance presentation time explicitly. Starting at zero avoids
    # cancellation error at exact transition deadlines on long-running hosts.
    monkeypatch.setattr("visualizer.src.services.camera_presentation.perf_counter", lambda: 0.0)
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._initialized = True
    renderer._camera = pygfx.PerspectiveCamera(60, 1.5)
    renderer._camera.local.position = INITIAL.eye
    renderer._controller = pygfx.OrbitController(renderer._camera, target=(0, 0, 0))
    renderer._renderer = None
    renderer._camera_state = INITIAL
    renderer._follow_target_lookat = None
    renderer._last_camera_look_distance = 10
    renderer._focus_canvas = lambda: None
    renderer.request_redraw = lambda: None
    position = np.zeros(3)
    viz = SimpleNamespace(
        renderer=renderer,
        app_state=SimpleNamespace(camera_mode="follow", pov_axis="forward"),
        animation_running=False,
        current_scenario_path="recording-a",
        vis_initialized=True,
        vis=renderer,
        target_focus_dropdown=SimpleNamespace(currentData=lambda: {"type": "rx", "index": 0}),
    )
    viz.set_state = lambda **kw: [setattr(viz.app_state, key, value) for key, value in kw.items()]
    query = SimpleNamespace(
        resolve_rx_position=lambda index: position.copy() if index == 0 else None,
        get_focus_position=lambda: position.copy(),
        get_entity_position_orientation_and_info=lambda: (position.copy(), [0, 0, 0], None),
        get_focus_motion_direction=lambda: np.array([0, 1, 0]),
    )
    owner = CameraController(viz, query)
    owner.restore_pov_entity_visibility = lambda **kw: None
    viz.camera_controller = owner
    ui = UIController.__new__(UIController)
    ui.visualizer = viz
    ui._camera_debug_enabled = False
    viz.ui_controller = ui
    renderer.visualizer = viz
    renderer.set_camera_state(INITIAL)
    tools = owner.presentation()
    yield tools, renderer, position
    tools.close()


@pytest.mark.parametrize("gesture", ["zoom", "rotate", "pan"])
def test_follow_retains_translation_during_real_damped_input(rig, monkeypatch, gesture):
    tools, renderer, position = rig
    controller = renderer._controller
    clock = [10.0]
    monkeypatch.setattr("pygfx.controllers._base.perf_counter", lambda: clock[0])
    control_camera = pygfx.PerspectiveCamera(60, 1.5)
    control_camera.set_state(renderer._camera.get_state())
    control = pygfx.OrbitController(control_camera, target=(0, 0, 0))
    tools.follow(position)
    for native in (controller, control):
        delta = 0.4 if gesture == "zoom" else (0.2, 0.1) if gesture == "rotate" else (2.0, 1.0)
        getattr(native, gesture)(delta, (0, 0, 1200, 800), animate=True)
    for _ in range(8):
        position[0] += 0.015  # Also prove that sub-10 cm motion is retained.
        tools.owner.update_follow_camera_focus()
        clock[0] += 0.02
        controller.tick()
        control.tick()
        np.testing.assert_allclose(
            renderer._camera.local.position, control_camera.local.position + position, atol=1e-3
        )
    assert not np.allclose(renderer.get_camera_state().eye, INITIAL.eye)


def test_follow_preserves_user_pan_when_pausing_and_scrubbing(rig):
    tools, renderer, position = rig
    tools.follow(position)
    renderer._controller.pan((2, 1), (0, 0, 1200, 800), animate=False)
    before = renderer.get_camera_state()
    position[:] = [2, 0, 0]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(renderer.get_camera_state().eye, np.asarray(before.eye) + position)
    position[:] = [0, 0, 0]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(renderer.get_camera_state().eye, before.eye, atol=1e-6)


def test_transition_updates_moving_actor_destination_and_can_be_interrupted(rig):
    tools, renderer, position = rig
    tools.prepare()
    tools.follow(position + [5, 0, 0])
    began = tools.transition[2]
    tools.tick(began + tools.duration / 2)
    tools.follow(position + [6, 0, 0])
    tools.tick(began + tools.duration)
    np.testing.assert_allclose(renderer.get_camera_state().lookat, [6, 0, 0], atol=1e-12)
    tools.prepare()
    tools.follow(position + [10, 0, 0])
    tools.tick(tools.transition[2] + 0.2)
    visible = renderer.get_camera_state()
    tools.input_event(SimpleNamespace(type="wheel"))
    assert tools.transition is None
    assert tools.viz.app_state.camera_mode == "follow"
    np.testing.assert_allclose(renderer.get_camera_state().eye, visible.eye)


def test_focus_does_not_move_native_target_before_transition_starts(rig):
    tools, renderer, _ = rig
    before = renderer.get_camera_state()
    tools.prepare()
    renderer.focus_camera([10, 10, 0])
    assert renderer.get_camera_state() == before
    tools.tick(tools.transition[2] + tools.duration)
    np.testing.assert_allclose(renderer.get_camera_state().lookat, [10, 10, 0], atol=1e-12)


@pytest.mark.parametrize("mode", ["overview", "follow", "pov"])
def test_object_table_focus_uses_interruptible_transition_from_visible_pose(rig, mode):
    tools, renderer, _ = rig
    tools.viz.set_state(camera_mode=mode)
    panel = ObjectManagementPanel(tools.viz)
    before = renderer.get_camera_state()
    destination = [10, 10, 0]
    panel._focus_camera_on_entry({"current_center": destination, "original_center": [-10, 0, 0]})
    assert renderer.get_camera_state() == before
    assert tools.transition is not None and tools.timer.isActive()
    assert tools.viz.app_state.camera_mode == "overview"
    tools.tick(tools.transition[2] + tools.duration / 2)
    midway = renderer.get_camera_state()
    assert not np.allclose(midway.lookat, before.lookat)
    assert not np.allclose(midway.lookat, destination)
    np.testing.assert_allclose(midway.eye, before.eye)
    tools.input_event(SimpleNamespace(type="wheel"))
    assert tools.transition is None
    assert renderer.get_camera_state() == midway

    panel._focus_camera_on_entry({"position": destination})
    assert tools.transition[0] == midway
    tools.tick(tools.transition[2] + tools.duration)
    final = renderer.get_camera_state()
    np.testing.assert_allclose(final.lookat, destination, atol=1e-6)
    np.testing.assert_allclose(final.eye, before.eye)
    assert final.fov_deg == before.fov_deg
    assert tools.history[-1] == midway


def test_object_focus_respects_disabled_smoothing_and_export_lock(rig):
    tools, renderer, _ = rig
    panel = ObjectManagementPanel(tools.viz)
    tools.smooth = False
    panel._focus_camera_on_entry({"original_center": [2, 3, 0]})
    assert tools.transition is None
    np.testing.assert_allclose(renderer.get_camera_state().lookat, [2, 3, 0])
    before = renderer.get_camera_state()
    tools.exporting = True
    tools.viz.set_state(camera_mode="follow")
    panel._focus_camera_on_entry({"position": [10, 10, 0]})
    assert renderer.get_camera_state() == before
    assert tools.viz.app_state.camera_mode == "follow"
    tools.exporting = False


def test_object_focus_keeps_immediate_fallback_for_other_renderers(rig):
    tools, renderer, _ = rig
    tools.owner.presentation = lambda: None
    before = renderer.get_camera_state()
    ObjectManagementPanel(tools.viz)._focus_camera_on_entry({"position": [4, 5, 0]})
    assert tools.transition is None
    np.testing.assert_allclose(renderer.get_camera_state().lookat, [4, 5, 0])
    np.testing.assert_allclose(renderer.get_camera_state().eye, before.eye)


def test_initial_scene_framing_applies_without_a_startup_transition(rig):
    tools, renderer, _ = rig
    tools.viz.current_scenario_path = None
    tools.prepare()
    renderer.set_camera_state(orbit(INITIAL, [0, 0, 0], 90))
    assert tools.transition is None and not tools.history
    np.testing.assert_allclose(renderer.get_camera_state().eye, [10, 0, 3], atol=1e-6)


def test_follow_chase_rotates_with_yaw_and_keeps_user_distance(rig):
    tools, renderer, position = rig
    tools.follow_heading = "chase"
    tools.follow(position)
    first = renderer.get_camera_state()
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [np.pi / 2, 0, 0],
        None,
    )
    position[:] = [1, 2, 0]
    tools.follow(position)
    expected = orbit(first, [0, 0, 0], 90)
    np.testing.assert_allclose(
        renderer.get_camera_state().eye, np.asarray(expected.eye) + position, atol=1e-6
    )


def test_chase_tracks_in_place_actor_rotation(rig):
    tools, renderer, position = rig
    tools.follow_heading = "chase"
    tools.owner.update_follow_camera_focus()
    first = renderer.get_camera_state()
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [np.pi / 2, 0, 0],
        None,
    )
    tools.owner.update_follow_camera_focus()
    expected = orbit(first, [0, 0, 0], 90)
    np.testing.assert_allclose(renderer.get_camera_state().eye, expected.eye, atol=1e-6)


def test_behind_actor_persists_across_selection_and_mode_changes(rig, qtbot):
    tools, renderer, position = rig
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    panel.run(panel.behind)
    assert tools.follow_heading == "chase" and tools.transition is not None
    tools.tick(tools.transition[2] + tools.duration)
    first = renderer.get_camera_state()
    assert first.eye[0] < 0 and first.eye[2] > 0
    assert first.eye[1] == pytest.approx(0, abs=1e-6)
    distance = np.linalg.norm(np.subtract(first.eye, first.lookat))

    # A different actor at the SAME position still needs its own rear heading.
    # No new frame or playback tick is needed to initiate this change.
    tools.viz.target_focus_dropdown.currentData = lambda: {"type": "rx", "index": 1}
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [np.pi / 2, 0, 0],
        None,
    )
    tools.viz.ui_controller.handle_target_focus_changed("RX2")
    assert tools.transition is not None
    tools.tick(tools.transition[2] + tools.duration)
    second = renderer.get_camera_state()
    assert second.eye[0] == pytest.approx(0, abs=1e-6)
    assert second.eye[1] < 0 and second.eye[2] > 0
    assert np.linalg.norm(np.subtract(second.eye, second.lookat)) == pytest.approx(distance)

    tools.viz.ui_controller.handle_camera_mode_changed("overview", True)
    tools.viz.ui_controller.handle_camera_mode_changed("follow", True)
    tools.tick(tools.transition[2] + tools.duration)
    np.testing.assert_allclose(renderer.get_camera_state().eye, second.eye, atol=1e-6)


def test_follow_setting_reframes_immediately_while_paused(rig, qtbot):
    tools, renderer, _ = rig
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    focused = []
    renderer._focus_canvas = lambda: focused.append(True)
    panel.heading.setCurrentIndex(panel.heading.findData("chase"))
    state = renderer.get_camera_state()
    assert state.eye[0] < 0 and state.eye[1] == pytest.approx(0, abs=1e-6)
    assert not tools.transition and not tools.timer.isActive()
    assert not focused  # Editing settings must not steal the keyboard from the panel.
    assert "applied" in panel.status.text()

    panel.heading.setCurrentIndex(panel.heading.findData("world"))
    np.testing.assert_allclose(renderer.get_camera_state().eye, state.eye, atol=1e-6)
    assert tools.follow_heading == "world"


def test_pov_settings_apply_during_transition_without_advancing_frames(rig, qtbot):
    tools, renderer, position = rig
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    tools.viz.ui_controller.handle_camera_mode_changed("pov", True)
    assert tools.transition is not None
    panel.sync_from_state(tools.viz.app_state)
    focused = []
    renderer._focus_canvas = lambda: focused.append(True)
    panel.true_pov.setChecked(False)
    panel.offset[2].setValue(5)
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [0, 0, 5])
    assert tools.transition is None and not tools.timer.isActive()
    assert tools.owner.set_field_of_view(90)
    assert renderer.get_camera_state().fov_deg == 90
    panel.offset[0].setValue(2)
    panel.offset[1].setValue(3)
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [2, 3, 5])
    assert not focused


@pytest.mark.parametrize("mode", ["overview", "follow", "pov"])
def test_framing_disables_inactive_mode_controls(rig, qtbot, mode):
    tools, _, _ = rig
    tools.viz.app_state.camera_mode = mode
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    assert panel.heading.isEnabled() == (mode == "follow")
    assert all(w.isEnabled() == (mode == "pov") for w in [panel.true_pov, panel.horizon])
    assert all(not w.isEnabled() for w in panel.offset)
    assert mode.casefold() in panel.framing_status.text().casefold()


def test_missing_actor_stops_sequence_without_changing_data(rig):
    tools, _, _ = rig
    tools.save_view("Actor", relative=True)
    tools.shots = [{"view": "Actor", "hold": 2, "move": 0}]
    tools.start_sequence()
    tools.owner.scene_query.resolve_rx_position = lambda _: None
    tools.tick(tools.sequence[0] + 0.1)
    assert tools.sequence is None and not tools.timer.isActive()
    assert "unavailable" in tools.message


def test_paused_autorotate_and_actor_relative_orbit(rig):
    tools, renderer, position = rig
    tools.start_orbit(None, 90, actor=True)
    began = tools.autorotate[3]
    position[:] = [5, 3, 0]
    tools.tick(began + 1)
    np.testing.assert_allclose(renderer.get_camera_state().eye, [15, 3, 3], atol=1e-5)
    assert tools.viz.animation_running is False
    tools.viz.current_scenario_path = "recording-b"
    tools.tick(began + 2)
    assert tools.autorotate is None


def test_opposite_views_do_not_cross_zero_camera_direction():
    end = CameraState((0, 10, 3), (0, 0, 0), (0, 0, 1), 40)
    for t in np.linspace(0, 1, 31):
        state = interpolate(INITIAL, end, t)
        assert np.isfinite(np.asarray(state.eye)).all()
        assert np.linalg.norm(np.subtract(state.eye, state.lookat)) > 10
        assert np.linalg.norm(state.up) == pytest.approx(1)
    assert interpolate(INITIAL, end, 0) == INITIAL
    assert interpolate(INITIAL, end, 1) == end


def test_named_views_sequence_roundtrip_and_relative_resolution(rig):
    tools, renderer, position = rig
    tools.save_view("Street", relative=True)
    renderer.set_camera_state(orbit(INITIAL, [0, 0, 0], 90))
    tools.save_view("Roof")
    tools.shots = [{"view": "Street", "hold": 2, "move": 4}, {"view": "Roof", "hold": 3, "move": 1}]
    payload = tools.payload()
    tools.views = {}
    tools.shots = []
    tools.restore(payload)
    assert tools.sequence_duration() == 6
    position[:] = [2, 4, 0]
    np.testing.assert_allclose(tools.sequence_pose(0).eye, [2, -6, 3])
    assert tools.sequence_pose(6) == tools.view_pose("Roof")
    assert tools.sequence_pose(2.5) == tools.sequence_pose(2.5)
    malformed = deepcopy(payload)
    malformed["shots"][0]["hold"] = float("nan")
    with pytest.raises(ValueError):
        tools.restore(malformed)
    assert tools.payload() == payload


def test_export_restores_camera_and_timers_after_interruption(rig):
    tools, renderer, _ = rig
    tools.start_orbit([0, 0, 0], 10)
    before = renderer.get_camera_state()
    snapshot = tools.suspend_for_export()
    assert not tools.timer.isActive()
    assert not renderer._controller.enabled
    renderer.set_camera_state(orbit(INITIAL, [0, 0, 0], 45))
    assert renderer.get_camera_state() == before  # frame pipeline cannot override export
    tools.apply(orbit(INITIAL, [0, 0, 0], 90))
    tools.resume_after_export(snapshot)
    np.testing.assert_allclose(renderer.get_camera_state().eye, before.eye)
    assert tools.timer.isActive() and renderer._controller.enabled


def test_pov_local_offsets_lens_and_horizon(rig):
    tools, _, _ = rig
    tools.true_pov = False
    tools.pov_offset = [2, 0, 1]
    tools.pov_fov = 75
    state = tools.pov_pose([5, 6, 7], [np.pi / 2, 0, 0], "forward")
    np.testing.assert_allclose(state.eye, [5, 8, 8])
    np.testing.assert_allclose(np.subtract(state.lookat, state.eye), [0, 10, 0], atol=1e-8)
    assert state.fov_deg == 75 and state.up == (0, 0, 1)


def test_motion_follow_turns_with_path_and_holds_heading_when_stopped(rig, qtbot):
    tools, renderer, position = rig
    direction = np.array([0.0, 1.0, 0.0])
    tools.owner.scene_query.get_focus_motion_direction = lambda: direction.copy()
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        None,
        {"type": "rx", "index": 0},
    )
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    panel.heading.setCurrentIndex(panel.heading.findData("motion"))
    for spin, value in zip(panel.follow_offset, [12, 4, 6]):
        spin.setValue(value)
    np.testing.assert_allclose(renderer.get_camera_state().eye, [-4, -12, 6], atol=1e-5)
    direction[:] = [1, 0, 0]
    position[:] = [3, 2, 0]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [-12, 4, 6], atol=1e-5)
    direction[:] = 0
    before = renderer.get_camera_state()
    for _ in range(3):
        tools.owner.update_follow_camera_focus()
    assert renderer.get_camera_state() == before
    panel.run(panel.reset_behind)
    assert tools.follow_heading == "motion"
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [-12, 0, 0], atol=1e-5)
    payload = tools.payload()
    tools.restore(payload)
    assert tools.payload() == payload


def test_motion_follow_preserves_mouse_zoom(rig):
    tools, renderer, position = rig
    tools.follow_heading = "motion"
    tools.follow(position)
    renderer._controller.zoom(0.5, (0, 0, 1200, 800), animate=False)
    before = renderer.get_camera_state()
    position[:] = [0, 2, 0]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(
        renderer.get_camera_state().eye, np.asarray(before.eye) + position, atol=1e-5
    )


@pytest.mark.parametrize("direction", [[0, 2, 1], [1, 0, -2], [0, 0, 1]])
def test_motion_pov_uses_three_dimensional_travel_and_holds_at_stops(rig, direction):
    tools, renderer, position = rig
    tools.viz.app_state.camera_mode = "pov"
    tools.viz.app_state.pov_axis = "motion"
    velocity = np.asarray(direction, dtype=float)
    tools.owner.scene_query.get_focus_motion_direction = lambda: velocity.copy()
    assert tools.owner.set_pov_camera()
    state = renderer.get_camera_state()
    expected = velocity / np.linalg.norm(velocity)
    actual = np.subtract(state.lookat, state.eye)
    np.testing.assert_allclose(actual / np.linalg.norm(actual), expected, atol=1e-6)
    velocity[:] = 0
    position[:] = [1, 2, 3]
    assert tools.owner.set_pov_camera()
    stopped = renderer.get_camera_state()
    actual = np.subtract(stopped.lookat, stopped.eye)
    np.testing.assert_allclose(actual / np.linalg.norm(actual), expected, atol=1e-6)
    assert np.isfinite(stopped.up).all()
    # Explicit world axes still override motion when selected.
    tools.viz.app_state.pov_axis = "-x"
    assert tools.owner.set_pov_camera()
    state = renderer.get_camera_state()
    np.testing.assert_allclose(np.subtract(state.lookat, state.eye), [-10, 0, 0], atol=1e-6)


def test_motion_pov_offsets_use_travel_frame_and_switch_actors(rig):
    tools, renderer, position = rig
    tools.viz.app_state.camera_mode = "pov"
    tools.viz.app_state.pov_axis = "motion"
    tools.true_pov = False
    tools.pov_offset = [2, 3, 1]
    assert tools.owner.set_pov_camera()
    np.testing.assert_allclose(renderer.get_camera_state().eye, [-3, 2, 1], atol=1e-5)
    tools.owner.scene_query.get_focus_motion_direction = lambda: np.array([1, 0, 0])
    tools.viz.ui_controller.handle_target_focus_changed("Another actor")
    tools.tick(tools.transition[2] + tools.duration)
    np.testing.assert_allclose(renderer.get_camera_state().eye, [2, 3, 1], atol=1e-5)


def test_motion_is_default_pov_choice_and_saved_axis_is_preserved(rig, qtbot):
    from visualizer.src.state import create_initial_state

    tools, _, _ = rig
    tools.viz.app_state = create_initial_state()
    panel = CameraControlPanel(tools.viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    assert panel.widgets["pov_axis_combo"].currentData() == "motion"
    panel.sync_from_state(create_initial_state(pov_axis="forward"))
    assert panel.widgets["pov_axis_combo"].currentData() == "forward"


@pytest.mark.parametrize("kind", ["tx", "rx", "target"])
def test_true_pov_toggle_restores_actor_and_remembers_offsets(rig, qtbot, kind):
    tools, renderer, position = rig
    syncs = []
    tools.viz.node_service = SimpleNamespace(
        sync_pov_entity_visibility=lambda refs: (syncs.append(refs) or True)
    )
    info = {"type": kind, "index": 0}
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [0, 0, 0],
        info,
    )
    tools.viz.app_state.camera_mode = "pov"
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    assert tools.owner.set_pov_camera()
    assert tools.viz.app_state.pov_hidden_node == (kind, 0)

    panel.true_pov.setChecked(False)
    assert tools.viz.app_state.pov_hidden_node is None
    assert all(w.isEnabled() for w in panel.offset)
    for spin, value in zip(panel.offset, [-4, 2, 1]):
        spin.setValue(value)
    assert tools.owner.set_field_of_view(90)
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [-4, 2, 1])

    panel.true_pov.setChecked(True)
    assert tools.viz.app_state.pov_hidden_node == (kind, 0)
    np.testing.assert_allclose(renderer.get_camera_state().eye, position)
    assert tools.pov_offset == [-4, 2, 1]  # Stored, rather than lost on reset.
    assert [w.value() for w in panel.offset] == [0, 0, 0]
    assert not any(w.isEnabled() for w in panel.offset)
    assert tools.owner.set_field_of_view(75)  # Keep saved offsets when editing the lens.
    panel.true_pov.setChecked(False)
    np.testing.assert_allclose(renderer.get_camera_state().eye, position + [-4, 2, 1])
    assert renderer.get_camera_state().fov_deg == 75
    assert tools.viz.app_state.pov_hidden_node is None
    assert len(syncs) == 4


def test_offset_pov_restores_hidden_actor_if_camera_update_fails(rig, monkeypatch):
    tools, renderer, position = rig
    tools.viz.app_state.camera_mode = "pov"
    tools.viz.node_service = SimpleNamespace(sync_pov_entity_visibility=lambda refs: True)
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [0, 0, 0],
        {"type": "rx", "index": 0},
    )
    assert tools.owner.set_pov_camera()
    assert tools.viz.app_state.pov_hidden_node == ("rx", 0)
    tools.true_pov = False
    monkeypatch.setattr(renderer, "set_camera_state", lambda state: False)
    assert not tools.owner.set_pov_camera()
    assert tools.viz.app_state.pov_hidden_node == ("rx", 0)


@pytest.mark.parametrize("offset", [[0, 0, 0], [1, 2, 3]])
def test_true_pov_backward_compatible_camera_json(rig, offset):
    tools, _, _ = rig
    payload = tools.payload()
    del payload["true_pov"]
    payload["pov_offset"] = offset
    tools.restore(payload)
    assert tools.true_pov == (not any(offset))
    tools.true_pov = True
    updated = tools.payload()
    tools.restore(updated)
    assert tools.true_pov and tools.pov_offset == offset
    bad = deepcopy(updated)
    bad["true_pov"] = "yes"
    with pytest.raises(ValueError, match="booleans"):
        tools.restore(bad)
    assert tools.payload() == updated


def test_fixed_camera_tracks_motion_and_actor_switch_without_translating(rig, qtbot):
    tools, renderer, position = rig
    tools.viz.app_state.camera_mode = "overview"
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    before = renderer.get_camera_state()
    panel.run(panel.track_from_here)
    assert tools.follow_heading == "fixed"
    tools.tick(tools.transition[2] + tools.duration)
    for xyz in ([3, 2, 0], [5, 4, 2], [-3, 1, 4]):
        position[:] = xyz
        tools.owner.update_follow_camera_focus()
        state = renderer.get_camera_state()
        np.testing.assert_allclose(state.eye, before.eye, atol=1e-6)
        np.testing.assert_allclose(state.lookat, xyz, atol=1e-6)
        assert state.fov_deg == before.fov_deg
    tools.viz.target_focus_dropdown.currentData = lambda: {"type": "rx", "index": 1}
    position[:] = [20, 10, 0]
    tools.viz.ui_controller.handle_target_focus_changed("RX2")
    began = tools.transition[2]
    for fraction in (0.2, 0.5, 1):
        tools.tick(began + fraction * tools.duration)
        np.testing.assert_allclose(renderer.get_camera_state().eye, before.eye, atol=1e-6)
    np.testing.assert_allclose(renderer.get_camera_state().lookat, position, atol=1e-6)
    assert not any(w.isEnabled() for w in panel.follow_offset)


def test_fixed_camera_coincident_actor_does_not_create_invalid_pose(rig):
    tools, renderer, _ = rig
    tools.follow_heading = "fixed"
    before = renderer.get_camera_state()
    assert tools.follow(np.asarray(before.eye))
    assert renderer.get_camera_state() == before
    assert "coincides" in tools.message


@pytest.mark.parametrize("interrupt", [False, True])
def test_fixed_tracking_target_roundtrips_keep_horizon_level(rig, interrupt):
    tools, renderer, position = rig
    tools.follow_heading = "fixed"
    tools.follow(position)
    before = renderer.get_camera_state()
    initial_rotation = renderer._camera.local.rotation.copy()
    for target in ([20, 0, 9], [0, 0, 0], [-15, 4, -6], [0, 0, 0]):
        position[:] = target
        tools.viz.ui_controller.handle_target_focus_changed("Switched actor")
        began = tools.transition[2]
        for fraction in ((0.2, 0.6) if interrupt else (0.2, 0.6, 1)):
            tools.tick(began + fraction * tools.duration)
            state = renderer.get_camera_state()
            np.testing.assert_allclose(state.eye, before.eye, atol=1e-6)
            direction = np.subtract(state.lookat, state.eye)
            direction /= np.linalg.norm(direction)
            level_up = np.array([0.0, 0.0, 1.0]) - direction * direction[2]
            level_up /= np.linalg.norm(level_up)
            actual_up = renderer._quat_up(renderer._camera.local.rotation)
            np.testing.assert_allclose(actual_up, level_up, atol=1e-6)
    tools.tick(began + tools.duration)
    assert abs(np.dot(renderer._camera.local.rotation, initial_rotation)) == pytest.approx(1)


@pytest.mark.parametrize("mode", ["overview", "follow", "pov"])
def test_main_camera_lens_preserves_pose_and_survives_tracking(rig, qtbot, mode):
    tools, renderer, position = rig
    tools.viz.app_state.camera_mode = mode
    if mode == "pov":
        tools.owner.set_pov_camera()
    elif mode == "follow":
        tools.follow(position)
    panel = CameraControlPanel(tools.viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    fov = panel.widgets["camera_fov_spin"]
    fov.valueChanged.connect(tools.owner.set_field_of_view)
    before = renderer.get_camera_state()
    focused = []
    renderer._focus_canvas = lambda: focused.append(True)
    fov.setValue(85)
    after = renderer.get_camera_state()
    assert after.fov_deg == 85
    np.testing.assert_allclose(after.eye, before.eye, atol=1e-6)
    np.testing.assert_allclose(after.lookat, before.lookat, atol=1e-6)
    np.testing.assert_allclose(after.up, before.up, atol=1e-6)
    assert not focused
    position[:] = [2, 3, 1]
    if mode == "pov":
        tools.owner.set_pov_camera()
    elif mode == "follow":
        tools.owner.update_follow_camera_focus()
    assert renderer.get_camera_state().fov_deg == 85
    assert tools.viz.app_state.camera_mode == mode
    assert fov.isEnabled() and not fov.isHidden()


def test_main_camera_lens_reflects_saved_views_without_stopping_motion(rig, qtbot):
    tools, renderer, _ = rig
    panel = CameraControlPanel(tools.viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    fov = panel.widgets["camera_fov_spin"]
    fov.valueChanged.connect(tools.owner.set_field_of_view)
    tools.owner.set_field_of_view(45)
    tools.save_view("Narrow")
    tools.owner.set_field_of_view(100)
    tools.recall_view("Narrow")
    began = tools.transition[2]
    tools.tick(began + tools.duration / 2)
    before = renderer.get_camera_state()
    panel.refresh_field_of_view()
    assert fov.value() == pytest.approx(before.fov_deg, abs=0.05)
    assert renderer.get_camera_state() == before and tools.transition is not None
    tools.tick(began + tools.duration)
    panel.refresh_field_of_view()
    assert fov.value() == 45
    tools.start_orbit([0, 0, 0])
    fov.setValue(70)
    assert tools.autorotate is None and renderer.get_camera_state().fov_deg == 70


@pytest.mark.parametrize("heading", ["world", "chase", "motion", "fixed"])
def test_reset_behind_zeroes_offsets_and_follows_actor_heading(rig, qtbot, heading):
    tools, renderer, position = rig
    tools.follow_heading = heading
    tools.follow_offset = [12, 4, 6]
    tools.follow_world_yaw = 0.0
    tools.owner.focus_on_target()
    if heading != "fixed":
        tools.follow_offset = [12, 4, 6]
        tools.apply_framing(offset_changed=True)
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    panel.refresh()
    before = renderer.get_camera_state()
    distance = np.linalg.norm(np.asarray(before.eye) - position) if heading == "fixed" else 12
    tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
        position.copy(),
        [np.pi / 2, 0, 0],
        None,
    )
    panel.run(panel.reset_behind)
    assert tools.follow_heading == ("motion" if heading == "motion" else "chase")
    assert tools.viz.app_state.camera_mode == "follow"
    np.testing.assert_allclose(tools.follow_offset, [distance, 0, 0], atol=1e-5)
    np.testing.assert_allclose(
        [spin.value() for spin in panel.follow_offset], [distance, 0, 0], atol=0.01
    )
    position[:] = [2, 3, 1]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(
        renderer.get_camera_state().eye, position + [0, -distance, 0], atol=1e-5
    )
    assert renderer.get_camera_state().fov_deg == before.fov_deg
    assert tools.transition is None


def test_advanced_actor_selection_stays_available_in_overview(rig):
    tools, _, _ = rig
    visibility = []
    tools.viz.advanced_camera_cb = SimpleNamespace(isChecked=lambda: True)
    tools.viz.track_group = SimpleNamespace(setVisible=visibility.append)
    tools.viz.ui_controller.handle_camera_mode_changed("overview", True)
    assert visibility[-1] is True


def test_mouse_framing_sampler_runs_only_on_visible_framing_tab(rig, qtbot):
    from PySide6.QtWidgets import QTabWidget

    tools, _, _ = rig
    panel = AdvancedCameraPanel(tools.viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    tabs = widget.findChild(QTabWidget)
    sampler = tabs.widget(1).timer
    widget.show()
    assert not sampler.isActive()
    tabs.setCurrentIndex(1)
    assert sampler.isActive()
    tabs.setCurrentIndex(0)
    assert not sampler.isActive()
    tabs.setCurrentIndex(1)
    widget.hide()
    assert not sampler.isActive()


@pytest.mark.parametrize("heading", ["world", "chase", "motion"])
def test_follow_numeric_offsets_apply_paused_and_persist_with_motion(rig, qtbot, heading):
    tools, renderer, position = rig
    tools.follow_heading = heading
    tools.owner.focus_on_target()
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    panel.refresh()
    for spin, value in zip(panel.follow_offset, [12, 4, 6]):
        spin.setValue(value)
    state = renderer.get_camera_state()
    yaw = tools.last_yaw if heading in {"chase", "motion"} else tools.follow_world_yaw
    expected = [-12 * np.cos(yaw) - 4 * np.sin(yaw), -12 * np.sin(yaw) + 4 * np.cos(yaw), 6]
    np.testing.assert_allclose(state.eye, expected, atol=1e-5)
    position[:] = [2, 3, 1]
    tools.owner.update_follow_camera_focus()
    np.testing.assert_allclose(
        renderer.get_camera_state().eye, np.asarray(expected) + position, atol=1e-5
    )
    assert not tools.transition
    if heading == "chase":
        tools.owner.scene_query.get_entity_position_orientation_and_info = lambda: (
            position.copy(),
            [np.pi / 2, 0, 0],
            None,
        )
        tools.viz.ui_controller.handle_target_focus_changed("Different actor")
        tools.tick(tools.transition[2] + tools.duration)
        np.testing.assert_allclose(
            renderer.get_camera_state().eye, position + [-4, -12, 6], atol=1e-5
        )


def test_follow_numbers_reflect_mouse_zoom_without_driving_camera(rig, qtbot):
    tools, renderer, position = rig
    tools.follow_heading = "chase"
    tools.follow(position)
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    renderer._controller.zoom(0.5, (0, 0, 1200, 800), animate=False)
    before = renderer.get_camera_state()
    panel.refresh_follow_framing()
    assert renderer.get_camera_state() == before
    np.testing.assert_allclose(
        [w.value() for w in panel.follow_offset],
        [-before.eye[0], before.eye[1], before.eye[2]],
        atol=0.01,
    )
    old_distance = -before.eye[0]
    panel.follow_offset[2].setValue(7)
    state = renderer.get_camera_state()
    assert state.eye[2] == pytest.approx(7)
    assert -state.eye[0] == pytest.approx(old_distance, abs=0.01)


def test_follow_framing_json_validates_and_roundtrips(rig):
    tools, _, _ = rig
    tools.follow_heading = "fixed"
    tools.follow_offset = [12, 4, 6]
    tools.follow_world_yaw = 0.8
    payload = tools.payload()
    tools.restore(payload)
    assert tools.payload() == payload
    for key, value in [("follow_offset", [1, 2]), ("follow_world_yaw", float("nan"))]:
        bad = deepcopy(payload)
        bad[key] = value
        with pytest.raises(ValueError):
            tools.restore(bad)
        assert tools.payload() == payload
    legacy = deepcopy(payload)
    for key in ("follow_offset", "follow_world_yaw", "true_pov"):
        del legacy[key]
    legacy["follow_heading"] = "world"
    tools.restore(legacy)
    assert tools.follow_offset is None and tools.follow_world_yaw is None and tools.true_pov


def test_advanced_panel_views_and_sequence_controls(rig, qtbot):
    tools, _, _ = rig
    panel = AdvancedCameraPanel(tools.viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    panel.name.setText("Wide")
    panel.run(panel.save_view)
    panel.run(panel.add_shot)
    assert len(tools.shots) == 1 and panel.shots.count() == 1
    panel.run(panel.delete_view)
    assert "Remove" in panel.status.text()
    panel.shots.setCurrentRow(0)
    panel.edit_shot(tools, 0)
    panel.run(panel.delete_view)
    assert not tools.views


@pytest.mark.parametrize("motion", ["orbit", "sequence"])
def test_recording_reset_updates_real_view_and_shot_controls(rig, qtbot, motion):
    tools, _, _ = rig
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    tools.save_view("First")
    tools.save_view("Second")
    tools.shots = [
        {"view": "First", "hold": 1.0, "move": 0.0},
        {"view": "Second", "hold": 1.0, "move": 0.5},
    ]
    saved = tools.payload()
    tools.restore(saved)
    assert panel.names.count() == panel.shots.count() == 2
    if motion == "orbit":
        tools.start_orbit([0, 0, 0])
    else:
        tools.start_sequence()
    tools.viz.current_scenario_path = "recording-b"
    tools.owner.reset_for_recording()
    assert not tools.timer.isActive()
    assert tools.autorotate is tools.sequence is None
    assert tools.views == {} and tools.shots == []
    assert panel.names.count() == panel.shots.count() == 0
    assert panel.names.currentIndex() == panel.shots.currentRow() == -1
    with pytest.raises(ValueError, match="nonzero duration"):
        tools.start_sequence()
    tools.restore(saved)
    assert panel.names.count() == panel.shots.count() == 2
    tools.start_sequence()
    assert tools.sequence is not None


@pytest.mark.parametrize("motion", ["orbit", "sequence", "idle"])
def test_export_recovery_restores_status_and_visible_caption(rig, qtbot, motion):
    tools, _, _ = rig
    panel = AdvancedCameraPanel(tools.viz)
    qtbot.addWidget(panel.create_panel())
    if motion == "orbit":
        tools.start_orbit([0, 0, 0])
    elif motion == "sequence":
        tools.save_view("First")
        tools.shots = [{"view": "First", "hold": 1.0, "move": 0.0}]
        tools.start_sequence()
    message = tools.message
    snapshot = tools.suspend_for_export()
    assert panel.status.text() == "Stopped"
    tools.resume_after_export(snapshot)
    assert tools.message == panel.status.text() == message
    assert tools.timer.isActive() is (motion != "idle")
