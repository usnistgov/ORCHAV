"""Demo coordination, input takeover and restoration without a GPU."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QMainWindow, QPushButton

from visualizer.src.app.lifecycle import shutdown_visualizer
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services.demo_service import DemoService
from visualizer.src.types.camera_state import CameraState


class Renderer:
    renderer_id = "pygfx"

    def __init__(self):
        self.camera = CameraState((0, -10, 3), (0, 0, 0), (0, 0, 1), 60)
        self.lights = dict(
            lighting_profile="custom",
            ambient_intensity=0.8,
            headlight_enabled=True,
            headlight_intensity=1.2,
            key_azimuth_deg=31.0,
            key_elevation_deg=-42.0,
            key_intensity=2.0,
            fill_azimuth_deg=-149.0,
            fill_elevation_deg=-60.0,
            fill_intensity=0.5,
            shadows_enabled=True,
        )
        self.ibl = 12000.0
        self.wireframe = False
        self.look_calls = 0
        self.sky = "neutral_outdoor"
        self.skybox = False
        self.environment_calls = []
        self.shadow_quality = "focus"

    def get_ibl_name(self):
        return self.sky

    def get_skybox_visible(self):
        return self.skybox

    def set_ibl(self, name):
        self.sky = name
        self.environment_calls.append(name)
        return True

    def show_skybox(self, visible):
        self.skybox = visible

    def apply_environment_lighting(self, name):
        self.lights["lighting_profile"] = name
        self.ibl = 8000.0
        self.shadow_quality = "scene"

    def get_shadow_quality(self):
        return self.shadow_quality

    def set_shadow_quality(self, name):
        self.shadow_quality = name

    def get_camera_state(self):
        return self.camera

    def get_light_rig_state(self):
        return dict(self.lights)

    def get_ibl_intensity(self):
        return self.ibl

    def get_wireframe(self):
        return self.wireframe

    def restore_light_rig_state(self, lights, *, ibl):
        self.lights, self.ibl = dict(lights), ibl
        self.look_calls += 1

    def set_wireframe(self, enabled):
        self.wireframe = enabled


class Tools:
    def __init__(self):
        self.history = ["original-history"]
        self.exporting = False
        self.autorotate = self.sequence = self.transition = None

    def start_orbit(self, center, *, speed):
        self.autorotate = (center, speed)
        self.history.append("demo-start")

    def stop(self):
        self.autorotate = None

    def reset_for_recording(self):
        self.stop()
        self.history.clear()
        self.views.clear()
        self.shots.clear()
        self.last_yaw = self.follow_position = None


class Appearance:
    def __init__(self, renderer):
        self.renderer = renderer
        self.details = {"surface": "unsaved-custom", "alpha": 0.61, "theme": "retained"}
        self.active_name = "Custom"
        self.previous = {"prior-look": 1}
        self.last_applied = None
        self.notice = "Original custom notice"
        self.changed = SimpleNamespace(emit=Mock())
        self.calls = []

    def capture(self):
        return deepcopy(
            dict(
                details=self.details,
                lights=self.renderer.lights,
                ibl=self.renderer.ibl,
                sky=self.renderer.sky,
                skybox=self.renderer.skybox,
                wireframe=self.renderer.wireframe,
            )
        )

    def restore(self, data, *, name="Custom"):
        self.calls.append(("restore", name))
        self.details = deepcopy(data["details"])
        self.renderer.lights = deepcopy(data["lights"])
        self.renderer.ibl = data["ibl"]
        self.renderer.sky = data["sky"]
        self.renderer.skybox = data["skybox"]
        self.renderer.wireframe = data["wireframe"]
        self.active_name = name
        self.last_applied = self.capture()

    def apply(self, name):
        self.calls.append(("apply", name))
        self.details["surface"] = name
        self.renderer.wireframe = name == "Wireframe"
        self.renderer.ibl = 0
        self.renderer.skybox = False
        self.active_name = name
        self.last_applied = self.capture()


@pytest.fixture
def rig(qapp, qtbot, monkeypatch):
    # The ordinary test scene permits textures; locked launches have a
    # dedicated case and must not leak in from other tests or the shell.
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    viz = QMainWindow()
    qtbot.addWidget(viz)
    viz.renderer = Renderer()
    viz.ready = True
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.current_scenario_path = "scene-a"
    viz.app_state = SimpleNamespace(camera_mode="overview")
    viz.animation_running = False
    viz.loop_cb = QCheckBox(viz)
    tools = Tools()
    viz.camera_controller = SimpleNamespace(presentation=lambda: tools)
    viz.animation_controller = SimpleNamespace(
        toggle_animation=lambda: setattr(viz, "animation_running", not viz.animation_running)
    )
    service = DemoService(viz)
    viz.demo_service = service
    yield viz, service, tools
    service.close()


def test_demo_delegates_camera_and_playback_then_restores_settings(rig):
    viz, service, tools = rig
    lights = deepcopy(viz.renderer.lights)

    assert service.start(orbit_speed=4)
    assert tools.autorotate == ((0, 0, 0), 4)
    assert viz.animation_running and viz.loop_cb.isChecked()
    service.tick(service._next_look)
    assert viz.renderer.lights != lights and viz.renderer.ibl == 0
    visible_camera = CameraState((10, 0, 3), (0, 0, 0), (0, 0, 1), 60)
    viz.renderer.camera = visible_camera

    service.stop()

    assert not service.active and not service._timer.isActive()
    assert tools.autorotate is None and tools.history == ["original-history"]
    assert viz.renderer.camera is visible_camera
    assert viz.renderer.lights == lights and viz.renderer.ibl == 12000
    assert not viz.renderer.wireframe
    assert not viz.animation_running and not viz.loop_cb.isChecked()


@pytest.mark.parametrize("running", [False, True])
def test_demo_restores_original_playback_and_loop_state(rig, running):
    viz, service, _tools = rig
    viz.animation_running = running
    assert service.start(orbit=False, cycle_lighting=False)
    service.stop()
    assert viz.animation_running is running
    assert not viz.loop_cb.isChecked()


def test_visual_only_demo_does_not_change_playback(rig):
    viz, service, _tools = rig
    viz.animation_running = True
    assert service.start(playback=False)
    service.close()
    assert viz.animation_running
    assert not viz.loop_cb.isChecked()


def test_partial_camera_start_failure_retires_its_timer_and_restores_history(rig):
    _viz, service, tools = rig

    def fail_after_start(_center, *, speed):
        tools.autorotate = "started"
        tools.history.append("partial-start")
        raise RuntimeError("camera startup failed")

    tools.start_orbit = fail_after_start
    assert not service.start()
    assert tools.autorotate is None and tools.history == ["original-history"]
    assert not service.active and not service._timer.isActive()


def test_shutdown_cannot_restart_original_playback(rig):
    viz, service, _tools = rig
    viz.animation_running = True
    assert service.start()
    service.close()
    assert not viz.animation_running
    service.tick(1e9)
    assert not viz.animation_running and not service.active


def test_restoration_continues_if_one_presentation_owner_fails(rig):
    viz, service, tools = rig
    assert service.start(include_wireframe=True)
    for _ in range(3):
        service.tick(service._next_look)
    assert viz.renderer.wireframe
    tools.stop = Mock(side_effect=RuntimeError("camera disposed"))
    viz.renderer.restore_light_rig_state = Mock(side_effect=RuntimeError("light disposed"))

    service.close()

    assert not service.active and not service._timer.isActive()
    assert not viz.animation_running and not viz.loop_cb.isChecked()
    assert not viz.renderer.wireframe
    assert "restoration failed" in service.status


def test_slow_look_cycle_coalesces_delayed_ticks_and_wireframe_is_opt_in(rig):
    viz, service, _tools = rig
    assert service.start(orbit=False, playback=False)
    first = service._next_look
    service.tick(first + 100)
    assert service._look_index == 1 and viz.renderer.look_calls == 1
    service.tick(first + 100)
    assert viz.renderer.look_calls == 1
    service.tick(service._next_look)
    service.tick(service._next_look)
    assert service._look_index == 0 and not viz.renderer.wireframe
    service.stop()
    assert service.start(orbit=False, playback=False, include_wireframe=True)
    for _ in range(3):
        service.tick(service._next_look)
    assert viz.renderer.wireframe
    service.stop()
    assert not viz.renderer.wireframe


def test_lighting_cycle_skips_unchanged_wireframe_traversal(rig):
    viz, service, _tools = rig
    viz.renderer.set_wireframe = Mock(wraps=viz.renderer.set_wireframe)
    assert service.start(orbit=False, playback=False)
    for _ in range(3):
        service.tick(service._next_look)
    service.stop()
    viz.renderer.set_wireframe.assert_not_called()


def test_five_second_default_and_exact_custom_appearance_cycle_restore(rig):
    viz, service, _tools = rig
    appearance = Appearance(viz.renderer)
    viz.appearance_preset_service = appearance
    original = appearance.capture()
    assert service.start(orbit=False, playback=False, cycle_appearance=True)
    assert service._look_interval == 5
    for _ in range(5):
        service.tick(service._next_look)
    assert appearance.calls == [
        ("apply", "Detailed"),
        ("apply", "Paper"),
        ("apply", "Dark"),
        ("apply", "Glow"),
        ("restore", "Custom"),
    ]
    assert appearance.capture() == original
    service.tick(service._next_look)
    service.stop()
    assert appearance.capture() == original
    assert appearance.active_name == "Custom"
    assert appearance.previous == {"prior-look": 1}
    assert appearance.last_applied is None
    assert appearance.notice == "Original custom notice"


def _add_dusk(renderer, *, available=True):
    """Model the native recipe's idempotent capture-on-enable semantics."""
    state = {"enabled": False, "available": available, "progress": 0.0}
    saved = {}
    calls = []

    def set_progress(progress, *, request_redraw=True):
        if not state["available"]:
            return False
        if not state["enabled"]:
            saved.update(lights=deepcopy(renderer.lights), ibl=renderer.ibl)
        renderer.ibl = saved["ibl"] * (1 - 0.75 * progress)
        renderer.lights = deepcopy(saved["lights"])
        renderer.lights["ambient_intensity"] *= 1 - 0.5 * progress
        state.update(enabled=True, progress=progress)
        return True

    def set_enabled(enabled):
        calls.append(enabled)
        if enabled:
            return set_progress(1.0)
        if state["enabled"]:
            renderer.lights = deepcopy(saved["lights"])
            renderer.ibl = saved["ibl"]
        state.update(enabled=False, progress=0.0)
        return True

    renderer.get_dusk_state = lambda: dict(state)
    renderer.set_dusk_enabled = set_enabled
    renderer.set_dusk_progress = Mock(side_effect=set_progress)
    return state, calls


@pytest.mark.parametrize("originally_enabled", [False, True])
@pytest.mark.parametrize("demo_enabled", [False, True])
def test_demo_blue_hour_only_on_current_look_and_exact_takeover(
    rig, originally_enabled, demo_enabled
):
    viz, service, _tools = rig
    appearance = viz.appearance_preset_service = Appearance(viz.renderer)
    original_daylight = appearance.capture()
    state, _calls = _add_dusk(viz.renderer)
    viz.renderer.set_dusk_enabled(originally_enabled)
    original_look = appearance.capture()
    assert service.start(
        orbit=False,
        playback=False,
        cycle_appearance=True,
        scenario_dusk=demo_enabled,
    )
    assert state["enabled"] is (demo_enabled or originally_enabled)
    for name in ("Detailed", "Paper", "Dark", "Glow"):
        service.tick(service._next_look)
        assert appearance.active_name == name
        assert not state["enabled"]
    service.tick(service._next_look)
    assert state["enabled"] is (demo_enabled or originally_enabled)
    assert viz.renderer.ibl == (3000 if state["enabled"] else 12000)
    service.stop()
    assert state["enabled"] is originally_enabled
    assert appearance.capture() == original_look
    # Disabling after takeover must recover daylight, not a nested dusk rig.
    viz.renderer.set_dusk_enabled(False)
    assert appearance.capture() == original_daylight


def test_demo_blue_hour_unavailable_does_not_start_or_change_appearance(rig):
    viz, service, _tools = rig
    state, calls = _add_dusk(viz.renderer, available=False)
    assert not service.start(scenario_dusk=True)
    assert "no authored blue-hour" in service.status
    assert not state["enabled"] and calls == []
    assert not service.active and not service._timer.isActive()


def test_demo_with_only_blue_hour_enabled_restores_on_takeover(rig):
    viz, service, _tools = rig
    state, _calls = _add_dusk(viz.renderer)
    assert service.start(orbit=False, playback=False, cycle_lighting=False, scenario_dusk=True)
    assert state["enabled"] and viz.renderer.ibl == 3000
    service.stop()
    assert not state["enabled"] and viz.renderer.ibl == 12000


def test_orbit_only_does_not_suspend_users_blue_hour(rig):
    viz, service, _tools = rig
    state, calls = _add_dusk(viz.renderer)
    viz.renderer.set_dusk_enabled(True)
    assert service.start(playback=False, cycle_lighting=False)
    service.stop()
    assert state["enabled"] and calls == [True]


def test_source_replacement_does_not_apply_old_blue_hour_to_new_scene(rig):
    viz, service, _tools = rig
    _add_dusk(viz.renderer)
    assert service.start(scenario_dusk=True)
    viz.renderer = Renderer()
    state, calls = _add_dusk(viz.renderer)
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.current_scenario_path = "scene-b"
    service.tick()
    assert not service.active
    assert not state["enabled"] and calls == []


def test_blue_hour_panel_availability_and_start_option(rig, qtbot):
    viz, _service, _tools = rig
    state, _calls = _add_dusk(viz.renderer, available=False)
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        toggle = panel.widgets["demo_lighting_mode"]
        assert not toggle.isEnabled() and toggle.currentData() == "off"
        state["available"] = True
        panel.refresh()
        assert toggle.isEnabled()
        toggle.setCurrentIndex(toggle.findData("blue_hour"))
        panel.service.start = Mock()
        panel.start()
        assert panel.service.start.call_args.kwargs["scenario_dusk"] is True
        assert panel.service.start.call_args.kwargs["scenario_sunset"] is False
        toggle.setCurrentIndex(toggle.findData("sunset"))
        panel.start()
        assert panel.service.start.call_args.kwargs["scenario_dusk"] is False
        assert panel.service.start.call_args.kwargs["scenario_sunset"] is True
    finally:
        panel.cleanup()


def test_sunset_uses_one_timer_and_uniform_updates_for_repeated_day_night_roundtrips(
    rig, monkeypatch
):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    state, _calls = _add_dusk(viz.renderer)
    clock = [10.0]
    monkeypatch.setattr(demo_service, "perf_counter", lambda: clock[0])
    assert service.start(
        orbit=False, playback=False, cycle_lighting=False, scenario_sunset=True, sunset_duration=10
    )
    assert state["enabled"] and state["progress"] == 0
    assert viz.renderer.ibl == 12000
    assert service._timer.interval() == 250
    assert "sunset cycle" in service.status
    clock[0] = 13
    service.tick()
    assert state["progress"] == pytest.approx(0.5)
    assert service._timer.interval() == 33
    assert viz.renderer.ibl == 7500
    updates = viz.renderer.set_dusk_progress.call_count
    service.tick()
    assert viz.renderer.set_dusk_progress.call_count == updates
    assert viz.renderer.look_calls == 0
    clock[0] = 15
    service.tick()
    assert state["progress"] == 1 and service._timer.interval() == 250
    assert "sunset cycle" in service.status
    updates = viz.renderer.set_dusk_progress.call_count
    clock[0] = 15.5
    service.tick()
    assert state["progress"] == 1
    assert viz.renderer.set_dusk_progress.call_count == updates
    clock[0] = 18
    service.tick()
    assert state["progress"] == pytest.approx(0.5)
    assert service._timer.interval() == 33
    clock[0] = 20
    service.tick()
    assert state["progress"] == 0 and viz.renderer.ibl == 12000
    assert service._timer.interval() == 250
    clock[0] = 23
    service.tick()
    assert state["progress"] == pytest.approx(0.5)
    assert viz.renderer.look_calls == 0
    service.stop()
    assert not state["enabled"] and viz.renderer.ibl == 12000
    assert not service._timer.isActive() and service._timer.interval() == 250


def test_sunset_restarts_only_on_current_look_and_preserves_technical_views(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    appearance = viz.appearance_preset_service = Appearance(viz.renderer)
    state, _calls = _add_dusk(viz.renderer)
    monkeypatch.setattr(demo_service, "perf_counter", lambda: 10.0)
    assert service.start(orbit=False, playback=False, cycle_appearance=True, scenario_sunset=True)
    service.tick(service._sunset_started + 1.5)
    assert state["progress"] == pytest.approx(0.5)
    assert appearance.calls == []
    for name in ("Detailed", "Paper", "Dark", "Glow"):
        service.tick(service._next_look)
        assert appearance.active_name == name and not state["enabled"]
        if name == "Detailed":
            assert viz.renderer.set_dusk_progress.call_args.args[0] == 0
        assert service._sunset_started is None and service._timer.interval() == 250
        calls = viz.renderer.set_dusk_progress.call_count
        service.tick(service._next_look - 1)
        assert viz.renderer.set_dusk_progress.call_count == calls
    service.tick(service._next_look)
    assert appearance.active_name == "Custom"
    assert state["enabled"] and state["progress"] == 0
    assert service._timer.interval() == 250
    assert "sunset cycle" in service.status
    service.tick(service._sunset_started + 2.5)
    assert state["progress"] == 1
    service.tick(service._sunset_started + 4.5)
    assert state["progress"] == pytest.approx(0.25)
    assert appearance.calls == [
        ("apply", "Detailed"),
        ("apply", "Paper"),
        ("apply", "Dark"),
        ("apply", "Glow"),
        ("restore", "Custom"),
    ]


def test_sunset_clock_begins_after_native_rig_preparation(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    _add_dusk(viz.renderer)
    clock = [10.0]
    monkeypatch.setattr(demo_service, "perf_counter", lambda: clock[0])
    prepare = viz.renderer.set_dusk_progress.side_effect

    def slow_prepare(progress, **kwargs):
        clock[0] += 2
        return prepare(progress, **kwargs)

    viz.renderer.set_dusk_progress.side_effect = slow_prepare
    assert service.start(orbit=False, playback=False, cycle_lighting=False, scenario_sunset=True)
    assert service._sunset_started == clock[0] == 12
    assert service._next_look == 17


@pytest.mark.parametrize("original_progress", [None, 0.0, 0.35, 1.0])
def test_sunset_takeover_restores_original_level_without_nesting_rigs(rig, original_progress):
    viz, service, _tools = rig
    appearance = viz.appearance_preset_service = Appearance(viz.renderer)
    state, _calls = _add_dusk(viz.renderer)
    daylight = appearance.capture()
    if original_progress is not None:
        viz.renderer.set_dusk_progress(original_progress)
    original = appearance.capture()
    assert service.start(orbit=False, playback=False, cycle_appearance=True, scenario_sunset=True)
    assert state["progress"] == 0 and viz.renderer.ibl == 12000
    service.tick(service._sunset_started + 1)
    service.stop()
    assert state["enabled"] is (original_progress is not None)
    assert state["progress"] == (original_progress or 0.0)
    assert appearance.capture() == original
    viz.renderer.set_dusk_enabled(False)
    assert appearance.capture() == daylight


def test_sunset_source_change_stops_fast_timer_without_mutating_new_renderer(rig):
    viz, service, _tools = rig
    _add_dusk(viz.renderer)
    assert service.start(orbit=False, playback=False, cycle_lighting=False, scenario_sunset=True)
    viz.renderer = Renderer()
    state, calls = _add_dusk(viz.renderer)
    viz.scenario = SimpleNamespace(data_mode="files")
    service.tick()
    assert not service.active and service._sunset_started is None
    assert not service._timer.isActive() and service._timer.interval() == 250
    assert not state["enabled"] and not calls
    viz.renderer.set_dusk_progress.assert_not_called()


@pytest.mark.parametrize("motion", ["paused", "playback", "orbit", "tour_hold"])
def test_sunset_reuses_playback_or_orbit_draws_and_redraws_paused_views(rig, motion):
    viz, service, tools = rig
    _add_dusk(viz.renderer)
    assert service.start(
        orbit=motion == "orbit",
        playback=motion == "playback",
        cycle_lighting=False,
        scenario_sunset=True,
    )
    if motion == "tour_hold":
        service._tour_owned = True
        service._camera_tools = tools
        service._snapshot.update(views=[], shots=[])
        tools.sequence = "holding"
    service.tick(service._sunset_started + 1)
    assert viz.renderer.set_dusk_progress.call_args.kwargs == {
        "request_redraw": motion in ("paused", "tour_hold")
    }


def test_failed_sunset_progress_retires_fast_timer_and_restores_daylight(rig):
    viz, service, _tools = rig
    state, _calls = _add_dusk(viz.renderer)
    assert service.start(orbit=False, playback=False, cycle_lighting=False, scenario_sunset=True)
    viz.renderer.set_dusk_progress = Mock(return_value=False)
    service.tick(service._sunset_started + 1)
    assert not service.active and not service._timer.isActive()
    assert not state["enabled"] and viz.renderer.ibl == 12000
    assert "no longer available" in service.status


def test_ibl_cycle_uses_only_installed_skies_and_restores_default(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    appearance = Appearance(viz.renderer)
    viz.appearance_preset_service = appearance
    original = appearance.capture()
    catalog = {key: SimpleNamespace(id=key, label=key) for key in ("sunny", "missing", "cloudy")}
    monkeypatch.setattr(demo_service, "environment_catalog", lambda: catalog)
    monkeypatch.setattr(
        demo_service,
        "cached_environment",
        lambda asset: None if asset.id == "missing" else "cached.hdr",
    )
    assert service.start(orbit=False, playback=False, cycle_ibl=True)
    assert service._environment_cycle == (
        ("neutral_outdoor", "current sky"),
        ("sunny", "sunny"),
        ("cloudy", "cloudy"),
    )
    for expected in ("sunny", "cloudy"):
        service.tick(service._next_look)
        assert viz.renderer.sky == expected and viz.renderer.skybox
        assert viz.renderer.ibl == 8000
    service.stop()
    assert appearance.capture() == original
    assert appearance.calls == []  # No material or geometry publication for sky-only changes.
    assert "missing" not in viz.renderer.environment_calls


def test_ibl_unavailable_does_not_start_or_mutate_demo(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    monkeypatch.setattr(demo_service, "environment_catalog", lambda: {})
    assert not service.start(cycle_ibl=True)
    assert "Install another sky" in service.status
    assert viz.renderer.sky == "neutral_outdoor"
    assert not viz.animation_running and not service._timer.isActive()


def test_combined_preset_and_sky_cycle_keeps_backdrop_visible_then_restores(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    appearance = Appearance(viz.renderer)
    viz.appearance_preset_service = appearance
    original = appearance.capture()
    monkeypatch.setattr(
        demo_service,
        "environment_catalog",
        lambda: {"sunny": SimpleNamespace(id="sunny", label="Sunny")},
    )
    monkeypatch.setattr(demo_service, "cached_environment", lambda _asset: "cached.hdr")
    assert service.start(orbit=False, playback=False, cycle_ibl=True, cycle_appearance=True)
    service.tick(service._next_look)
    assert appearance.active_name == "Detailed"
    assert viz.renderer.skybox and viz.renderer.sky == "sunny"
    service.stop()
    assert appearance.capture() == original


def test_slow_switch_gets_full_dwell_and_does_not_catch_up(rig, monkeypatch):
    from visualizer.src.services import demo_service

    viz, service, _tools = rig
    appearance = Appearance(viz.renderer)
    viz.appearance_preset_service = appearance
    clock = [10.0]
    monkeypatch.setattr(demo_service, "perf_counter", lambda: clock[0])
    real_apply = appearance.apply

    def slow_apply(name):
        real_apply(name)
        clock[0] += 8

    appearance.apply = slow_apply
    assert service.start(orbit=False, playback=False, cycle_appearance=True)
    clock[0] = 15
    service.tick()
    assert service.last_switch_duration_s == 8
    assert service._next_look == 28
    service.tick(23)
    assert appearance.calls == [("apply", "Detailed")]


def test_appearance_cycle_does_not_require_textures(rig, monkeypatch):
    viz, service, _tools = rig
    viz.appearance_preset_service = Appearance(viz.renderer)
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    assert service.start(orbit=False, playback=False, cycle_appearance=True)
    assert service._appearance_cycle == ("Paper", "Dark", "Glow")


def test_manual_mouse_input_takes_over_and_reaches_original_widget(rig, qtbot):
    viz, service, _tools = rig
    button = QPushButton("Interact", viz)
    button.resize(100, 40)
    viz.show()
    clicked = Mock()
    button.clicked.connect(clicked)
    assert service.start()

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    assert not service.active
    clicked.assert_called_once()


def test_pause_click_stops_demo_and_pauses_instead_of_toggling_back_to_play(rig):
    viz, service, _tools = rig
    viz.play_btn = QPushButton("Pause", viz)
    viz.play_btn.clicked.connect(viz.animation_controller.toggle_animation)
    viz.play_btn.resize(100, 40)
    viz.show()
    assert service.start()
    QTest.mouseClick(viz.play_btn, Qt.MouseButton.LeftButton)
    assert not service.active and not viz.animation_running


def test_loop_click_acts_on_the_checked_state_visible_during_demo(rig):
    viz, service, _tools = rig
    viz.loop_cb.setText("Loop")
    viz.loop_cb.resize(100, 40)
    viz.show()
    assert service.start()
    assert viz.loop_cb.isChecked()
    QTest.mouseClick(viz.loop_cb, Qt.MouseButton.LeftButton, pos=QPoint(8, 20))
    assert not service.active and not viz.loop_cb.isChecked()
    assert not viz.animation_running


def test_space_shortcut_pauses_demo_playback(rig, qtbot):
    viz, service, _tools = rig
    shortcut = QShortcut(QKeySequence("Space"), viz)
    shortcut.activated.connect(viz.animation_controller.toggle_animation)
    viz.show()
    viz.activateWindow()
    qtbot.wait(10)
    assert service.start()
    QTest.keyClick(viz, Qt.Key.Key_Space)
    assert not service.active and not viz.animation_running


def test_escape_takes_over_without_a_separate_shortcut(rig):
    viz, service, _tools = rig
    viz.show()
    assert service.start()
    QTest.keyClick(viz, Qt.Key.Key_Escape)
    assert not service.active


def test_unrelated_window_input_does_not_stop_demo(rig, qtbot):
    _viz, service, _tools = rig
    other = QPushButton("Unrelated")
    qtbot.addWidget(other)
    other.show()
    assert service.start()
    QTest.mouseClick(other, Qt.MouseButton.LeftButton)
    assert service.active


def test_new_scene_never_receives_previous_scene_appearance(rig):
    viz, service, _tools = rig
    assert service.start()
    service.tick(service._next_look)
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.current_scenario_path = "scene-b"
    viz.renderer = Renderer()
    viz.renderer.ibl = 777
    service.tick()
    assert not service.active
    assert viz.renderer.ibl == 777 and viz.renderer.look_calls == 0


def test_external_camera_interruption_preserves_demo_and_playback(rig):
    viz, service, tools = rig
    assert service.start()
    tools.stop()
    service.tick()
    assert service.active and service.camera_paused and viz.animation_running
    assert service.resume_camera() and tools.autorotate is not None


def test_demo_blocks_camera_export_even_without_orbit(rig):
    _viz, service, tools = rig
    tools.exporting = True
    assert not service.start(orbit=False)
    assert "export" in service.status


def test_demo_restores_appearance_metadata_and_notifies_controls(rig):
    viz, service, _tools = rig
    metadata = dict(
        active_name="Outdoor presentation",
        previous={"old": 1},
        last_applied={"look": 2},
        notice="Original notice",
    )
    appearance = SimpleNamespace(**deepcopy(metadata), changed=SimpleNamespace(emit=Mock()))
    viz.appearance_preset_service = appearance
    assert service.start()
    service.tick(service._next_look)
    assert viz.renderer.lights["lighting_profile"] == "custom"
    appearance.active_name = "Custom"
    appearance.previous = {"demo": 3}
    service.stop()
    assert {key: getattr(appearance, key) for key in metadata} == metadata
    appearance.changed.emit.assert_called_once()


def test_scene_cleanup_stops_demo_before_clearing_camera_history(rig, monkeypatch):
    from visualizer.src.app import scenario_workflow
    from visualizer.src.controllers.camera_controller import CameraController

    viz, service, tools = rig
    tools.views, tools.shots = {}, []
    camera_owner = SimpleNamespace(_presentation=tools, _camera_presets={})
    viz.camera_controller.reset_for_recording = lambda: CameraController.reset_for_recording(
        camera_owner
    )
    viz.cancel_startup_preload = Mock()
    viz.reset_preloading_state = Mock()
    viz.trajectory_load_coordinator = SimpleNamespace(reset=Mock())
    viz.scene_service = SimpleNamespace(
        cleanup_previous_scene=lambda: service.stop(restore_playback=False)
    )
    viz._set_frame_data_available = Mock()
    monkeypatch.setattr(scenario_workflow, "clear_runtime_extension_services", Mock())
    assert service.start()

    scenario_workflow.cleanup_previous_scene(viz)

    assert not service.active and tools.history == []
    assert not viz.animation_running


def test_demo_uses_real_camera_presentation_and_leaves_visible_pose_on_stop(rig):
    from visualizer.src.services.camera_presentation import CameraPresentation

    viz, service, _fake_tools = rig
    viz.set_state = lambda **updates: [
        setattr(viz.app_state, key, value) for key, value in updates.items()
    ]
    viz.renderer.set_camera_input_callback = Mock()
    viz.renderer.cancel_camera_gestures = Mock()
    viz.renderer.set_camera_state = lambda pose: setattr(viz.renderer, "camera", pose) or True
    owner = SimpleNamespace(
        visualizer=viz,
        restore_pov_entity_visibility=Mock(),
        clear_pre_pov_camera_state=Mock(),
    )
    tools = CameraPresentation(owner)
    viz.camera_controller.presentation = lambda: tools
    initial = viz.renderer.camera
    try:
        assert service.start(playback=False, cycle_lighting=False, orbit_speed=3)
        tools.tick(tools.autorotate[3] + 1)
        visible = viz.renderer.camera
        assert visible != initial
        service.stop()
        assert viz.renderer.camera is visible
        assert tools.autorotate is None and not tools.timer.isActive()
    finally:
        tools.close()


def test_demo_scene_without_frames_can_run_without_playback(rig):
    viz, service, _tools = rig
    viz.get_available_animation_steps = lambda: []
    assert not service.start()
    assert "No recorded frames" in service.status
    assert service.start(playback=False)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda v: setattr(v.renderer, "renderer_id", "open3d"), "pygfx"),
        (lambda v: setattr(v, "ready", False), "Open a scene"),
        (lambda v: setattr(v.app_state, "camera_mode", "follow"), "Overview"),
        (
            lambda v: setattr(v, "live_preview_service", SimpleNamespace(enabled=True)),
            "actor editing",
        ),
        (lambda v: setattr(v.scenario, "data_mode", "live_grpc"), "local recorded frames"),
    ],
)
def test_demo_rejects_unavailable_or_conflicting_owners(rig, change, expected):
    viz, service, _tools = rig
    change(viz)
    assert not service.start()
    assert expected in service.status
    assert not service._timer.isActive() and not viz.animation_running


def test_panel_controls_start_and_stop_existing_service(rig, qtbot):
    viz, _service, _tools = rig
    viz.appearance_preset_service = Appearance(viz.renderer)
    panel = DemoPanel(viz)
    panel.service.available_environments = lambda: (("installed", "Installed sky"),)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        assert panel.widgets["demo_appearance"].isChecked()
        assert not panel.widgets["demo_lighting"].isEnabled()
        panel.widgets["demo_appearance"].setChecked(False)
        panel.widgets["demo_playback"].setChecked(False)
        panel.widgets["demo_lighting"].setChecked(False)
        assert not panel.widgets["demo_wireframe"].isEnabled()
        panel.widgets["demo_lighting"].setChecked(True)
        assert panel.widgets["demo_wireframe"].isEnabled()
        assert panel.widgets["demo_interval"].value() == 5
        panel.widgets["demo_appearance"].setChecked(True)
        assert not panel.widgets["demo_lighting"].isEnabled()
        assert not panel.widgets["demo_wireframe"].isEnabled()
        panel.widgets["demo_appearance"].setChecked(False)
        panel.widgets["demo_ibl"].setChecked(True)
        assert not panel.widgets["demo_lighting"].isEnabled()
        panel.widgets["demo_ibl"].setChecked(False)
        assert panel.widgets["demo_lighting"].isEnabled()
        panel.start()
        assert panel.service.active and not viz.animation_running
        assert not panel.widgets["demo_start"].isEnabled()
        panel.widgets["demo_stop"].click()
        assert not panel.service.active and panel.widgets["demo_start"].isEnabled()
    finally:
        panel.cleanup()


def test_shutdown_restores_demo_before_persisting_session():
    events = []
    viz = SimpleNamespace(
        demo_service=SimpleNamespace(close=lambda: events.append("demo")),
        session_service=SimpleNamespace(auto_save_on_exit=lambda: events.append("save")),
        renderer=SimpleNamespace(close=lambda: events.append("renderer")),
    )
    assert shutdown_visualizer(viz, persist_state=True)
    assert events.index("demo") < events.index("save") < events.index("renderer")


@pytest.mark.parametrize("palette_change", [False, True])
@pytest.mark.parametrize("initially_running", [False, True])
@pytest.mark.parametrize("retired", [None, "source", "epoch", "shutdown"])
def test_stop_publishes_restored_current_frame_packet(
    qtbot, palette_change, initially_running, retired
):
    from types import MethodType

    import numpy as np
    from PySide6.QtCore import QTimer

    from shared.statistics.themes import theme_manager
    from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
    from tests.visualizer.unit.test_frame_pipeline import dummy_visualizer
    from visualizer.src.io.packed_frame_payload import (
        standard_frame_to_visual_frame,
        visual_frame_read_request,
    )
    from visualizer.src.pipeline.core import MPCCore
    from visualizer.src.pipeline.frame_pipeline import FramePipeline
    from visualizer.src.renderers.protocol import RendererCapabilities
    from visualizer.src.services.mpc_appearance_service import MpcAppearanceService
    from visualizer.visualizer import OrchavVisualizer

    original_theme = theme_manager.current
    theme_manager.set_colormap("continuous", "viridis")
    packets = []
    viz = dummy_visualizer.__wrapped__()
    viz.set_state = MethodType(OrchavVisualizer.set_state, viz)
    viz.vis_initialized, viz.vis = True, object()
    viz.animation_step = 0
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.current_scenario_path = "semantic-mpc-fixture"
    viz.animation_running = True
    viz.update_pending = False
    viz.update_timer_coalesce = QTimer()
    viz.update_timer_coalesce.setSingleShot(True)
    viz.schedule_update = MethodType(OrchavVisualizer.schedule_update, viz)
    viz.update_timer_coalesce.timeout.connect(MethodType(OrchavVisualizer._flush_update, viz))
    viz.renderer.capabilities = RendererCapabilities()
    viz.renderer.get_line_width = lambda: 1.0
    viz.renderer.get_point_size = lambda: 4.0
    viz.renderer.set_line_width = lambda value: None
    viz.renderer.set_point_size = lambda value: None

    def apply(packet):
        packets.append((len(packet.mpc_lines), np.array(packet.mpc_colors), packet.colorbar))
        return True

    viz.renderer.apply_frame = apply
    viz.mpc_core = MPCCore(visualizer=viz)
    viz.pipeline = FramePipeline(viz)
    viz._process_frame_step = viz.pipeline.update
    raw = standard_frame_to_visual_frame(
        build_standard_mpc_frame(), request=visual_frame_read_request()
    )
    viz.pipeline.load_frame = lambda _step: raw
    mpc = MpcAppearanceService(viz)
    before_state, before_mpc = viz.app_state, mpc.capture()
    demo = DemoService(viz)
    try:
        assert viz.pipeline.update(0)
        original_packet = packets[-1]
        viz.set_state(color_mode="delay", topk_render_enabled=True, topk_render_max_paths=1)
        if palette_change:
            theme_manager.set_colormap("continuous", "orchav_glow")
        assert viz.pipeline.update(0)
        assert packets[-1][0] == 1 < original_packet[0]
        viz.animation_controller = SimpleNamespace(
            toggle_animation=lambda: setattr(viz, "animation_running", not viz.animation_running)
        )
        demo.active = True
        demo._snapshot = dict(
            renderer=viz.renderer,
            scenario=viz.scenario,
            source=viz.current_scenario_path,
            source_epoch=0,
            owned_palettes={"continuous": "orchav_glow"} if palette_change else {},
            mpc_appearance=before_mpc,
            mpc=mpc,
            presentation_state={
                key: getattr(before_state, key)
                for key in ("color_mode", "topk_render_enabled", "topk_render_max_paths")
            },
            dusk=None,
            lights=None,
            playback=True,
            running=initially_running,
            play_direction=None,
            loop=None,
            appearance_metadata=None,
        )
        if retired == "source":
            viz.current_scenario_path = "replacement"
        elif retired == "epoch":
            viz._mpc_presented_source_epoch = 1
        elif retired == "shutdown":
            viz._shutdown_started = True
        demo.stop()
        if retired is not None:
            assert not viz.update_pending
            assert not viz.update_timer_coalesce.isActive()
            return
        assert viz.update_pending
        qtbot.waitUntil(lambda: not viz.update_pending)
        assert viz.app_state.step == viz.animation_step == 0
        assert viz.animation_running is initially_running
        assert viz.app_state.color_mode == "reflection_order"
        assert not viz.app_state.topk_render_enabled
        assert packets[-1][0] == original_packet[0]
        np.testing.assert_array_equal(packets[-1][1], original_packet[1])
        assert packets[-1][2] == original_packet[2]
    finally:
        viz.update_timer_coalesce.stop()
        demo.close()
        theme_manager.remove_listener(viz.mpc_core._on_theme_change)
        theme_manager._current_theme = original_theme


@pytest.mark.parametrize("preset", ["Dark", "Glow", "Standard"])
def test_direct_path_preset_publishes_matching_current_frame_legend(qtbot, preset):
    from types import MethodType

    from PySide6.QtCore import QTimer

    from shared.statistics.themes import theme_manager
    from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
    from tests.visualizer.unit.test_frame_pipeline import dummy_visualizer
    from visualizer.src.io.packed_frame_payload import (
        standard_frame_to_visual_frame,
        visual_frame_read_request,
    )
    from visualizer.src.pipeline.core import MPCCore
    from visualizer.src.pipeline.frame_pipeline import FramePipeline
    from visualizer.src.renderers.protocol import RendererCapabilities
    from visualizer.src.services.mpc_appearance_service import MpcAppearanceService
    from visualizer.visualizer import OrchavVisualizer

    original_theme = theme_manager.current
    theme_manager.set_colormap("continuous", "viridis")
    packets = []
    viz = dummy_visualizer.__wrapped__()
    viz.set_state = MethodType(OrchavVisualizer.set_state, viz)
    viz.vis_initialized, viz.vis = True, object()
    viz.animation_step = 0
    viz.animation_running = False
    viz.update_pending = False
    viz.update_timer_coalesce = QTimer()
    viz.update_timer_coalesce.setSingleShot(True)
    viz.schedule_update = MethodType(OrchavVisualizer.schedule_update, viz)
    viz.update_timer_coalesce.timeout.connect(MethodType(OrchavVisualizer._flush_update, viz))
    viz.renderer.capabilities = RendererCapabilities()
    viz.renderer.get_line_width = lambda: 1.0
    viz.renderer.get_point_size = lambda: 4.0
    viz.renderer.set_line_width = lambda value: None
    viz.renderer.set_point_size = lambda value: None
    viz.renderer.apply_frame = lambda packet: packets.append(packet) or True
    viz.mpc_core = MPCCore(visualizer=viz)
    viz.pipeline = FramePipeline(viz)
    viz._process_frame_step = viz.pipeline.update
    raw = standard_frame_to_visual_frame(
        build_standard_mpc_frame(), request=visual_frame_read_request()
    )
    viz.pipeline.load_frame = lambda _step: raw
    try:
        if preset == "Standard":
            viz.set_state(color_mode="delay")
        assert viz.pipeline.update(0)
        original_line_count = len(packets[-1].mpc_lines)
        MpcAppearanceService(viz).apply(preset)
        intermediate_count = len(packets)
        assert viz.update_pending
        qtbot.waitUntil(lambda: not viz.update_pending)
        assert len(packets) == intermediate_count + 1
        assert viz.app_state.step == viz.animation_step == 0
        assert not viz.animation_running
        assert len(packets[-1].mpc_lines) == original_line_count
        if preset == "Standard":
            assert packets[-1].colorbar is None
            assert viz.app_state.color_mode == "reflection_order"
        else:
            assert packets[-1].colorbar[0] == "Delay (ns)"
            assert viz.app_state.color_mode == "delay"
    finally:
        viz.update_timer_coalesce.stop()
        theme_manager.remove_listener(viz.mpc_core._on_theme_change)
        theme_manager._current_theme = original_theme
