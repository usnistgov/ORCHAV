"""Report reveals and optional array work retain scene/source ownership."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pygfx
import pytest
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from tests.visualizer.unit.test_report_geometry import report_request
from visualizer.src.experiments.scene_explanation import SceneExplanationController
from visualizer.src.sensing.scene_explanation import SceneExplanationControls


class Overlay:
    def __init__(self):
        self.layers = {}

    def clear(self):
        self.layers.clear()

    def remove(self, name):
        self.layers.pop(name, None)

    def angular_cell(self, cell):
        self.layers["angle"] = cell

    def report_position(self, point):
        self.layers["position"] = point

    def receive_response(self, obj):
        self.layers["response"] = obj


@pytest.fixture
def scene(qtbot, rig):  # noqa: F811 - imported pytest fixture
    tools, renderer, _ = rig
    renderer._gfx, renderer._scene = pygfx, pygfx.Scene()
    renderer._qt_window_closed = False
    renderer.set_hud_overlay = lambda *args, **kwargs: None
    renderer.clear_hud_overlay = lambda *args, **kwargs: None
    request, selection = report_request()
    dashboard = QWidget()
    qtbot.addWidget(dashboard)
    dashboard._last_request, dashboard._linked_selection = request, selection
    dashboard.selected_world_position = lambda: (
        (4, 5, 6) if dashboard._linked_selection is not None else None
    )
    shell = QCheckBox(dashboard)
    controls = SceneExplanationControls(shell, dashboard)
    dashboard.scene_explanation = controls
    QVBoxLayout(dashboard).addWidget(controls)
    controller = SceneExplanationController(tools.viz)
    controller.overlay = Overlay()
    _runtime_proof(dashboard, controller)
    controls.changed.connect(lambda: controller.sync(dashboard, request.effective_display_step))
    dashboard.show()
    controller.sync(dashboard, request.effective_display_step)
    yield dashboard, controls, controller
    controller.close()
    dashboard.close()


def test_reveal_uses_cinematic_geometry_then_stops_on_result_change(scene):
    dashboard, controls, controller = scene
    assert controller.executor is None
    controls._play()
    assert not controls.range_shell.isChecked() and controls.timer.isActive()
    assert controller.cinematic.phase == "range"
    controls._advance()
    assert not controls.range_shell.isChecked()
    assert controller.cinematic.phase == "angle"
    controls._advance()
    assert controller.cinematic.phase == "position"
    controls._advance()
    assert not controls.timer.isActive()
    assert controls.phase.currentData() == "position"
    dashboard._linked_selection = None
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert controls.phase.currentData() == "off"
    assert not controller.overlay.layers
    assert controller.cinematic.overlay is None
    assert controller.executor is None


def test_stale_scene_clears_geometry_and_hidden_controls_stop_reveal(scene):
    dashboard, controls, controller = scene
    controls.phase.setCurrentIndex(2)
    assert controller.cinematic.phase == "angle"
    controller.sync(dashboard, 999)
    assert not controller.overlay.layers
    assert "Waiting" in controls.note.text()
    controls._play()
    controls.hide()
    assert not controls.timer.isActive()
    assert not controls.range_shell.isChecked()
    assert not controller.overlay.layers


def test_playback_stops_reveal_and_pause_allows_restart_without_mode_toggle(scene):
    dashboard, controls, controller = scene
    controls._play()
    reveal = controller.cinematic
    assert reveal.fade.isActive() and reveal.camera.timer.isActive()
    controller.viz.animation_running = True
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert not controls.timer.isActive() and not reveal.fade.isActive()
    assert not reveal.camera.timer.isActive() and reveal.overlay is None
    assert not controls.play.isEnabled() and not controls.phase.isEnabled()
    assert "Pause playback" in controls.note.text()
    controller.viz.animation_running = False
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert controls.play.isEnabled() and controls.phase.isEnabled()
    controls._play()
    assert reveal.phase == "range" and reveal.fade.isActive()
    assert reveal.camera.timer.isActive()


def test_missing_beam_skips_angle_without_hiding_saved_position(scene):
    dashboard, controls, controller = scene
    dashboard._last_request.display_frame["sensing"]["detections"][0].pop("beam_index")
    controls._play()
    controls._advance()
    assert controls.timer.isActive()
    assert controls.available_phases == ("range", "position")
    assert controller.cinematic.phase == "position"
    assert not controller.overlay.layers


def test_start_without_saved_report_never_runs_timer_or_leaves_range_shell_on(scene):
    dashboard, controls, controller = scene
    dashboard._linked_selection = None
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    controls._play()
    assert not controls.timer.isActive()
    assert controls.phase.currentData() == "off"
    assert not controls.range_shell.isChecked()
    assert "unavailable" in controls.note.text()


def test_missing_axes_reveals_saved_position_without_reentrant_reset(scene):
    dashboard, controls, controller = scene
    dashboard._last_request = replace(
        dashboard._last_request, rd_result=None, detection_stage_bundle=None
    )
    dashboard._linked_selection = replace(dashboard._linked_selection, cell=None)
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    # Production also connects the existing shell checkbox to overlay refresh.
    controls.range_shell.toggled.connect(
        lambda: controller.sync(dashboard, dashboard._last_request.effective_display_step)
    )
    controls._play()
    assert controls.timer.isActive()
    assert not controls.range_shell.isChecked()
    assert controls.phase.currentData() == "position"
    assert "POSITION" in controls.note.text()
    assert not controller.overlay.layers


def test_saved_position_survives_unavailable_range_geometry(scene, monkeypatch):
    dashboard, controls, controller = scene
    dashboard._last_request = replace(
        dashboard._last_request, rd_result=None, detection_stage_bundle=None
    )
    dashboard._linked_selection = replace(dashboard._linked_selection, cell=None)
    controller.sync(dashboard, dashboard._last_request.effective_display_step)

    def no_range_geometry(*_):
        raise ValueError("Range axes unavailable")

    monkeypatch.setattr(
        "visualizer.src.experiments.cinematic_reveal.build_range_locus", no_range_geometry
    )
    controls.phase.setCurrentIndex(3)
    assert set(controller.cinematic.stages) == {"position"}
    np.testing.assert_array_equal(controller.cinematic.stages["position"][0], (4, 5, 6))
    assert "saved spatial estimate" in controls.note.text()
    dashboard._last_request.display_frame["sensing"]["detections"] = []
    dashboard._linked_selection = replace(dashboard._linked_selection, detection_index=None)
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    controls.phase.setCurrentIndex(3)
    assert set(controller.cinematic.stages) == {"position"}
    assert "TRACK" in controls.note.text()


def test_report_selected_before_rd_arrives_gets_exact_cell_without_new_user_selection(qtbot):
    from tests.visualizer.unit.test_sensing_selection_geometry import spatial_request
    from visualizer.src.sensing.sensing_window import SensingWindow

    request, _ = spatial_request()
    request.display_frame["sensing"]["detections"][0].update(range_bin=1, doppler_bin=2)
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("current")
    window.update_from_request(replace(request, rd_result=None))
    window.select_detection_index(0)
    assert window._linked_selection.cell is None
    notifications = []
    window.selection_changed.connect(notifications.append)
    window.update_from_request(request)
    cell = window._linked_selection.cell
    assert (cell.range_bin, cell.doppler_bin) == (1, 2)
    assert cell.source == request.source_identity
    assert cell.origin == "detection"
    assert not notifications


def test_recipe_change_stops_reveal_even_at_same_source_scene_and_acquisition(scene):
    dashboard, controls, controller = scene
    controls._play()
    dashboard._last_request.display_frame["sensing"]["temporal"]["result"][
        "processing_recipe_id"
    ] = "another-recipe"
    controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert not controls.timer.isActive()
    assert controls.phase.currentData() == "off"
    assert not controller.overlay.layers


def test_response_worker_coalesces_and_never_publishes_obsolete_radius(scene):
    dashboard, controls, controller = scene
    jobs = []

    def submit(*args, **kwargs):
        future = Future()
        future.set_running_or_notify_cancel()
        jobs.append((future, args, kwargs))
        return future

    controller.executor = SimpleNamespace(submit=submit, shutdown=lambda **_: None)
    controls.response.setCurrentIndex(1)
    assert len(jobs) == 1
    controls.radius.setValue(30)
    controls.radius.setValue(40)
    assert len(jobs) == 1
    jobs[0][0].set_result("old-radius")
    controller._poll()
    assert "response" not in controller.overlay.layers
    assert len(jobs) == 2 and jobs[1][1][3] == 40
    jobs[1][0].set_result("current-radius")
    controller._poll()
    assert controller.overlay.layers["response"] == "current-radius"
    assert not controller.poll_timer.isActive()
    controls.response.setCurrentIndex(0)
    assert "response" not in controller.overlay.layers
    assert controller.pending is None


def test_unchanged_cached_response_does_not_rewrite_its_widget_caption(scene, monkeypatch):
    from visualizer.src.experiments.report_geometry import receive_pattern_for_request

    dashboard, controls, controller = scene
    pattern = receive_pattern_for_request(dashboard._last_request)
    controller.response_cache[(pattern, "broadside", 20, None)] = "cached surface"
    controls.response.setCurrentIndex(1)
    assert "Broadside beam" in controls.response_note.text()
    updates = []
    monkeypatch.setattr(controls.response_note, "setText", updates.append)
    for _ in range(10):
        controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert not updates
    assert controller.overlay.layers["response"] == "cached surface"
    assert controller.executor is None


def _runtime_proof(dashboard, controller):
    request = dashboard._last_request
    acquisition = request.display_frame["sensing"]["temporal"]["acquisition"]
    frame = dict(request.display_frame)
    for role in ("tx", "rx"):
        frame[f"{role}_names"] = [acquisition[f"{role}_sensor_id"]]
    frame["tx_rx_pairs"] = np.array([[0, 0]], dtype=np.int32)
    viz = controller.viz
    viz._mpc_presented_source_epoch = 3
    viz.app_state = SimpleNamespace(
        camera_mode="follow",
        tx_device_names=tuple(frame["tx_names"]),
        rx_device_names=tuple(frame["rx_names"]),
        selected_tx="all",
        selected_rx="all",
    )
    service = SimpleNamespace(
        step=request.effective_display_step,
        token=(3, request.effective_display_step, 1),
        view_model=SimpleNamespace(
            rx_positions=frame["rx_positions"].copy(),
            rx_orientations=frame["rx_orientations"].copy(),
        ),
    )
    viz.extension_services = {"experiments": service}
    current = deepcopy(frame["sensing"])

    def metadata(step, scope):
        from visualizer.src.sensing.volume_antenna import receive_metadata_signature

        current["_receive_array_signature"] = receive_metadata_signature(current)
        return current if step == service.step else None

    session = SimpleNamespace(
        source_identity=request.source_identity,
        provider=object(),
        provider_identity=lambda _: request.source_identity,
        service=SimpleNamespace(cached_dashboard_result_metadata=metadata),
    )
    panel = SimpleNamespace(_sensing_window=dashboard, _dashboard_session=session)
    viz.ui_manager = SimpleNamespace(panels={"sensing": panel})
    viz.cache_service = SimpleNamespace(
        get_frame=lambda step: frame if step == service.step else None
    )
    return service, panel


def _static_pose(scene, mode):
    from visualizer.src.experiments.report_geometry import receive_pattern_for_request

    dashboard, controls, controller = scene
    request = dashboard._last_request
    service, panel = _runtime_proof(dashboard, controller)
    pattern = receive_pattern_for_request(request)
    beam = (
        request.display_frame["sensing"]["detections"][0]["beam_index"] if mode == "beam" else None
    )
    controller.response_cache[(pattern, mode, 20, beam)] = "cached surface"
    controls.response.setCurrentIndex(controls.response.findData(mode))
    assert controller.overlay.layers["response"] == "cached surface"
    return service, panel


@pytest.mark.parametrize("mode", ["broadside", "scan"])
def test_static_array_response_survives_late_dashboard_but_report_geometry_does_not(scene, mode):
    dashboard, controls, controller = scene
    service, _panel = _static_pose(scene, mode)
    controls.phase.setCurrentIndex(2)
    assert controller.cinematic.phase == "angle"
    key = controller.response_key
    service.step += 1
    controller.sync(dashboard, service.step)
    assert controller.overlay.layers == {"response": "cached surface"}
    assert controller.response_key == key
    assert controls.phase.currentData() == "off"
    assert controller.executor is None


@pytest.mark.parametrize(
    "change",
    [
        "position",
        "orientation",
        "source",
        "dashboard",
        "epoch",
        "pair",
        "identity",
        "selection",
        "scene",
        "live_generation",
        "array",
        "carrier",
        "scan",
        "unindexed",
        "current_pair",
        "missing_proof",
    ],
)
def test_late_dashboard_response_rejects_changed_or_unproven_current_sensor(scene, change):
    dashboard, _controls, controller = scene
    service, panel = _static_pose(scene, "broadside")
    service.step += 1
    viz = controller.viz
    if change == "position":
        service.view_model.rx_positions[0, 0] += 0.001
    elif change == "orientation":
        service.view_model.rx_orientations[0, 0] += 0.001
    elif change == "source":
        panel._dashboard_session.source_identity = ("replacement source",)
    elif change == "dashboard":
        panel._sensing_window = object()
    elif change == "epoch":
        viz._mpc_presented_source_epoch += 1
    elif change == "pair":
        viz.cache_service.get_frame(service.step)["tx_rx_pairs"][0, 1] = 1
    elif change == "identity":
        viz.app_state.rx_device_names = ("another receiver",)
    elif change == "selection":
        viz.app_state.selected_rx = 1
    elif change == "live_generation":
        panel._dashboard_session.provider_identity = lambda _: ("replacement generation",)
    elif change == "unindexed":
        panel._dashboard_session.service.cached_dashboard_result_metadata = lambda *_: None
    elif change == "missing_proof":
        panel._dashboard_session.service.cached_dashboard_result_metadata = lambda *_: {}
    elif change == "current_pair":
        current = panel._dashboard_session.service.cached_dashboard_result_metadata(
            service.step, ()
        )
        current["temporal"]["acquisition"]["rx_sensor_id"] = "replacement sensor"
    elif change in {"array", "carrier", "scan"}:
        current = panel._dashboard_session.service.cached_dashboard_result_metadata(
            service.step, ()
        )
        if change == "array":
            current["rx_array"]["horizontal_spacing_wavelengths"] *= 2
        elif change == "carrier":
            current["config_resolved"]["carrier_freq_hz"] *= 2
        else:
            current["detection_filter_metadata"]["array_detection_mode"] = "none"
    else:
        service.step += 1
    controller.sync(dashboard, dashboard._last_request.effective_display_step + 1)
    assert not controller.overlay.layers
    assert controller.response_key is None


def test_selected_report_beam_never_survives_a_different_accepted_scene(scene):
    dashboard, _controls, controller = scene
    service, _panel = _static_pose(scene, "beam")
    service.step += 1
    controller.sync(dashboard, service.step)
    assert not controller.overlay.layers
    assert controller.response_key is None


def test_same_step_pair_change_clears_array_until_current_evidence_matches(scene):
    dashboard, _controls, controller = scene
    service, _panel = _static_pose(scene, "broadside")
    controller.viz.app_state.selected_rx = 1
    controller.sync(dashboard, service.step)
    assert not controller.overlay.layers


def test_cached_current_metadata_never_reads_and_rejects_stale_or_ambiguous_index(monkeypatch):
    from visualizer.src.sensing.selection import selection_scope
    from visualizer.src.services.sensing_service import SensingService

    request, _selection = report_request()
    service = SensingService(SimpleNamespace())
    rows = ((127, request.display_frame),)
    token = service._current_metadata_index_token((127,))
    service._sensing_index_token = token
    service._sensing_result_index = rows

    def no_read(*_args, **_kwargs):
        pytest.fail("Cached array proof must never perform a source read")

    monkeypatch.setattr(service, "candidate_storage_frame_ids", no_read)
    monkeypatch.setattr(service, "_indexed_sensing_results", no_read)
    scope = selection_scope(request.display_frame["sensing"])

    def get():
        return service.cached_dashboard_result_metadata(request.effective_display_step, scope)

    assert get() is request.display_frame["sensing"]
    index = service._dashboard_scene_metadata
    assert get() is request.display_frame["sensing"]
    assert service._dashboard_scene_metadata is index
    monkeypatch.setattr(service, "_current_metadata_index_token", lambda _: ("new generation",))
    assert get() is None
    monkeypatch.setattr(service, "_current_metadata_index_token", lambda _: token)
    service._sensing_result_index = rows + rows
    assert get() is None
    service._sensing_result_index = None
    assert get() is None
    assert not service._dashboard_scene_metadata
    assert service._dashboard_scene_metadata_rows is None


def test_late_response_cannot_draw_after_source_replacement(scene):
    dashboard, controls, controller = scene
    future = Future()
    future.set_running_or_notify_cancel()
    controller.executor = SimpleNamespace(submit=lambda *a, **k: future, shutdown=lambda **_: None)
    controls.response.setCurrentIndex(1)
    controls.response.setCurrentIndex(0)
    dashboard._last_request = replace(dashboard._last_request, source_identity=("new-source",))
    future.set_result("retired-source")
    controller._poll()
    assert not controller.overlay.layers
    assert not controller.poll_timer.isActive()


def test_close_retires_pending_response_without_late_publication(scene):
    _dashboard, controls, controller = scene
    future = Future()
    future.set_running_or_notify_cancel()
    controller.executor = SimpleNamespace(submit=lambda *a, **k: future, shutdown=lambda **_: None)
    controls.response.setCurrentIndex(1)
    controller.close()
    future.set_result("late")
    controller._poll()
    assert controller.pending is None
    assert not controller.response_cache
    assert not controller.overlay.layers


def test_failed_response_is_not_recomputed_for_each_frame_callback(scene):
    dashboard, controls, controller = scene
    calls = []

    def submit(*_args, **_kwargs):
        future = Future()
        future.set_exception(ValueError("unsupported response"))
        calls.append(future)
        return future

    controller.executor = SimpleNamespace(submit=submit, shutdown=lambda **_: None)
    controls.response.setCurrentIndex(1)
    controller._poll()
    for _ in range(3):
        controller.sync(dashboard, dashboard._last_request.effective_display_step)
    assert len(calls) == 1
    assert "unsupported response" in controls.response_note.text()
    assert not controller.poll_timer.isActive()


def test_native_pygfx_objects_share_scene_and_release_without_gpu():
    import pygfx as gfx

    from visualizer.src.experiments.report_geometry import (
        build_angular_cell,
        receive_pattern_for_request,
    )
    from visualizer.src.renderers.pygfx.experiments import ExperimentOverlay
    from visualizer.src.sensing.volume_antenna import response_object

    request, selection = report_request()
    renderer = SimpleNamespace(
        renderer_id="pygfx", _gfx=gfx, _scene=gfx.Scene(), request_redraw=lambda: None
    )
    overlay = ExperimentOverlay(renderer)
    overlay.angular_cell(build_angular_cell(request, selection))
    overlay.report_position([4, 5, 6])
    overlay.receive_response(
        response_object(receive_pattern_for_request(request), "beam", 20, 0.4, beam_index=20)
    )
    assert set(overlay.layers) == {"angle", "position", "response"}
    assert overlay.layers["response"].material.opacity == pytest.approx(0.4)
    assert not overlay.layers["response"].material.pick_write
    overlay.clear()
    assert not renderer._scene.children
