"""Saved sensing stories share the real camera/reveal without a dashboard window."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx
import pytest

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from tests.visualizer.unit.test_report_geometry import report_request
from visualizer.src.sensing.runtime import state_value
from visualizer.src.services import demo_sensing_story as story_module


@pytest.fixture
def story(rig, monkeypatch):  # noqa: F811
    camera, renderer, _ = rig
    viz = camera.viz
    renderer._gfx = pygfx
    renderer._scene = pygfx.Scene()
    renderer._qt_window_closed = False
    renderer.set_hud_overlay = Mock()
    renderer.clear_hud_overlay = Mock()
    viz.set_state(camera_mode="overview")
    viz.provider = SimpleNamespace(info=SimpleNamespace(generation_id="one", frame_set_id="run"))
    viz.scenario = SimpleNamespace(data_mode="files")
    viz.schedule_update = Mock()
    viz.animation_step = 5
    viz.get_available_animation_steps = lambda: [5, 7, 11]
    viz.cancel_pending_slider_scrub = Mock()
    viz.ui_controller.handle_frame_input_changed = Mock()
    request, _ = report_request()
    identity = story_module.DashboardSession.provider_identity(viz.provider)
    request = replace(request, source_identity=identity)
    viz.extension_services = {
        "sensing": SimpleNamespace(prepared_scene=SimpleNamespace(records=[request.display_frame])),
        "experiments": SimpleNamespace(step=5),
    }

    def toggle():
        viz.animation_running = not viz.animation_running

    viz.animation_controller = SimpleNamespace(toggle_animation=Mock(side_effect=toggle))
    real_identity = story_module.DashboardSession.provider_identity

    class Session:
        provider_identity = staticmethod(real_identity)

        def __init__(self, provider, **kwargs):
            self.pending = None
            self.options = kwargs
            self.keys = []
            self.closed = False

        def request(self, key):
            self.keys.append(key)
            self.pending = Future()

        def close(self):
            self.closed = True

    monkeypatch.setattr(story_module, "DashboardSession", Session)
    monkeypatch.setattr(story_module, "visualizer_frame_provider", lambda owner: owner.provider)
    monkeypatch.setattr(story_module, "perf_counter", lambda: 100.0)
    value = story_module.DemoSensingStory(viz, interval=1.0)
    yield value, viz, camera, renderer, request
    value.close()


def _begin_reveal(story):
    value, viz, _camera, _renderer, request = story
    value.start(0.0)
    value.tick(1.0)
    value.session.pending.set_result(request)
    value.tick(1.1)
    assert not viz.animation_running
    assert value.reveal.phase is None  # Wait for renderer-accepted scene geometry.
    viz.extension_services["experiments"].step = request.effective_display_step
    value.tick(1.2)
    return value


def test_saved_report_freezes_exact_scene_reveals_and_resumes_without_mutation(story):
    value, viz, camera, renderer, request = story
    before = deepcopy(request.display_frame["sensing"]["detections"])
    original_camera = renderer.get_camera_state()
    original_settings = {
        key: state_value(viz.app_state, key) for key in story_module._OVERLAY_SETTINGS
    }
    _begin_reveal(story)
    assert value.session.options["cache_entry_limit"] == 2
    assert value.session.keys == [(5, "live", "primary", True, "auto", ("range_doppler",))]
    viz.ui_controller.handle_frame_input_changed.assert_called_once_with(2)
    assert value.reveal.phase == "range" and "RANGE" in value.caption
    assert not value.controls.isVisible()
    assert "angle" in value.reveal.stages
    assert camera.transition is not None
    camera.tick(camera.transition[2] + camera.duration)
    assert not np.allclose(renderer.get_camera_state().eye, original_camera.eye)
    value.tick(value._next)
    assert value.reveal.phase == "angle" and "ANGLE" in value.caption
    while value._phase == "reveal":
        value.tick(value._next)
    assert viz.animation_running
    np.testing.assert_allclose(renderer.get_camera_state().eye, original_camera.eye)
    assert request.display_frame["sensing"]["detections"] == before
    session = value.session
    value.close()
    assert session.closed
    assert {key: state_value(viz.app_state, key) for key in original_settings} == original_settings


def test_manual_camera_interrupt_keeps_playback_and_inhibits_further_reveals(story):
    value, viz, camera, renderer, _request = story
    camera.smooth, camera.duration = False, 1.7
    history = list(camera.history)
    _begin_reveal(story)
    camera.tick(camera.transition[2] + 0.4)
    visible = renderer.get_camera_state()
    value.pause_camera()
    assert viz.animation_running and value.reveal.phase is None
    assert camera.transition is None
    assert (camera.smooth, camera.duration) == (False, 1.7)
    assert camera.history == history
    np.testing.assert_allclose(renderer.get_camera_state().eye, visible.eye)
    count = len(value.session.keys)
    value.tick(999.0)
    assert len(value.session.keys) == count
    value.resume_camera()
    value.tick(101.0)
    assert len(value.session.keys) == count + 1


def test_manual_input_while_loading_never_seeks_the_late_result(story):
    value, viz, _camera, _renderer, request = story
    value.start(0)
    value.tick(1)
    future = value.session.pending
    value.pause_camera()
    future.set_result(request)
    value.tick(1000)
    viz.ui_controller.handle_frame_input_changed.assert_not_called()
    assert viz.animation_running


@pytest.mark.parametrize("outcome", [None, "error", "empty", "wrong source"])
def test_missing_or_failed_saved_report_keeps_playback(story, outcome):
    value, viz, _camera, _renderer, request = story
    value.start(0)
    value.tick(1)
    if outcome == "error":
        value.session.pending.set_exception(ValueError("missing saved product"))
    else:
        if outcome == "empty":
            request = replace(
                request,
                display_frame={
                    **request.display_frame,
                    "sensing": {**request.display_frame["sensing"], "detections": []},
                },
            )
        elif outcome == "wrong source":
            request = replace(request, source_identity=("old",))
        value.session.pending.set_result(None if outcome is None else request)
    value.tick(1.1)
    assert viz.animation_running and value._phase == "playback"
    viz.ui_controller.handle_frame_input_changed.assert_not_called()


def test_scene_replacement_retires_geometry_without_restoring_old_settings(story):
    value, viz, camera, _renderer, _request = story
    _begin_reveal(story)
    session = value.session
    viz.provider = SimpleNamespace(info=SimpleNamespace(generation_id="two", frame_set_id="next"))
    viz.renderer = SimpleNamespace(clear_hud_overlay=Mock())
    viz.set_state(extension_state={"sensing": {"show_detections": False, "scene_mode": "tracks"}})
    # A replacement scene can already have installed its own camera motion.
    replacement_transition = camera.transition = ["replacement"]
    camera.duration = 3.7
    camera.history = ["replacement view"]
    value.tick(2.0)
    assert session.closed and value._phase == "closed"
    assert not state_value(viz.app_state, "show_detections")
    assert state_value(viz.app_state, "scene_mode") == "tracks"
    assert camera.transition is replacement_transition
    assert camera.duration == 3.7 and camera.history == ["replacement view"]
    viz.renderer.clear_hud_overlay.assert_not_called()


def test_capability_uses_only_ready_local_sensing_metadata(story):
    value, viz, _camera, _renderer, _request = story
    assert value.available(viz)
    viz.extension_services["sensing"].preparation_pending = True
    assert not value.available(viz)
    viz.extension_services["sensing"].preparation_pending = False
    viz.scenario.data_mode = "live_grpc"
    assert not value.available(viz)
