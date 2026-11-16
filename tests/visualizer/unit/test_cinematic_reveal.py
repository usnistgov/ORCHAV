"""Report reveals use guided camera transitions and preserve scene ownership."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pygfx
import pytest
from PySide6.QtWidgets import QCheckBox

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from tests.visualizer.unit.test_report_geometry import report_request
from visualizer.src.experiments.cinematic_reveal import CinematicReveal
from visualizer.src.sensing.scene_explanation import SceneExplanationControls


@pytest.fixture
def cinematic(qtbot, rig):  # noqa: F811 - imported pytest fixture
    tools, renderer, _ = rig
    renderer._gfx = pygfx
    renderer._scene = pygfx.Scene()
    renderer._qt_window_closed = False
    renderer.set_hud_overlay = lambda *args, **kwargs: None
    renderer.clear_hud_overlay = lambda *args, **kwargs: None
    controls = SceneExplanationControls(QCheckBox())
    qtbot.addWidget(controls)
    request, selection = report_request()
    controls.bind_context(request, selection)
    dashboard = SimpleNamespace(selected_world_position=lambda: np.array([4.0, 5.0, 6.0]))
    session = CinematicReveal(tools.viz, controls)
    yield session, controls, tools, renderer, request, selection, dashboard
    session.close()


def test_stage_camera_pause_takeover_and_restore(cinematic):
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    original = renderer.get_camera_state()
    assert not hasattr(controls, "cinematic")
    controls._play()
    session.sync(request, selection, dashboard)
    assert not controls.range_shell.isChecked()
    assert camera.transition is not None and camera.timer.isActive()
    assert len(session.stages) == 3 and session.fade.isActive()
    controls._pause()
    assert (
        not controls.timer.isActive()
        and not camera.timer.isActive()
        and not session.fade.isActive()
    )
    controls._pause()
    assert controls.timer.isActive() and camera.timer.isActive()
    camera.tick(camera.transition[2] + camera.duration)
    session._fade_tick(session.fade_start + 1)
    controls._advance()
    session.sync(request, selection, dashboard)
    assert session.phase == "angle"
    assert "ANGLE" in controls.note.text()
    camera.interrupt()  # native mouse/keyboard takeover uses the same stop path
    assert controls.phase.currentData() == "off" and not controls.timer.isActive()
    assert not session.fade.isActive()
    assert controls.restore_camera.isEnabled()
    session.restore()
    np.testing.assert_allclose(renderer.get_camera_state().eye, original.eye)
    assert camera.viz.app_state.camera_mode == "follow"


def test_missing_stages_skip_and_playback_disables(cinematic):
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    request = replace(request, rd_result=None, detection_stage_bundle=None)
    selection = replace(selection, cell=None)
    controls.bind_context(request, selection)
    controls._play()
    session.sync(request, selection, dashboard)
    assert controls.phase.currentData() == "position"
    assert set(session.stages) == {"position"}
    controls.set_playing(True)
    session.clear()
    assert not controls.play.isEnabled() and not controls.phase.isEnabled()
    assert not session.fade.isActive() and not controls.timer.isActive()
    controls.set_playing(False)
    assert controls.play.isEnabled() and controls.phase.isEnabled()
    controls._play()
    session.sync(request, selection, dashboard)
    assert session.phase == "position" and session.fade.isActive()


def test_context_change_cancels_and_new_source_discards_restore(cinematic):
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    controls._play()
    session.sync(request, selection, dashboard)
    controls.bind_context(request, None)
    session.sync(request, None, dashboard)
    assert session.phase is None and not session.fade.isActive()
    request = replace(request, source_identity=("replacement",))
    controls.bind_context(request, selection)
    session.sync(request, selection, dashboard)
    assert session.original is None and not controls.restore_camera.isEnabled()
