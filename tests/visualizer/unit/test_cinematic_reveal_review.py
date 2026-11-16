# ruff: noqa: F811  # Imported pytest fixtures are intentionally named in test arguments.
"""Independent cinematic lifecycle and actual material transition checks."""

import numpy as np
import pytest
from shiboken6 import delete

from tests.visualizer.unit.test_camera_presentation import rig  # noqa: F401
from tests.visualizer.unit.test_cinematic_reveal import cinematic  # noqa: F401


def test_fades_preserve_saved_geometry_and_basic_material_alpha(cinematic):
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    controls._play()
    session.sync(request, selection, dashboard)
    layer = session.overlay.layers["range"]
    objects = tuple(layer.iter())
    original = [getattr(node, "_reveal_base_opacity", None) for node in objects]
    session._fade_tick(session.fade_start + 0.2)
    assert session._opacity("range") == pytest.approx(0.5)
    for node, base in zip(objects, original, strict=True):
        if base is not None:
            assert node.material.opacity == pytest.approx(base * 0.5)
    session._fade_tick(session.fade_start + 0.5)
    assert session._opacity("range") == 1
    assert not session.fade.isActive()
    controls._advance()
    session.sync(request, selection, dashboard)
    session._fade_tick(session.fade_start + 0.5)
    assert session._opacity("range") == pytest.approx(0.16)
    assert session._opacity("angle") == 1
    assert tuple(layer.iter()) == objects
    controls._advance()
    session.sync(request, selection, dashboard)
    session._fade_tick(session.fade_start + 0.5)
    assert not layer.visible
    assert session.overlay.layers["position"].visible
    assert "POSITION" in controls.note.text() and "TRACK" in controls.note.text()
    assert session._opacity("position") == 1


def test_close_after_qt_controls_destroyed_is_safe(rig):
    from types import SimpleNamespace

    import pygfx
    from PySide6.QtWidgets import QCheckBox

    from tests.visualizer.unit.test_report_geometry import report_request
    from visualizer.src.experiments.cinematic_reveal import CinematicReveal
    from visualizer.src.sensing.scene_explanation import SceneExplanationControls

    camera, renderer, _ = rig
    renderer._gfx, renderer._scene = pygfx, pygfx.Scene()
    renderer._qt_window_closed = False
    renderer.set_hud_overlay = lambda *args, **kwargs: None
    renderer.clear_hud_overlay = lambda *args, **kwargs: None
    controls = SceneExplanationControls(QCheckBox())
    request, selection = report_request()
    controls.bind_context(request, selection)
    session = CinematicReveal(camera.viz, controls)
    controls._play()
    session.sync(
        request,
        selection,
        SimpleNamespace(selected_world_position=lambda: np.array([4.0, 5.0, 6.0])),
    )
    delete(controls)
    try:
        session.close()
        assert session.overlay is None
        assert not camera.timer.isActive()
    finally:
        if session.camera is not None:
            camera.status_changed.disconnect(session._camera_status)
        camera.stop()
        if session.overlay is not None:
            session.overlay.clear()


def test_takeover_restores_smoothing_options_and_stops_all_clocks(cinematic):
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    camera.smooth, camera.duration = False, 1.9
    controls._play()
    session.sync(request, selection, dashboard)
    controls._takeover()
    assert not controls.timer.isActive() and not session.fade.isActive()
    assert not camera.timer.isActive()
    assert camera.smooth is False and camera.duration == 1.9
    assert session.overlay is None


def test_replay_recaptures_current_camera_even_with_only_position_stage(cinematic):
    from dataclasses import replace

    from visualizer.src.types.camera_state import CameraState

    session, controls, camera, renderer, request, selection, dashboard = cinematic
    request = replace(request, rd_result=None, detection_stage_bundle=None)
    selection = replace(selection, cell=None)
    controls.bind_context(request, selection)
    controls._play()
    session.sync(request, selection, dashboard)
    assert session.phase == "position"
    camera.tick(camera.transition[2] + camera.duration)
    session._fade_tick(session.fade_start + 1)
    controls._advance()
    assert not controls.timer.isActive()
    user_view = CameraState((60, -50, 40), (4, 5, 6), (0, 0, 1), 50)
    camera.apply(user_view)
    controls._play()
    session.sync(request, selection, dashboard)
    assert session.phase == "position" and session.fade.isActive()
    assert camera.transition is not None
    np.testing.assert_allclose(session.original.eye, user_view.eye)
    session.restore()
    np.testing.assert_allclose(renderer.get_camera_state().eye, user_view.eye)


def test_native_pygfx_reveal_materials_render_all_stages(cinematic):
    import os

    import pygfx
    from rendercanvas.offscreen import RenderCanvas

    if os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1":
        pytest.skip("Native pygfx smoke requires ORCHAV_RUN_PYGFX_TESTS=1")
    session, controls, camera, renderer, request, selection, dashboard = cinematic
    canvas = RenderCanvas(size=(320, 240))
    native = pygfx.WgpuRenderer(canvas, pixel_ratio=1)
    images = []
    try:
        controls._play()
        for phase in ("range", "angle", "position"):
            session.sync(request, selection, dashboard)
            assert session.phase == phase
            camera.tick(camera.transition[2] + camera.duration)
            session._fade_tick(session.fade_start + 1)
            native.render(renderer._scene, renderer._camera)
            images.append(native.snapshot().copy())
            controls._advance()
        assert all(image.shape == (240, 320, 4) for image in images)
        assert not np.array_equal(images[0], images[1])
        assert not np.array_equal(images[1], images[2])
        session.clear()
        native.render(renderer._scene, renderer._camera)
        assert session.overlay is None
    finally:
        canvas.close()
