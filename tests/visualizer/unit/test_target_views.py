"""Auxiliary target cameras must share data while keeping input and work local."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx as gfx
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow, QWidget

from visualizer.src.panels.camera_panel import CameraControlPanel
from visualizer.src.renderers.pygfx.target_views import (
    TargetViewManager,
    target_choices,
    without_main_view_annotations,
)


class _Canvas(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.kwargs = kwargs
        self.draw_requests = 0
        self.draw_callback = None
        self.force_draw_calls = 0
        self.close_called = False

    def request_draw(self, callback=None):
        self.draw_requests += 1
        if callback is not None:
            self.draw_callback = callback

    def close(self):
        self.close_called = True
        return super().close()

    def force_draw(self):
        self.force_draw_calls += 1
        self.draw_callback()


class _Renderer:
    def __init__(self, canvas, **kwargs):
        self.canvas = canvas
        self.kwargs = kwargs
        self.render = Mock()


@pytest.fixture
def scene(qtbot):
    window = QMainWindow()
    window.resize(1100, 700)
    window.setCentralWidget(QWidget(window))
    window.target_entries = [
        {"object_id": "target:a", "name": "First car"},
        {"object_id": "target:b", "name": "Second car"},
    ]
    objects = {
        name: gfx.Mesh(gfx.box_geometry(4, 2, 1), gfx.MeshBasicMaterial())
        for name in ("target:a::mesh", "target:b::mesh")
    }
    objects["target:b::mesh"].local.position = (30, 20, 1)
    main_camera = gfx.PerspectiveCamera()
    backend = SimpleNamespace(
        visualizer=window,
        _objects=objects,
        _kinds={},
        _scene=object(),
        _camera=main_camera,
        _WgpuWidget=_Canvas,
        _gfx=SimpleNamespace(
            WgpuRenderer=_Renderer,
            PerspectiveCamera=gfx.PerspectiveCamera,
            OrbitController=gfx.OrbitController,
        ),
        _initialized=True,
        _frame_update_paused=False,
    )
    manager = TargetViewManager(backend)
    qtbot.addWidget(window)
    window.show()
    yield manager, backend
    manager.dispose()


def _draw(view):
    view.manager.next_draw_at = 0
    view._draw()


def _dispatch(manager):
    manager.timer.stop()
    manager.next_draw_at = 0
    manager._dispatch()


def _assert_camera_state(camera, expected):
    actual = camera.get_state()
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_equal(actual[key], expected[key])


def test_target_choice_uses_stable_identity_and_updates_display_name(scene):
    manager, backend = scene
    view = manager.add()
    assert view.target_combo.currentData() == "target:a::mesh"
    backend.visualizer.target_entries[0]["name"] = "Renamed"
    view.refresh_targets()
    assert view.target_combo.currentData() == "target:a::mesh"
    assert view.target_combo.currentText() == "Renamed"
    assert target_choices(backend)[0] == ("target:a::mesh", "Renamed")


def test_missing_target_does_not_silently_switch_to_another_actor(scene):
    manager, backend = scene
    view = manager.add()
    actor = backend._objects.pop("target:a::mesh")
    view.refresh_targets()
    assert view.target_combo.currentIndex() == -1
    backend.visualizer.target_entries[1]["name"] = "Changed list"
    view.refresh_targets()
    assert view.target_combo.currentIndex() == -1
    backend._objects["target:a::mesh"] = actor
    view.refresh_targets()
    assert view.target_combo.currentData() == "target:a::mesh"


@pytest.mark.parametrize("kind", ["tx", "rx"])
def test_actor_view_follows_nodes_without_targets_or_another_frame_reader(scene, kind):
    manager, backend = scene
    backend.visualizer.target_entries = []
    setattr(backend.visualizer, f"{kind}_entries", [{}])
    backend.visualizer.app_state = SimpleNamespace(**{f"{kind}_labels": ["Roof station"]})
    name = f"node:{kind}_0::marker"
    node = gfx.Mesh(gfx.box_geometry(1, 1, 1), gfx.MeshBasicMaterial())
    backend._objects[name] = node
    view = manager.add()
    assert view.windowTitle() == "Actor view 1"
    assert view.target_combo.currentData() == name
    assert "Roof station" in view.target_combo.currentText()
    eye = np.array(view.camera.local.position)
    node.local.position = (9, 5, 2)
    _draw(view)
    np.testing.assert_allclose(view.camera.local.position, eye + (9, 5, 2))
    np.testing.assert_allclose(view.controller.target, (9, 5, 2))


def test_follow_preserves_auxiliary_offset_and_never_moves_main_camera(scene):
    manager, backend = scene
    view = manager.add()
    initial_main = backend._camera.get_state()
    position = np.asarray(view.camera.local.position).copy()
    backend._objects["target:a::mesh"].local.position = (8, 3, 2)
    _draw(view)
    np.testing.assert_allclose(view.camera.local.position, position + (8, 3, 2))
    np.testing.assert_allclose(view.controller.target, (8, 3, 2))
    _assert_camera_state(backend._camera, initial_main)
    view.renderer.render.assert_called_once_with(backend._scene, view.camera)
    assert view.renderer.kwargs["enable_events"] is False
    assert view.render_count == manager.render_count == 1


def test_switching_target_refits_only_auxiliary_camera(scene):
    manager, backend = scene
    view = manager.add()
    main_state = backend._camera.get_state()
    view.target_combo.setCurrentIndex(1)
    np.testing.assert_allclose(view.controller.target, (30, 20, 1))
    _assert_camera_state(backend._camera, main_state)


def test_auxiliary_mouse_orbit_does_not_change_main_camera(scene):
    manager, backend = scene
    view = manager.add()
    before = np.array(view.camera.local.position)
    main_state = backend._camera.get_state()
    QTest.mousePress(view.canvas, Qt.LeftButton, pos=QPoint(80, 80))
    QTest.mouseMove(view.canvas, QPoint(120, 100))
    QTest.mouseRelease(view.canvas, Qt.LeftButton, pos=QPoint(120, 100))
    assert not np.allclose(view.camera.local.position, before)
    _assert_camera_state(backend._camera, main_state)


@pytest.mark.parametrize("hide_content", [False, True])
def test_hidden_dock_or_canvas_never_draws(scene, hide_content):
    manager, _backend = scene
    view = manager.add()
    (view.canvas if hide_content else view).hide()
    manager.scene_presented()
    _draw(view)
    assert not manager.timer.isActive()
    view.renderer.render.assert_not_called()


def test_requests_coalesce_and_draw_budget_increases_for_expensive_scenes(scene):
    manager, _backend = scene
    manager.set_refresh_policy("main_priority")
    view = manager.add()
    for _ in range(30):
        manager.scene_presented()
    manager.timer.stop()
    before = view.canvas.draw_requests
    manager._dispatch()
    manager.schedule()
    assert view.canvas.draw_requests == before + 1
    assert not manager.timer.isActive()
    manager.record_draw(0.1, succeeded=True)
    assert manager.next_draw_at - time.perf_counter() > 0.5


def test_refresh_controls_share_policy_and_new_views_inherit_it(scene):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    assert manager.refresh_policy == "main_priority"
    first.refresh_combo.setCurrentIndex(first.refresh_combo.findData("smooth"))
    assert manager.refresh_policy == "smooth"
    assert second.refresh_combo.currentData() == "smooth"
    third = manager.add()
    assert third.refresh_combo.currentData() == "smooth"
    manager.set_refresh_policy("main_priority")
    assert all(view.refresh_combo.currentData() == "main_priority" for view in manager.views)
    assert first.canvas.kwargs["update_mode"] == "ondemand"
    assert first.canvas.kwargs["min_fps"] == 0
    assert first.canvas.kwargs["max_fps"] == 60
    with pytest.raises(ValueError, match="refresh policy"):
        manager.set_refresh_policy("unknown")
    assert manager.refresh_policy == "main_priority"


def test_policy_cadence_is_measured_from_render_start(scene, monkeypatch):
    manager, _backend = scene
    manager.set_refresh_policy("smooth")
    monkeypatch.setattr("visualizer.src.renderers.pygfx.target_views.time.perf_counter", lambda: 10)
    manager.record_draw(0.002, succeeded=True)
    assert manager.next_draw_at == pytest.approx(10 + 1 / 30 - 0.002)
    manager.record_draw(0.1, succeeded=True)
    assert manager.next_draw_at == pytest.approx(10 + 0.1 * 0.7 / 0.3)


def test_clean_callbacks_skip_work_but_physical_resize_still_renders(scene):
    manager, _backend = scene
    view = manager.add()
    view.canvas.get_physical_size = Mock(return_value=(400, 300))
    _draw(view)
    _draw(view)
    assert view.render_count == 1
    view.canvas.get_physical_size.return_value = (800, 600)
    _draw(view)
    assert view.render_count == 2


def test_interaction_shortens_active_timer_but_retains_a_duty_limit(scene, monkeypatch):
    manager, _backend = scene
    view = manager.add()
    manager.set_refresh_policy("main_priority")
    monkeypatch.setattr("visualizer.src.renderers.pygfx.target_views.time.perf_counter", lambda: 10)
    manager.record_draw(0.1, succeeded=True)
    manager.timer.stop()
    manager.schedule()
    original_wait = manager.timer.remainingTime()
    manager.request_interaction(view)
    assert manager.next_draw_at == pytest.approx(10.1)
    assert 0 < manager.timer.remainingTime() < original_wait
    manager.record_draw(0.001, succeeded=True)
    manager.request_interaction(view)
    assert manager.next_draw_at == pytest.approx(10 + 1 / 30 - 0.001)


def test_repeated_interaction_coalesces_without_another_in_flight_request(scene):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    _dispatch(manager)
    requested = first.canvas.draw_requests + second.canvas.draw_requests
    for _ in range(20):
        manager.request_interaction(second)
        manager._dispatch()
    assert first._requested and not second._requested
    assert first.canvas.draw_requests + second.canvas.draw_requests == requested
    assert not manager.timer.isActive()


def test_interaction_priority_services_peers_between_repeated_preferred_draws(scene):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    manager.request_interaction(second)
    _dispatch(manager)
    assert second.canvas.force_draw_calls == 1 and not first._requested
    manager.scene_presented()
    manager.request_interaction(second)
    _dispatch(manager)
    assert first._requested and not second._requested


@pytest.mark.parametrize("state", ["hidden", "closed", "failed"])
def test_unavailable_interaction_view_does_not_block_other_views(scene, state):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    manager.request_interaction(second)
    if state == "hidden":
        second.hide()
    elif state == "closed":
        second.close()
    else:
        second.failed = True
    manager.request_interaction(second)
    _dispatch(manager)
    assert first._requested and not second._requested


def test_interaction_cannot_render_during_frame_mutation(scene):
    manager, backend = scene
    view = manager.add()
    backend._frame_update_paused = True
    manager.request_interaction(view)
    manager._dispatch()
    _draw(view)
    view.renderer.render.assert_not_called()
    assert view.dirty
    assert not manager.timer.isActive()
    backend._frame_update_paused = False
    manager.scene_presented()
    _draw(view)
    view.renderer.render.assert_called_once()


def test_internal_refit_has_no_interaction_priority_but_target_selection_does(scene):
    manager, _backend = scene
    view = manager.add()
    assert manager._preferred_view is None
    view._last_center = None
    _draw(view)
    assert manager._preferred_view is None
    view.target_combo.setCurrentIndex(1)
    assert manager._preferred_view is view


def test_smooth_draw_accounts_full_presentation_cost_without_changing_submit_metric(
    scene, monkeypatch
):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    manager.set_refresh_policy("smooth")
    clock = [10.0]
    monkeypatch.setattr(
        "visualizer.src.renderers.pygfx.target_views.time.perf_counter", lambda: clock[0]
    )

    def render(*_args):
        clock[0] += 0.002

    def force_draw():
        first.canvas.draw_callback()
        assert not manager.timer.isActive()
        assert not second._requested
        manager._dispatch()  # A nested native paint/event turn cannot dispatch another view.
        assert not second._requested
        clock[0] += 0.018

    first.renderer.render.side_effect = render
    first.canvas.force_draw = force_draw
    _dispatch(manager)
    assert manager.render_count == 1
    assert manager.render_submit_ms_total == pytest.approx(2)
    assert manager._last_draw_duration == pytest.approx(0.020)
    assert manager.next_draw_at == pytest.approx(10.020 + 0.020 * 0.7 / 0.3)
    assert not manager._synchronous_draw and not first._requested
    assert manager.timer.isActive()


def test_failed_synchronous_presentation_latches_failure_and_releases_scheduler(scene):
    manager, _backend = scene
    first = manager.add()
    second = manager.add()
    manager.set_refresh_policy("smooth")
    first.canvas.force_draw = Mock(side_effect=RuntimeError("Qt paint failed"))
    _dispatch(manager)
    assert first.failed and first.render_failures == manager.render_failures == 1
    assert not first._requested and not manager._synchronous_draw
    assert manager.timer.isActive()
    _dispatch(manager)
    assert second.render_count == 1
    first.canvas.force_draw.assert_called_once()


def test_early_timer_callback_does_not_start_synchronous_render(scene):
    manager, _backend = scene
    view = manager.add()
    manager.set_refresh_policy("smooth")
    manager.next_draw_at = time.perf_counter() + 1
    manager.timer.stop()
    manager._dispatch()
    assert view.canvas.force_draw_calls == 0
    assert manager.timer.isActive()


def test_main_annotations_restore_even_when_auxiliary_render_fails(scene):
    manager, backend = scene
    label = SimpleNamespace(visible=True)
    hidden_label = SimpleNamespace(visible=False)
    guide = SimpleNamespace(visible=True)
    hidden_guide = SimpleNamespace(visible=False)
    gizmo = SimpleNamespace(visible=True)
    backend._objects.update(label=label, hidden_label=hidden_label)
    backend._kinds.update(label="text", hidden_label="text")
    backend._transform_gizmo = gizmo
    backend._scene_guide_labels = lambda: (guide, hidden_guide)
    with pytest.raises(ValueError), without_main_view_annotations(backend):
        assert not label.visible and not gizmo.visible
        assert not guide.visible and not hidden_guide.visible
        raise ValueError("render failed")
    assert label.visible and gizmo.visible and not hidden_label.visible
    assert guide.visible and not hidden_guide.visible
    with without_main_view_annotations(backend, guides_only=True):
        assert label.visible and gizmo.visible
        assert not guide.visible and not hidden_guide.visible
    assert guide.visible and not hidden_guide.visible
    view = manager.add()
    view.renderer.render.side_effect = RuntimeError("shader failure")
    _draw(view)
    assert view.failed
    assert view.render_count == manager.render_count == 0
    assert view.render_failures == manager.render_failures == 1
    manager.scene_presented()
    assert not manager.timer.isActive()


def test_close_releases_view_and_dispose_retires_timer_callback(scene):
    manager, _backend = scene
    view = manager.add()
    renderer = view.renderer
    view.close()
    _draw(view)
    assert manager.views == []
    assert view.renderer is None
    assert not manager.timer.isActive()
    renderer.render.assert_not_called()
    manager.dispose()
    manager.schedule()
    assert manager._disposed


def test_no_target_allocates_no_view_or_draw_timer(scene):
    manager, backend = scene
    backend.visualizer.target_entries.clear()
    assert manager.add() is None
    assert manager.views == []
    assert not manager.timer.isActive()


def test_renderer_creation_failure_closes_unregistered_canvas(scene, monkeypatch):
    manager, backend = scene
    monkeypatch.setattr(backend._gfx, "WgpuRenderer", Mock(side_effect=RuntimeError("device lost")))

    with pytest.raises(RuntimeError, match="device lost"):
        manager.add()

    assert manager.views == []
    assert not manager.timer.isActive()
    canvases = backend.visualizer.findChildren(_Canvas)
    assert len(canvases) == 1
    assert canvases[0].close_called
    assert not canvases[0].isVisible()


def test_add_action_reports_native_creation_failure_without_escaping_qt_slot(scene):
    _manager, backend = scene
    backend.open_target_view = Mock(side_effect=RuntimeError("device lost"))
    backend.visualizer.renderer = backend
    panel = SimpleNamespace(parent=backend.visualizer)

    CameraControlPanel._open_target_view(panel)

    assert backend.visualizer.statusBar().currentMessage() == "Actor view unavailable: device lost"
