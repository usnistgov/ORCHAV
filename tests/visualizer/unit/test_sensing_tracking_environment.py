"""World-frame geometry and Qt lifecycle checks for tracking scene context."""

from concurrent.futures import Future
from threading import Event

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt

from visualizer.src.model import RenderObjectState, Transform
from visualizer.src.sensing import tracking_environment as module
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.update_request import SensingUpdateRequest
from visualizer.src.types.render_payloads import MeshPayload


def square(x=0):
    return MeshPayload(
        vertices=np.array([[x, 0, 0], [x + 2, 0, 0], [x + 2, 3, 0], [x, 3, 0]]),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]),
    )


def test_projection_uses_world_pose_once_and_removes_coplanar_diagonal():
    transform = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 5], [0, 0, 0, 1]])
    edges = module.project_scene_edges(((square(), transform),), Event())
    assert edges.shape == (4, 2, 2)
    np.testing.assert_array_equal(edges.min(axis=(0, 1)), [7, 20])
    np.testing.assert_array_equal(edges.max(axis=(0, 1)), [10, 22])
    assert np.all(np.count_nonzero(np.diff(edges, axis=1), axis=2) == 1)
    assert not edges.flags.writeable


def test_projection_welds_exact_seams_and_keeps_crease_and_open_edges():
    # Two triangles share an edge through duplicate vertices, with a crease.
    mesh = MeshPayload(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0], [0, 2, 0], [2, 2, 1]]),
        triangles=np.array([[0, 1, 2], [3, 5, 4]]),
    )
    edges = module.project_scene_edges(((mesh, np.eye(4)),), Event())
    assert len(edges) == 5


def test_projection_cancellation_invalid_geometry_and_budget(monkeypatch):
    cancelled = Event()
    cancelled.set()
    assert not len(module.project_scene_edges(((square(), np.eye(4)),), cancelled))
    bad = np.eye(4)
    bad[0, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        module.project_scene_edges(((square(), bad),), Event())
    monkeypatch.setattr(module, "MAX_PROJECTION_TRIANGLES", 1)
    with pytest.raises(ValueError, match="triangle background limit"):
        module.project_scene_edges(((square(), np.eye(4)),), Event())


def test_optional_background_preserves_track_range_and_survives_view_switch(qtbot):
    window = SensingWindow()
    window.show()
    layer = window.environment
    window._set_view_mode("tracking")
    assert not layer.toggle.isHidden()
    assert not layer.toggle.isEnabled()
    window.tracking_plot.plot([0, 1], [0, 1])
    window.tracking_plot.enableAutoRange()
    qtbot.wait(100)
    layer.set_scene("run", [{"mesh": square(1000)}])
    assert layer._future is None  # No geometry work until requested.
    layer.toggle.setChecked(True)
    qtbot.waitUntil(lambda: layer._ready)
    assert layer.item.isVisible()
    # Aspect-locked viewport padding depends on the responsive plot dimensions;
    # the background a kilometre away must never enter the actual data bounds.
    view = window.tracking_plot.getViewBox()
    assert layer.item not in view.addedItems
    bounds = np.asarray(view.childrenBounds())
    assert bounds.min() > -1 and bounds.max() < 2
    assert layer.item.acceptedMouseButtons() == Qt.MouseButton.NoButton
    assert layer.item.zValue() < 0
    layer.opacity.setValue(20)
    assert layer.item.opacity() == 0.2
    path = layer.item.path()
    window._set_view_mode("analysis")
    assert layer.toggle.isHidden()
    window._set_view_mode("tracking")
    assert layer.item.path() == path
    layer.toggle.setChecked(False)
    assert not layer.item.isVisible()
    layer.toggle.setChecked(True)
    assert layer._future is None  # Reuse, not rebuild.
    window.reset_recording()
    assert layer.item.path().isEmpty()
    assert not layer.item.isVisible()
    assert not layer.toggle.isEnabled()
    assert not layer.opacity.isEnabled()
    window.close()


def test_source_payload_and_pose_changes_invalidate_and_late_completion_is_ignored(qtbot):
    window = SensingWindow()
    layer = window.environment
    state = RenderObjectState(id="building", payload=square())
    entries = [{"mesh": state}]
    layer.set_scene("a", entries)
    first_key = layer._key
    obsolete = Future()
    obsolete.set_running_or_notify_cancel()
    layer._future = obsolete
    layer.set_scene("b", entries)
    assert layer._key != first_key
    obsolete.set_result(np.array([[[99, 99], [100, 100]]]))
    layer._poll()
    assert layer.item.path().isEmpty()
    key = layer._key
    state.world_transform = Transform.from_translation([8, 9, 0])
    layer.set_scene("b", entries)
    assert layer._key != key
    key = layer._key
    state.replace_payload(square(1))
    layer.set_scene("b", entries)
    assert layer._key != key
    layer.set_scene("b", [])
    assert not layer.toggle.isEnabled()
    window.close()


def test_projection_failure_explained_and_deferred_close_releases_worker(qtbot, monkeypatch):
    monkeypatch.setattr(module, "MAX_PROJECTION_SEGMENTS", 1)
    for _ in range(3):
        window = SensingWindow()
        layer = window.environment
        layer.set_scene("scene", [{"mesh": square()}])
        layer.toggle.setChecked(True)
        qtbot.waitUntil(lambda: "edge background limit" in layer.note.text())
        assert not layer.item.isVisible()
        assert not layer.toggle.isEnabled()
        window.close()
        assert layer._closed and layer._future is None and layer._executor is None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_close_with_running_worker_cancels_without_touching_deleted_qt(qtbot, monkeypatch):
    started, finish = Event(), Event()

    def delayed(meshes, cancelled):
        started.set()
        finish.wait(2)
        assert cancelled.is_set()
        return np.empty((0, 2, 2))

    monkeypatch.setattr(module, "project_scene_edges", delayed)
    window = SensingWindow()
    layer = window.environment
    layer.set_scene("scene", [{"mesh": square()}])
    layer.toggle.setChecked(True)
    qtbot.waitUntil(started.is_set)
    future = layer._future
    window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not future.done()
    finish.set()
    assert not len(future.result(timeout=2))


def test_raw_spectrum_absence_explained_and_source_change_resets_selection():
    window = SensingWindow()
    window._set_view_mode("analysis")
    descriptor = {
        "kind": "range_doppler",
        "variant": "primary",
        "processing": {"slow_time_mean_removal_applied": True},
    }
    sensing = {"products": {"rd": descriptor}, "config_resolved": {}}
    request = SensingUpdateRequest(display_frame={"sensing": sensing}, state_frame={}, state_step=0)
    window._update_product_options(sensing, request)
    assert window._variant_box.isHidden()
    raw_index = window._variant_box.findData("raw")
    assert not window._variant_box.model().item(raw_index).isEnabled()
    assert "was not saved" in window._variant_box.model().item(raw_index).toolTip()
    sensing["products"]["raw"] = {**descriptor, "variant": "raw"}
    window._update_product_options(sensing, request)
    assert not window._variant_box.isHidden()
    assert window._variant_box.model().item(raw_index).isEnabled()
    assert window._variant_box.itemText(raw_index) == "Before clutter suppression"
    assert window._variant_box.currentText() == "After clutter suppression"
    descriptor["processing"] = {}
    window._update_product_options(sensing, request)
    assert window._variant_box.itemText(raw_index) == "Alternate saved spectrum"
    assert window._variant_box.currentText() == "Saved spectrum"
    window._variant_box.setCurrentIndex(raw_index)
    window.reset_recording()
    assert window.product_variant() == "primary"
    assert window._variant_box.isHidden()
    window.close()


def test_no_clutter_suppression_has_one_spectrum_and_restores_comparison():
    window = SensingWindow()
    window._set_view_mode("analysis")
    descriptor = {
        "kind": "range_doppler",
        "variant": "primary",
        "processing": {"slow_time_mean_removal_applied": False},
    }
    sensing = {
        "products": {"rd": descriptor, "raw": {**descriptor, "variant": "raw"}},
        "config_resolved": {},
    }
    request = SensingUpdateRequest(display_frame={"sensing": sensing}, state_frame={}, state_step=0)
    raw_index = window._variant_box.findData("raw")
    window._variant_box.setCurrentIndex(raw_index)
    window._update_product_options(sensing, request)
    assert window.product_variant() == "primary"
    assert window._variant_box.currentText() == "Spectrum (no clutter suppression)"
    assert window._variant_box.view().isRowHidden(raw_index)
    assert not window._variant_box.model().item(raw_index).isEnabled()
    assert window._variant_box.isHidden()
    descriptor["processing"]["slow_time_mean_removal_applied"] = True
    window._update_product_options(sensing, request)
    assert not window._variant_box.view().isRowHidden(raw_index)
    assert window._variant_box.model().item(raw_index).isEnabled()
    assert not window._variant_box.isHidden()
    assert window._variant_box.currentText() == "After clutter suppression"
    window.close()
