"""Production Qt controls and renderer-neutral selected evidence integration."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from tests.shared.test_sensing_volume_inspection import write_fixture
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.sensing.volume_evidence import trial_objects
from visualizer.src.sensing.volume_window import SensingVolumeWindow
from visualizer.src.types.camera_state import CameraState


@pytest.fixture
def opened(qtbot, tmp_path, monkeypatch):
    reader = write_fixture(tmp_path / "wall")
    objects = {}
    legends = []
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = reader.scene_path
    parent.open_scenario = lambda *a, **kw: SimpleNamespace(succeeded=True)
    camera = CameraState((20, 30, 40), (10, 20, 30), (0, 0, 1), 60)
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        ensure_object=lambda o: objects.__setitem__(o.id, o) is None,
        remove_object=lambda n: objects.pop(n, None) is not None,
        request_redraw=lambda: None,
        set_analysis_legend=legends.append,
        set_overview_camera=lambda *a: None,
        get_camera_state=lambda: camera,
        set_camera_state=lambda value: None,
    )
    window = SensingVolumeWindow(parent, reader.path)
    window.show()
    qtbot.waitUntil(lambda: window.evaluation_controls.ready, timeout=20000)
    yield window, objects, legends
    if isValid(window):
        window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(window) and not objects


def test_categories_selection_and_display_invariance(opened, monkeypatch):
    w, objects, _ = opened
    c = w.inspection_controls
    original = w.evaluation_snapshot
    calls = []
    evaluate = w.evaluation.evaluate
    monkeypatch.setattr(
        w.evaluation, "evaluate", lambda *a, **kw: (calls.append(kw), evaluate(*a, **kw))[1]
    )
    w.quantity.setCurrentIndex(w.quantity.findData("details"))
    w.mode.setCurrentIndex(w.mode.findData("both"))
    np.testing.assert_array_equal(
        objects["sensing:outcome_volume"].payload.values[0, 0], [1, 2, 3, 4, 0, 0]
    )
    assert not w.mode.model().item(w.mode.findData("mip")).isEnabled()
    c.orientation.setCurrentIndex(c.orientation.findData("xz"))
    w.opacity.setValue(0.25)
    w.uniform_colour.setChecked(True)
    assert w.workflow.context()["view"]["uniform_colour"] is True
    w.evaluation_controls.error_max.setValue(9)
    c.focus.setCurrentIndex(c.focus.findData("extras"))
    c.category_filters[3].setChecked(False)
    assert not calls and w.evaluation_snapshot is original
    assert (objects["sensing:outcome_volume"].payload.values > 0).sum() == 1
    assert not w.inspect.isChecked()
    w.workflow.action("next")
    assert not w.inspect.isChecked()
    w.workflow.select_volume_cell("sensing:outcome_volume", (0, 0, 0), "iso")
    w.workflow.action("next")
    assert w._cell_key() == (0, 0, 0, 1)
    assert len(w.workflow.rows) == 3 and c.table.rowCount() == 3
    c.table.selectRow(1)
    c.vectors.setChecked(True)
    c.through.setChecked(True)
    point = objects["sensing:volume:report:1"]
    np.testing.assert_array_equal(point.payload.points, [[0, 0, -5]])
    assert point.metadata["depth_test"] is False
    segment = objects["sensing:volume:error_vector:1"]
    np.testing.assert_array_equal(segment.payload.points, [[12, 20, 30], [0, 0, -5]])
    assert "not propagation paths" in c.report_note.text()
    assert not w.open_native.isEnabled() and "not included" in w.open_native.toolTip()
    w.evaluation_controls.position_gate.setValue(100)
    assert len(calls) == 1 and w.evaluation_snapshot["tp"] == 3


def test_new_volume_is_unlit_and_lighting_never_changes_outcomes(opened):
    w, objects, _ = opened
    assert w.uniform_colour.isChecked()
    w.mode.setCurrentIndex(w.mode.findData("both"))
    before = objects["sensing:outcome_volume"].payload
    snapshot = w.evaluation_snapshot
    assert before.lighting is False
    w.uniform_colour.setChecked(False)
    after = objects["sensing:outcome_volume"].payload
    assert after.lighting is True
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.palette, before.palette)
    assert w.evaluation_snapshot is snapshot


@pytest.mark.parametrize("uniform", [True, False, None])
def test_sensing_saved_view_preserves_lighting(opened, qtbot, uniform):
    w, _, _ = opened
    view = w.workflow.context()["view"]
    if uniform is None:
        view.pop("uniform_colour")
    else:
        view["uniform_colour"] = uniform
    restored = SensingVolumeWindow(w.visualizer, w.reader.path, view_state=view)
    try:
        qtbot.waitUntil(lambda: restored.evaluation_controls.ready, timeout=20000)
        assert restored.uniform_colour.isChecked() is bool(uniform)
    finally:
        restored.close()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize(
    "orientation,normal,axis", [("xy", "z", 0), ("xz", "y", 1), ("yz", "x", 2)]
)
def test_linked_slice_spatial_sweep_and_context(opened, orientation, normal, axis):
    w, objects, _ = opened
    c = w.inspection_controls
    c.orientation.setCurrentIndex(c.orientation.findData(orientation))
    c.slice_plot.setChecked(True)
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    w.workflow.select_slice(
        orientation, 12 if orientation != "yz" else 20, 20 if orientation == "xy" else 30
    )
    state = w.workflow.context()
    assert state["index_zyx"] == list(w._cell_key()[1:])
    payload = objects["sensing:outcome_volume"].payload
    assert payload.plane[:3] == tuple(1 if n == normal else 0 for n in ("x", "y", "z"))
    old = w._cell_key()
    w.workflow.action("play")
    assert w.workflow.sweep.isActive()
    w.workflow.action("play")
    assert not w.workflow.sweep.isActive()
    w.workflow.step_plane(1)
    new = w._cell_key()
    # Only the normal grid index can change, and no native worker starts.
    assert all(
        a == b for i, (a, b) in enumerate(zip(old, new)) if i != {"z": 1, "y": 2, "x": 3}[normal]
    )
    assert w.native_pending is None


def test_missing_position_never_creates_a_world_point(opened):
    w, _, _ = opened
    row = dict(detection_id="missing", position_m=None, matched=True, error_m=None)
    objects = trial_objects(w.reader, (0, 0, 0, 0), [row], selected="missing", vectors=True)
    assert not any("report:" in o.id or "error_vector" in o.id for o in objects)


def test_stale_native_completion_does_not_open_a_different_trial(opened):
    w, _, _ = opened
    opened_sources = []
    w.visualizer.open_scenario = lambda *a, **kw: opened_sources.append(a)
    future = Future()
    future.set_result(w.reader.path.parent / "native")
    w.native_pending = future
    w.native_pending_key = (0, 0, 0, 0)
    w.x.setCurrentIndex(1)
    w._poll()
    assert not opened_sources and w.native_pending is None


def test_context_restores_exact_semantic_selection(opened, qtbot):
    w, _, _ = opened
    w.altitude.setCurrentIndex(0)
    w.workflow.select_slice("xy", 12, 20)
    w.quantity.setCurrentIndex(w.quantity.findData("extras"))
    w.inspection_controls.through.setChecked(True)
    state = w.workflow.context()
    assert state["trial_id"] == "t1" and state["position_m"] == [12, 20, 30]
    parent = w.visualizer
    w.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    restored = SensingVolumeWindow(parent, state["volume"], context=state)
    qtbot.waitUntil(lambda: restored.evaluation_controls.ready, timeout=20000)
    assert restored._cell_key() == (0, 0, 0, 1) and restored.quantity.currentData() == "extras"
    assert restored.inspection_controls.through.isChecked()
    restored.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_depth_override_reaches_real_pygfx_material_and_is_reversible():
    import pygfx

    from visualizer.src.model import RenderObject
    from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
    from visualizer.src.types.render_payloads import PointCloudPayload

    native = pygfx.Points(
        pygfx.Geometry(positions=np.zeros((1, 3), np.float32)), pygfx.PointsMaterial()
    )
    renderer = SimpleNamespace(_objects={"report": native})
    for enabled in (False, True):
        obj = RenderObject(
            id="report",
            payload=PointCloudPayload(points=np.zeros((1, 3))),
            metadata={
                "depth_test": enabled,
                "depth_write": enabled,
                "depth_compare": "<=",
                "render_order": 100,
            },
        )
        PygfxGeometryMixin._sync_render_object_metadata(renderer, obj)
        assert native.material.depth_test is enabled and native.material.depth_write is enabled
        assert native.material.depth_compare == "<=" and native.render_order == 100


def test_context_uses_heading_vector_when_display_names_repeat(opened, qtbot):
    import json

    from shared.sensing.volume.inspection import context_snapshot
    from shared.sensing.volume.reader import SensingVolume

    w, _, _ = opened
    path = w.reader.path
    data = json.loads(path.read_text())
    data["headings"].append(dict(name="+X", velocity_m_s=[0, 1, 0]))
    path.write_text(json.dumps(data))
    reader = SensingVolume(path)
    view = dict(w.workflow.context()["view"], x=10.0, y=20.0, altitude=30.0)
    state = context_snapshot(reader, 1, (1, 0, 0, 0), view)
    parent = w.visualizer
    w.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    restored = SensingVolumeWindow(parent, path, context=state)
    qtbot.waitUntil(lambda: restored.evaluation_controls.ready, timeout=20000)
    assert restored.heading.currentIndex() == 1
    assert restored.evaluation_snapshot["measured"] == 0
    restored.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize(
    "mode,label",
    [("iso", "Visible ISO surface"), ("mip", "Ray-maximum hit"), ("slice", "Visible slice")],
)
def test_direct_volume_hit_updates_all_three_source_slices(opened, mode, label):
    w, objects, _ = opened
    original = w.reader.outcome_grid(0).copy()
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), mode)
    assert w._cell_key() == (0, 0, 0, 1)
    assert label in w.cell_note.text()
    assert "Centre: X 12, Y 20, Z 30 m" in w.cell_note.text()
    assert "Raw source value:" in w.cell_note.text()
    assert w.inspect.isChecked() and not w.inspect_section.isHidden()
    assert not w.inspection_controls.slice_plot.isChecked()
    w.inspection_controls.slice_plot.setChecked(True)
    assert set(w.inspection_controls.slice_view.views) == {"xy", "xz", "yz"}
    for orientation, view in w.inspection_controls.slice_view.views.items():
        h, v = {"xy": (12, 20), "xz": (12, 30), "yz": (20, 30)}[orientation]
        assert view.vertical.value() == h and view.horizontal.value() == v
    np.testing.assert_array_equal(w.reader.outcome_grid(0), original)
    assert w.native_pending is None


def test_three_slices_keep_misses_and_unavailable_distinct_from_hit_only_3d(opened):
    w, objects, _ = opened
    w.mode.setCurrentIndex(w.mode.findData("hit"))
    w.workflow.select_volume_cell("sensing:outcome_volume", (0, 0, 0), "iso")
    w.inspection_controls.slice_plot.setChecked(True)
    view = w.inspection_controls.slice_view.views["xy"]
    # Fixture cell 2 is an acquired miss, while cell 4 is unavailable.
    assert view.image.image[0, 2, 3] == 255
    assert view.image.image[0, 4, 3] == 0
    w.workflow.select_slice("xy", 18, 20)
    assert "No measured outcome" in w.cell_note.text()
    assert "unavailable" in w.cell_note.text()
    assert not w.open_native.isEnabled()
