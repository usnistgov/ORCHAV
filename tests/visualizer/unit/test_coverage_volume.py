"""Real Qt coverage inspection through renderer-neutral scene operations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget

from shared.coverage.volume import ALL_TRANSMITTERS
from tests.shared.test_coverage_volume import write_volume
from visualizer.src.coverage.volume_window import open_coverage_volume
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.types.camera_state import CameraState


class Renderer:
    capabilities = RendererCapabilities(volume_rendering=True)

    def __init__(self):
        self.objects = {}
        self.legend = None
        self.camera = CameraState((20, 30, 40), (10, 20, 0), (0, 0, 1), 50)

    def ensure_object(self, obj):
        self.objects[obj.id] = obj
        return True

    def remove_object(self, name):
        self.objects.pop(name, None)
        return True

    def set_analysis_legend(self, legend):
        self.legend = legend

    def set_overview_camera(self, *args):
        self.camera = CameraState((40, 50, 60), (10, 20, 0), (0, 0, 1), 60)

    def get_camera_state(self):
        return self.camera

    def set_camera_state(self, camera):
        self.camera = camera
        return True


@pytest.fixture
def inspector(qtbot, tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.renderer = Renderer()
    parent.coverage_data = {
        "coverage_file": str(write_volume(tmp_path / "coverage.h5")),
        "tx_names": ["A", "B"],
        "tx_positions": [[10.0, 20.0, 8.0], [30.0, 40.0, 12.0]],
    }
    parent.app_state = SimpleNamespace(show_coverage=True)
    parent.ui_controller = SimpleNamespace(handle_coverage_toggled=lambda value: None)
    w = open_coverage_volume(parent)
    qtbot.waitUntil(lambda: w.model is not None, timeout=10000)
    yield w
    active = parent.coverage_volume_window
    if shiboken6.isValid(active) and not active.closed:
        active.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_loop_wrap_pause_disable_and_close(inspector):
    w = inspector
    w.loop.setChecked(True)
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    w._play()
    for _ in range(w.coordinate.count() * 2):
        before = w.coordinate.currentIndex()
        w._step()
        assert w.coordinate.currentIndex() == (before + 1) % w.coordinate.count()
        assert w.sweep.isActive()
    w._play()
    assert not w.sweep.isActive()

    w.loop.setChecked(False)
    w.coordinate.setCurrentIndex(w.coordinate.count() - 2)
    w._play()
    w._step()
    w._step()
    assert not w.sweep.isActive()
    w.loop.setChecked(True)
    w._play()
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert not w.sweep.isActive() and w.loop.isEnabled()
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    w._play()
    w.visible.setChecked(False)
    assert not w.sweep.isActive()
    w._play()
    assert not w.sweep.isActive()


@pytest.mark.parametrize("with_cell_charts", [False, True])
def test_retirement_unregisters_only_owned_plots_before_qt_deletion(
    inspector, qtbot, with_cell_charts
):
    from pyqtgraph import ViewBox

    window = inspector
    parent = window.visualizer
    comparison = window.height_comparison
    charts = window.cell_table.charts
    owned = [comparison.plot.getViewBox()]
    if with_cell_charts:
        charts.set_context(window.model, (0, 0, 0), None)
        charts.enabled.setChecked(True)
        qtbot.waitUntil(lambda: charts.canvas is not None)
        owned.extend(plot.getViewBox() for plot in charts.canvas.plots)

    unrelated = ViewBox()
    try:
        # Hiding the inspector keeps its plots available when shown again.
        window.hide()
        window.show()
        assert all(view in ViewBox.AllViews for view in owned)

        # Close permanently retires this source-bound, DeleteOnClose window.
        window.close()
        comparison.shutdown()
        charts.shutdown()
        assert all(view not in ViewBox.AllViews for view in owned)
        assert unrelated in ViewBox.AllViews
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert all(not shiboken6.isValid(view) for view in owned)
        assert all(view not in ViewBox.AllViews for view in owned)
        assert shiboken6.isValid(unrelated) and unrelated in ViewBox.AllViews

        reopened = open_coverage_volume(parent)
        qtbot.waitUntil(lambda: reopened.model is not None, timeout=10000)
        assert reopened.height_comparison.plot.getViewBox() in ViewBox.AllViews
    finally:
        unrelated.close()


def test_combined_selection_locators_and_bookmark(inspector, qtbot, tmp_path):
    w = inspector
    original = w.path.read_bytes()
    w.tx.setCurrentIndex(w.tx.findData(ALL_TRANSMITTERS))
    w.quantity.setCurrentIndex(w.quantity.findData("rss_dbm"))
    qtbot.waitUntil(lambda: w.loaded_key == (ALL_TRANSMITTERS, "rss_dbm"), timeout=10000)
    assert "All transmitters" in w.visualizer.renderer.legend.lines[0]
    labels = [
        obj.payload.text
        for key, obj in w.visualizer.renderer.objects.items()
        if "transmitter_label" in key
    ]
    assert (
        len(labels) == 2 and any("TX A" in v for v in labels) and any("TX B" in v for v in labels)
    )
    view = tmp_path / "combined.json"
    w._save_view(view)
    restored = w._load_view(view)
    qtbot.waitUntil(lambda: restored.loaded_key == (ALL_TRANSMITTERS, "rss_dbm"), timeout=10000)
    restored.tx.setCurrentText("B")
    qtbot.waitUntil(lambda: restored.loaded_key == ("B", "rss_dbm"), timeout=10000)
    assert not any(
        key.endswith(":0") or key.endswith(":1")
        for key in restored.visualizer.renderer.objects
        if "transmitter" in key
    )
    assert restored.path.read_bytes() == original


def test_readable_locator_toggle_preserves_radio_position_and_saved_values(inspector):
    w = inspector
    assert not w.readable_transmitter.isChecked()
    w.show_transmitter.setChecked(True)
    renderer = w.visualizer.renderer
    point_id = "analysis:coverage_transmitter"
    label_id = "analysis:coverage_transmitter_label"
    position = renderer.objects[point_id].payload.points.copy()
    samples = w.model.values.copy()
    recording = w.path.read_bytes()
    w.readable_transmitter.setChecked(True)
    assert renderer.objects[label_id].metadata["label_bold"]
    assert w._view_state()["readable_transmitter"]
    np.testing.assert_array_equal(renderer.objects[point_id].payload.points, position)
    w.readable_transmitter.setChecked(False)
    assert "label_background_rgba" not in renderer.objects[label_id].metadata
    np.testing.assert_array_equal(renderer.objects[point_id].payload.points, position)
    np.testing.assert_array_equal(w.model.values, samples)
    assert w.path.read_bytes() == recording


@pytest.mark.parametrize("mode,visible", [("slice", True), ("iso", True), ("mip", False)])
def test_view_roundtrip_reopens_paused_and_restores_camera(
    inspector, qtbot, tmp_path, mode, visible
):
    w = inspector
    w.tx.setCurrentText("B")
    w.quantity.setCurrentIndex(w.quantity.findData("rss_dbm"))
    qtbot.waitUntil(lambda: w.loaded_key == ("B", "rss_dbm"), timeout=10000)
    w.orientation.setCurrentIndex(w.orientation.findData(0))
    w.coordinate.setCurrentIndex(1)
    w.opacity.setValue(0.3)
    w.threshold.setValue(-77)
    w.low.setValue(-110)
    w.high.setValue(-40)
    w.rate.setValue(4)
    w.loop.setChecked(True)
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    w.mode.setCurrentIndex(w.mode.findData(mode))
    w.uniform_colour.setChecked(True)
    w.readable_transmitter.setChecked(True)
    w.visible.setChecked(visible)
    w.visualizer.renderer.camera = CameraState((80, 50, 40), (15, 0, 10), (0, 0, 1), 35)
    expected = w._view_state()
    source = w.path.read_bytes()
    path = tmp_path / "view.json"
    w._save_view(path)
    w.tx.setCurrentText("A")
    qtbot.waitUntil(lambda: w.loaded_key[0] == "A", timeout=10000)
    w._play()
    restored = w._load_view(path)
    qtbot.waitUntil(lambda: restored.model is not None, timeout=10000)
    assert restored._view_state() == expected
    assert not restored.sweep.isActive()
    assert w.closed
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(w)
    assert restored.path.read_bytes() == source


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("colormap", "unknown"),
        ("show_nonfinite", "yes"),
        ("source", ["other", 0, 0]),
        ("transmitter", "missing"),
        ("coordinate", 12345),
        ("opacity", float("nan")),
        ("rate", 0),
        ("low", 100),
        ("loop", "true"),
        ("show_transmitter", "true"),
        ("cell_edges", "true"),
        ("unavailable_edges", 1),
        ("camera", {"eye": [0, 0, 0], "lookat": [0, 0, 0], "up": [0, 0, 1], "fov": 60}),
    ],
)
def test_invalid_view_preserves_active_inspector(inspector, tmp_path, field, value):
    import json

    w = inspector
    original = w._view_state()
    changed = dict(original, **{field: value})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        w._load_view(path)
    assert not w.closed and w._view_state() == original


def test_open_inherits_ordinary_path_loss_and_bookmark_colormap(inspector, qtbot, tmp_path):
    parent = inspector.visualizer
    parent.coverage_data.update(
        metric_name="best_path_loss_db", value_min=60.123456, value_max=95.234567
    )
    parent.coverage_height_index = 0
    parent.coverage_opacity = 0.9
    w = open_coverage_volume(parent)
    qtbot.waitUntil(lambda: w.model is not None, timeout=10000)
    assert w.loaded_key == (ALL_TRANSMITTERS, "path_loss_db")
    assert w.coordinate.currentData() == 7
    assert (w.low.value(), w.high.value()) == (60.123456, 95.234567)
    assert w.colormap.currentText() == "RdYlGn_r"
    assert w.opacity.value() == 0.9
    assert parent.renderer.legend.colorbar_colormap == "RdYlGn_r"
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.threshold.setValue(80)
    assert "≤" in w.threshold_label.text()
    assert "1/" in w.status.text()  # A's 60 dB cell wins at the fixture's powers.
    w.colormap.setCurrentText("viridis")
    path = tmp_path / "loss.json"
    w._save_view(path)
    restored = w._load_view(path)
    qtbot.waitUntil(lambda: restored.model is not None, timeout=10000)
    assert restored.colormap.currentText() == "viridis"
    assert restored.threshold.value() == 80
    assert restored.loaded_key == (ALL_TRANSMITTERS, "path_loss_db")


def test_legacy_bookmark_without_colormap_stays_viridis(inspector, qtbot):
    state = inspector._view_state()
    state.pop("colormap")
    parent = inspector.visualizer
    parent.coverage_data.update(metric_name="best_path_loss_db", value_min=60, value_max=90)
    w = open_coverage_volume(parent, view=state)
    qtbot.waitUntil(lambda: w.model is not None, timeout=10000)
    assert w.colormap.currentText() == "viridis"
    assert w.loaded_key == ("A", "path_gain_db")


def test_optional_grey_fill_preserves_values_thresholds_and_bookmark(inspector, qtbot, tmp_path):
    w = inspector
    source = w.path.read_bytes()
    w.show_nonfinite.setChecked(True)
    assert ("No data", "#808080") in w.visualizer.renderer.legend.swatches
    np.testing.assert_equal(w.template.palette[:2], [[0.5, 0.5, 0.5, 1]] * 2)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert not w.show_nonfinite.isEnabled()
    with_grey = w.template.values.copy()
    w.show_nonfinite.setChecked(False)
    np.testing.assert_equal(w.template.values, with_grey)
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    w.show_nonfinite.setChecked(True)
    path = tmp_path / "grey.json"
    w._save_view(path)
    restored = w._load_view(path)
    qtbot.waitUntil(lambda: restored.model is not None, timeout=10000)
    assert restored.show_nonfinite.isChecked()
    assert restored.path.read_bytes() == source


def test_changed_source_and_recording_overwrite_are_rejected(inspector, tmp_path):
    import os

    w = inspector
    path = tmp_path / "view.json"
    w._save_view(path)
    original = w.path.read_bytes()
    with pytest.raises(ValueError, match="separate"):
        w._save_view(w.path)
    assert w.path.read_bytes() == original
    stat = w.path.stat()
    os.utime(w.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="changed"):
        w._load_view(path)
    assert not w.closed


def test_oversized_bookmark_and_changed_active_source_leave_view_open(inspector, tmp_path):
    w = inspector
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * 100001)
    with pytest.raises(ValueError, match="100 kB"):
        w._load_view(path)
    path = tmp_path / "view.json"
    w._save_view(path)
    w.visualizer.coverage_data = dict(w.source)
    with pytest.raises(ValueError, match="changed"):
        w._load_view(path)
    assert not w.closed


def test_views_reuse_samples_sweep_is_spatial_and_close_releases(qtbot, tmp_path, monkeypatch):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.renderer = Renderer()
    parent.coverage_data = {
        "coverage_file": str(write_volume(tmp_path / "coverage.h5")),
        "tx_names": ["A", "B"],
    }
    parent.app_state = SimpleNamespace(show_coverage=True)
    toggles = []
    parent.ui_controller = SimpleNamespace(handle_coverage_toggled=toggles.append)
    w = open_coverage_volume(parent)
    qtbot.waitUntil(lambda: w.model is not None, timeout=10000)

    def no_read(*args):
        raise AssertionError("A display change reloaded the coverage file")

    monkeypatch.setattr("visualizer.src.coverage.volume_window.load_coverage_volume", no_read)
    values = w.model.values.copy()
    for axis in range(3):
        w.orientation.setCurrentIndex(w.orientation.findData(axis))
        w.coordinate.setCurrentIndex(0)
        w._play()
        w.rate.setValue(5)
        assert w.sweep.interval() == 200
        assert w.sweep.isActive()
        w.rate.setValue(0.2)
        assert w.sweep.interval() == 5000
        w._step()
        assert w.sweep.isActive()
        obj = parent.renderer.objects[w.OBJECT_ID]
        assert obj.payload.plane[axis] == 1
        assert obj.payload.plane[3] == -w.model.axes[axis][1]
        for _ in range(w.coordinate.count()):
            w._step()
        assert not w.sweep.isActive()
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert w.rate.isEnabled()
    assert not w.sweep.isActive()
    w.threshold.setValue(-80)
    assert np.count_nonzero(parent.renderer.objects[w.OBJECT_ID].payload.values) == 1
    w.mode.setCurrentIndex(w.mode.findData("mip"))
    assert "Maximum" in parent.renderer.legend.lines[0]
    w.opacity.setValue(0.25)
    assert parent.renderer.objects[w.OBJECT_ID].payload.opacity == 0.25
    np.testing.assert_equal(w.model.values, values)
    w.close()
    assert toggles == [False, True]
    assert not parent.renderer.objects
    assert not w.sweep.isActive()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(w)


def test_recorded_transmitter_locator_follows_selection_and_visibility(inspector, qtbot):
    from visualizer.src.coverage.volume_sites import SITE_OBJECT_IDS

    w = inspector
    renderer = w.visualizer.renderer
    np.testing.assert_array_equal(
        renderer.objects[SITE_OBJECT_IDS[0]].payload.points, [[10.0, 20.0, 8.0]]
    )
    assert "TX A" in renderer.objects[SITE_OBJECT_IDS[2]].payload.text
    leader = renderer.objects[SITE_OBJECT_IDS[1]].payload.points
    np.testing.assert_array_equal(leader[0], [10.0, 20.0, 8.0])
    assert leader[1, 2] > w.model.axes[2][-1]
    w.tx.setCurrentText("B")
    qtbot.waitUntil(lambda: w.loaded_key[0] == "B", timeout=10000)
    np.testing.assert_array_equal(
        renderer.objects[SITE_OBJECT_IDS[0]].payload.points, [[30.0, 40.0, 12.0]]
    )
    assert "TX B" in renderer.objects[SITE_OBJECT_IDS[2]].payload.text
    w.show_transmitter.setChecked(False)
    assert not any(name in renderer.objects for name in SITE_OBJECT_IDS)
    assert w._view_state()["show_transmitter"] is False
    w.show_transmitter.setChecked(True)
    assert all(name in renderer.objects for name in SITE_OBJECT_IDS)
    w.close()
    assert not any(name in renderer.objects for name in SITE_OBJECT_IDS)


@pytest.mark.parametrize(
    "positions", [None, [[0, 0]], [[0, 0, float("nan")]], [[0, 0, 0], [1, 2, 3]]]
)
def test_missing_or_invalid_transmitter_position_is_not_invented(positions):
    from visualizer.src.coverage.volume_sites import transmitter_objects

    source = {"tx_names": ["A"], "tx_positions": positions}
    axes = (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    assert transmitter_objects(source, "A", axes) == []


def test_cell_edges_are_optional_and_distinguish_zero_from_unavailable(inspector):
    w = inspector
    renderer = w.visualizer.renderer
    usable_id = f"{w.EDGE_PREFIX}:available"
    unavailable_id = f"{w.EDGE_PREFIX}:unavailable"
    assert not w.cell_edges.isChecked()
    assert not w.unavailable_edges.isEnabled()
    assert not any(name.startswith(w.EDGE_PREFIX) for name in renderer.objects)
    w.coordinate.setCurrentIndex(0)
    w.cell_edges.setChecked(True)
    assert w.unavailable_edges.isEnabled()
    assert usable_id in renderer.objects and unavailable_id not in renderer.objects
    lines = renderer.objects[usable_id].payload
    endpoints = lines.points[lines.lines]
    # The bottom edge belongs only to the known zero-contribution cell. It
    # must remain visible although that cell has -inf gain and no finite fill.
    zero_edge = np.array([[10.0, -20.0, 7.0], [12.0, -20.0, 7.0]])
    assert np.any(np.all(endpoints == zero_edge, axis=(1, 2)))
    assert "19 usable saved cells" in w.status.toolTip()
    w.unavailable_edges.setChecked(True)
    assert unavailable_id in renderer.objects
    assert len(renderer.objects[unavailable_id].payload.lines) == 4 * 4
    assert "1 unavailable cells" in w.status.toolTip()
    w.cell_edges.setChecked(False)
    assert not w.unavailable_edges.isEnabled()
    assert not any(name.startswith(w.EDGE_PREFIX) for name in renderer.objects)


@pytest.mark.parametrize("axis", range(3))
def test_cell_edges_use_exact_selected_slice_plane(inspector, axis):
    w = inspector
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    w.orientation.setCurrentIndex(w.orientation.findData(axis))
    w.coordinate.setCurrentIndex(0)
    objects = w.visualizer.renderer.objects
    points = np.concatenate(
        [obj.payload.points for name, obj in objects.items() if name.startswith(w.EDGE_PREFIX)]
    )
    np.testing.assert_array_equal(points[:, axis], w.model.axes[axis][0])
    for dimension in range(3):
        if dimension != axis:
            assert points[:, dimension].min() == pytest.approx(
                w.model.axes[dimension][0] - w.model.spacing[dimension] / 2
            )
            assert points[:, dimension].max() == pytest.approx(
                w.model.axes[dimension][-1] + w.model.spacing[dimension] / 2
            )


def test_edges_preserve_source_and_fill_and_ignore_threshold(inspector, monkeypatch):
    w = inspector
    renderer = w.visualizer.renderer
    source = w.path.read_bytes()
    values = w.model.values.copy()

    def no_read(*args):
        raise AssertionError("An edge display change reloaded the coverage recording")

    monkeypatch.setattr("visualizer.src.coverage.volume_window.load_coverage_volume", no_read)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.threshold.setValue(-80)
    assert "1/59 usable cells ≥ -80" in w.status.text()
    original_fill = renderer.objects[w.OBJECT_ID].payload.values.copy()
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    np.testing.assert_array_equal(renderer.objects[w.OBJECT_ID].payload.values, original_fill)
    edge_points = {
        name: obj.payload.points.copy()
        for name, obj in renderer.objects.items()
        if name.startswith(w.EDGE_PREFIX)
    }
    w.threshold.setValue(-100)
    assert "58/59 usable cells ≥ -100" in w.status.text()
    assert not np.array_equal(renderer.objects[w.OBJECT_ID].payload.values, original_fill)
    for name, points in edge_points.items():
        np.testing.assert_array_equal(renderer.objects[name].payload.points, points)
    w.mode.setCurrentIndex(w.mode.findData("mip"))
    for name, points in edge_points.items():
        np.testing.assert_array_equal(renderer.objects[name].payload.points, points)
    np.testing.assert_equal(w.model.values, values)
    assert w.path.read_bytes() == source


def test_cell_edges_release_on_hidden_source_change_and_close(inspector, qtbot):
    w = inspector
    renderer = w.visualizer.renderer
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    w.coordinate.setCurrentIndex(0)
    assert f"{w.EDGE_PREFIX}:unavailable" in renderer.objects
    w.visible.setChecked(False)
    assert not any(name.startswith(w.EDGE_PREFIX) for name in renderer.objects)
    w.visible.setChecked(True)
    assert f"{w.EDGE_PREFIX}:available" in renderer.objects
    w.tx.setCurrentText("B")
    assert not any(name.startswith(w.EDGE_PREFIX) for name in renderer.objects)
    qtbot.waitUntil(lambda: w.loaded_key[0] == "B", timeout=10000)
    assert f"{w.EDGE_PREFIX}:available" in renderer.objects
    assert f"{w.EDGE_PREFIX}:unavailable" not in renderer.objects
    w.close()
    assert not any(name.startswith(w.EDGE_PREFIX) for name in renderer.objects)


def test_legacy_bookmarks_default_to_no_cell_edges(inspector, tmp_path, qtbot):
    import json

    w = inspector
    state = w._view_state()
    del state["cell_edges"]
    del state["unavailable_edges"]
    path = tmp_path / "legacy-view.json"
    path.write_text(json.dumps(state))
    w.cell_edges.setChecked(True)
    restored = w._load_view(path)
    qtbot.waitUntil(lambda: restored.model is not None, timeout=10000)
    assert not restored.cell_edges.isChecked()
    assert not restored.unavailable_edges.isChecked()
    assert not any(
        name.startswith(restored.EDGE_PREFIX) for name in restored.visualizer.renderer.objects
    )


def test_edge_budget_explanation_replaces_old_edges(inspector, monkeypatch):
    w = inspector
    w.cell_edges.setChecked(True)
    assert f"{w.EDGE_PREFIX}:available" in w.visualizer.renderer.objects
    monkeypatch.setattr("visualizer.src.utils.cell_edges.MAX_CELL_EDGE_SEGMENTS", 1)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert not any(name.startswith(w.EDGE_PREFIX) for name in w.visualizer.renderer.objects)
    assert "Cell edges disabled" in w.status.text()
    assert "no cells were subsampled" in w.status.toolTip()


def test_material_changes_reuse_edges_and_repair_missing_object(inspector, monkeypatch):
    from visualizer.src.coverage import volume_window

    w = inspector
    renderer = w.visualizer.renderer
    w.cell_edges.setChecked(True)
    edge_id = f"{w.EDGE_PREFIX}:available"
    original = renderer.objects[edge_id]
    builder = Mock(wraps=volume_window.grid_edge_objects)
    monkeypatch.setattr(volume_window, "grid_edge_objects", builder)
    remove = Mock(wraps=renderer.remove_object)
    monkeypatch.setattr(renderer, "remove_object", remove)
    w.opacity.setValue(0.5)
    w.low.setValue(w.low.value() - 10)
    w.high.setValue(w.high.value() + 10)
    w.uniform_colour.setChecked(True)
    assert renderer.objects[edge_id] is original
    assert not builder.called
    assert edge_id not in [call.args[0] for call in remove.call_args_list]
    del renderer.objects[edge_id]
    w._draw()
    assert renderer.objects[edge_id] is original
    assert not builder.called


def test_slice_and_selected_threshold_rebuild_edges(inspector):
    w = inspector
    renderer = w.visualizer.renderer
    edge_id = f"{w.EDGE_PREFIX}:available"
    w.cell_edges.setChecked(True)
    original = renderer.objects[edge_id]
    w.coordinate.setCurrentIndex(0)
    assert renderer.objects[edge_id] is not original
    np.testing.assert_array_equal(
        renderer.objects[edge_id].payload.points[:, 2], w.model.axes[2][0]
    )
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.threshold_edges.setChecked(True)
    w.threshold.setValue(-80)
    original = renderer.objects[edge_id]
    assert "1 above-threshold saved cells" in w.status.toolTip()
    w.opacity.setValue(0.5)
    assert renderer.objects[edge_id] is original
    w.threshold.setValue(-100)
    assert renderer.objects[edge_id] is not original
    assert "58 above-threshold saved cells" in w.status.toolTip()


def test_edge_upload_failure_clears_cache_and_identical_retry_repairs(inspector, monkeypatch):
    w = inspector
    renderer = w.visualizer.renderer
    w.cell_edges.setChecked(True)
    edge_id = f"{w.EDGE_PREFIX}:available"
    ensure = renderer.ensure_object
    monkeypatch.setattr(
        renderer, "ensure_object", lambda obj: False if obj.id == edge_id else ensure(obj)
    )
    w._draw()
    assert edge_id not in renderer.objects
    assert w.edge_model is None
    assert "could not display coverage cell edges" in w.status.text()
    monkeypatch.setattr(renderer, "ensure_object", ensure)
    w._draw()
    assert edge_id in renderer.objects
    assert w.edge_model is w.model


def test_hidden_edges_retry_failed_native_removal(inspector, monkeypatch):
    w = inspector
    renderer = w.visualizer.renderer
    w.cell_edges.setChecked(True)
    edge_id = f"{w.EDGE_PREFIX}:available"
    remove = renderer.remove_object
    monkeypatch.setattr(
        renderer, "remove_object", lambda name: False if name == edge_id else remove(name)
    )
    w.visible.setChecked(False)
    assert edge_id in renderer.objects
    monkeypatch.setattr(renderer, "remove_object", remove)
    w._draw()
    assert edge_id not in renderer.objects


def test_unlit_iso_changes_display_only_and_preserves_binary_support(inspector):
    w = inspector
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    renderer = w.visualizer.renderer
    before = renderer.objects[w.OBJECT_ID].payload
    assert w.uniform_colour.isChecked() and before.lighting is False
    w.uniform_colour.setChecked(False)
    after = renderer.objects[w.OBJECT_ID].payload
    assert after.lighting is True
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.palette, before.palette)
    assert after.threshold == before.threshold and after.limits == before.limits
    w.uniform_colour.setChecked(True)
    assert renderer.objects[w.OBJECT_ID].payload.lighting is False


@pytest.mark.parametrize("uniform", [True, False, None])
def test_saved_lighting_choice_survives_new_default(inspector, qtbot, uniform):
    w = inspector
    view = w._view_state()
    if uniform is None:
        view.pop("uniform_colour")  # Older saved views used surface lighting.
    else:
        view["uniform_colour"] = uniform
    restored = open_coverage_volume(w.visualizer, view=view)
    qtbot.waitUntil(lambda: restored.model is not None, timeout=10000)
    assert restored.uniform_colour.isChecked() is bool(uniform)
    restored.mode.setCurrentIndex(restored.mode.findData("iso"))
    assert restored.visualizer.renderer.objects[w.OBJECT_ID].payload.lighting is not bool(uniform)


def test_direct_pick_synchronizes_raw_value_and_three_slices(inspector):
    w = inspector
    w._select_volume_cell(w.OBJECT_ID, (2, 0, 0), "mip")
    assert w.selected_indices == (2, 0, 0)
    assert "Ray-maximum hit" in w.cell_note.toolTip()
    assert "-60 dB" in w.cell_note.text()
    assert len(w.cell_note.text().splitlines()) == 2
    assert "Transmitter: A" in w.cell_note.toolTip()
    assert "X 15, Y -18.5, Z 7 m" in w.cell_note.toolTip()
    assert set(w.slice_view.views) == {"xy", "xz", "yz"}
    # Switching the viewed normal must preserve the selected source cell.
    for axis in (0, 1, 2):
        w.orientation.setCurrentIndex(w.orientation.findData(axis))
        assert w.coordinate.currentIndex() == w.selected_indices[axis]
    w._orient(coordinate=15)
    assert w.selected_indices[2] == 2
    assert w.coordinate.currentIndex() == 2


def test_coverage_slice_can_inspect_unavailable_without_fabricating_miss(inspector):
    w = inspector
    w._select_volume_cell(w.OBJECT_ID, (2, 0, 0), "slice")
    w._select_slice("xy", 13, -18.5)
    assert w.selected_indices == (1, 0, 0)
    assert "unavailable (NaN)" in w.cell_note.toolTip()
    assert "historical reason not recorded" in w.cell_note.toolTip()
    assert "no sensing detection outcome or trial" in w.cell_note.toolTip()


def test_source_switch_retires_old_pick_semantics(inspector, qtbot):
    w = inspector
    w._select_volume_cell(w.OBJECT_ID, (2, 0, 0), "mip")
    w.tx.setCurrentText("B")
    assert "Loading" in w.cell_note.text()
    qtbot.waitUntil(lambda: w.loaded_key == ("B", "path_gain_db"), timeout=10000)
    assert "Ray-maximum" not in w.cell_note.text()
    assert "Transmitter: B" in w.cell_note.toolTip()
