"""Unified Coverage panel: source parity, lifecycle and allocation boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget

from shared.coverage.hdf5 import CoverageHDF5Reader
from tests.shared.test_coverage_volume import write_volume
from tests.visualizer.unit.test_coverage_volume import Renderer
from visualizer.src.controllers.coverage_controller import CoverageController
from visualizer.src.panels.coverage_panel import CoverageMapPanel
from visualizer.src.services.coverage_service import CoverageService


@pytest.fixture
def panel_view(qtbot, tmp_path):
    viz = QWidget()
    qtbot.addWidget(viz)
    viz.renderer = Renderer()
    viz.schedule_update = Mock()
    viz.app_state = SimpleNamespace(show_coverage=False, coverage_height_index=0)
    viz.set_state = lambda **values: vars(viz.app_state).update(values)
    viz.coverage_height_index = 0
    viz.coverage_opacity = 0.8
    viz.coverage_heights = [7, 11, 15]
    metrics = [
        "rss_dbm/A",
        "rss_dbm/B",
        "best_rss_dbm",
        "sinr_db/A",
        "sinr_db",
        "serving_tx",
        "path_loss_db/B",
        "rss_w/A",
    ]
    viz.coverage_data = dict(
        coverage_file=str(write_volume(tmp_path / "coverage.h5")),
        tx_names=["A", "B"],
        tx_positions=[[10, 20, 8], [30, 40, 12]],
        metric_name=metrics[0],
        available_metrics=metrics,
        heights=viz.coverage_heights,
        grid_shape=[5, 4, 3],
        grid_origin=[10, -20, 7],
        grid_spacing=[2, 3],
        value_min=-100.0,
        value_max=-50.0,
    )
    service = CoverageService()
    ui = SimpleNamespace(visualizer=viz, coverage_service=service)
    controller = CoverageController(ui)
    ui.coverage_controller = controller
    for name in dir(controller):
        if name.startswith("handle_coverage_"):
            setattr(ui, name, getattr(controller, name))
    viz.ui_controller = ui
    panel = CoverageMapPanel(viz)
    group = panel.create_panel()
    qtbot.addWidget(group)
    viz.ui_manager = SimpleNamespace(panels={"coverage": panel})
    panel.update_coverage_status(True, viz.coverage_data)
    yield viz, panel, controller
    panel.volume_view.reset()


def loaded(qtbot, panel):
    qtbot.waitUntil(
        lambda: panel.volume_view.window is not None
        and panel.volume_view.window.model is not None
        and panel.volume_view.window.pending is None
        and panel.volume_view.window.loaded_key == panel.volume_view.window.request_key,
        timeout=10000,
    )
    return panel.volume_view.window


def test_lazy_single_owner_and_reuse_for_visual_changes(panel_view, qtbot, monkeypatch):
    viz, panel, controller = panel_view
    panel.sync_coverage_view()
    assert panel.volume_view.window is None
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert w.embedded and not w.isWindow()
    assert "coverage_volume" not in panel.widgets
    assert panel._plane_analysis.isHidden()
    assert viz.app_state.show_coverage and viz.coverage_volume_active
    assert not w.poll.isActive()
    # Grid outlines retain known computed zeros, unlike unknown cells.
    # Their colour fill stays transparent (covered by scalar-template tests).
    np.testing.assert_array_equal(
        w.valid_cells, np.isfinite(w.model.values) | w.model.zero_contributions
    )
    assert w.valid_count == int(w.valid_cells.sum())
    values = w.model.values
    texture = w.template.values
    # Slice/opacity/labels and graphs must reuse the retained array, not read it again.
    monkeypatch.setattr(
        CoverageHDF5Reader, "read_metric", Mock(side_effect=AssertionError("Unexpected read"))
    )
    for index in [1, 2, 0, 2]:
        w.coordinate.setCurrentIndex(index)
        w.opacity.setValue(0.7)
        panel.sync_coverage_view()
        assert w.model.values is values
        assert w.template.values is texture
        assert viz.coverage_height_index == index
    controller.handle_coverage_toggled(False)
    assert not viz.app_state.show_coverage
    assert not viz.renderer.objects[w.OBJECT_ID].visibility.visible
    assert viz.renderer.legend is None
    assert not any("transmitter" in key for key in viz.renderer.objects)
    controller.handle_coverage_toggled(True)
    assert panel.volume_view.window is w
    assert viz.renderer.objects[w.OBJECT_ID].visibility.visible


@pytest.mark.parametrize(
    "metric", ["sinr_db", "sinr_db/A", "serving_tx", "path_loss_db/B", "rss_w/A"]
)
def test_main_metric_selection_drives_exact_volume(panel_view, qtbot, metric):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    loaded(qtbot, panel)
    controller.handle_coverage_metric_changed(metric)
    w = loaded(qtbot, panel)
    np.testing.assert_array_equal(
        w.model.values, CoverageHDF5Reader(viz.coverage_data["coverage_file"]).read_metric(metric)
    )
    assert w.model.quantity == metric
    assert w.quantity.isHidden() and w.tx.isHidden()
    if metric == "serving_tx":
        assert not w.mode.model().item(2).isEnabled()
        w.mode.setCurrentIndex(1)
        w.region_tx.setCurrentIndex(w.region_tx.findData(1))
        np.testing.assert_array_equal(w.template.values, (w.model.values == 1).astype(np.float32))
    if metric == "rss_w/A":
        assert w.low.value() > 0
        assert w.low.suffix() == " dBm"
        assert float(w.low.cleanText()) == pytest.approx(
            10 * np.log10(w.low.value()) + 30, abs=0.05
        )


def test_embedded_bookmark_restores_same_panel_and_categorical_region(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    loaded(qtbot, panel)
    controller.handle_coverage_metric_changed("serving_tx")
    w = loaded(qtbot, panel)
    w.mode.setCurrentIndex(1)
    w.region_tx.setCurrentIndex(w.region_tx.findData(1))
    path = tmp_path / "view.json"
    w._save_view(path)
    controller.handle_coverage_metric_changed("rss_dbm/A")
    loaded(qtbot, panel)
    restored = w._load_view(path)
    loaded(qtbot, panel)
    assert restored is w and panel.volume_view.window is w
    assert w.mode.currentData() == "iso" and w.region_tx.currentData() == 1
    assert w.model.quantity == "serving_tx"
    assert not w.sweep.isActive()


@pytest.mark.parametrize("heights", [[7], [7, 11, 16]])
def test_single_height_or_irregular_recording_keeps_planar_controls(panel_view, heights):
    viz, panel, controller = panel_view
    viz.coverage_data["heights"] = heights
    controller.handle_coverage_toggled(True)
    assert panel.volume_view.window is None
    assert not panel._volume_controls_active
    assert "Height-plane" in panel.volume_view.note.text()


def test_source_replacement_releases_model_and_pending_reads(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    source = viz.coverage_data
    viz.coverage_data = None
    panel.sync_coverage_view()
    assert w.closed and not w.poll.isActive() and not w.sweep.isActive()
    assert "_volume_layer" not in source
    assert not viz.coverage_volume_active
    assert not viz.renderer.objects


@pytest.mark.parametrize("mode", ["slice", "iso"])
def test_visible_controls_are_deleted_when_source_closes(panel_view, qtbot, mode):
    viz, panel, controller = panel_view
    panel.volume_view.widget.window().show()
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.mode.setCurrentIndex(w.mode.findData(mode))
    assert w.isVisible()
    # Escape must leave embedded controls alone, but source teardown must close them.
    w.reject()
    assert w.isVisible() and not w.closed
    viz.coverage_data = None
    panel.sync_coverage_view()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(w)
    assert not panel.volume_view.widget.findChildren(type(w))


def test_replacement_map_slice_uses_new_heights_and_new_samples(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    previous = loaded(qtbot, panel)
    previous.coordinate.setCurrentIndex(2)
    path = write_volume(tmp_path / "new_coverage.h5")
    with h5py.File(path, "r+") as h:
        h["grid/heights_m"][...] = [10, 20, 30]
        h["grid/origin"][2] = 10
        for index in range(3):
            h["values/path_gain_linear"][0, index, ...] = 10.0 ** (-6 - index)
    source = dict(viz.coverage_data, coverage_file=str(path), heights=[10, 20, 30])
    source.pop("_volume_layer", None)
    viz.coverage_data = source
    viz.coverage_heights = source["heights"]
    panel.update_coverage_status(True, source)
    panel.sync_coverage_view()
    current = loaded(qtbot, panel)
    assert previous.closed and current is not previous
    for index, height in enumerate([10, 20, 30]):
        current.coordinate.setCurrentIndex(index)
        panel.sync_coverage_view()
        assert current.coordinate.currentData() == height
        assert viz.coverage_height_index == index
        np.testing.assert_allclose(current.model.values[index], -60 - 10 * index, atol=1e-4)
        assert viz.renderer.objects[current.OBJECT_ID].payload.plane == (0, 0, 1, -height)


def test_slice_analysis_is_integrated_and_reuses_source(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert w.mode.findData("planar") == -1
    assert panel._plane_analysis.isHidden()
    assert panel._height_animation_group.isHidden()
    values = w.model.values
    for axis in (0, 1, 2):
        w.orientation.setCurrentIndex(w.orientation.findData(axis))
        w.smoothing.setCurrentIndex(w.smoothing.findData("linear"))
        w.contours.setChecked(True)
        assert viz.coverage_volume_active
        assert w.model.values is values
        assert w.template.values.shape[2 - axis] == 1
        obj = viz.renderer.objects[w.OBJECT_ID]
        assert obj.transform.matrix[axis, 3] == w.coordinate.currentData()
    w.smoothing.setCurrentIndex(0)
    controller.handle_coverage_height_changed(2)
    assert w.model.values is values and w.coordinate.currentIndex() == 2


def test_invalid_bookmark_does_not_change_panel(panel_view, qtbot, tmp_path):
    import json

    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    state = w._view_state()
    state.update(metric="serving_tx", quantity="serving_tx", mode="mip")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="maximum-projected"):
        w._load_view(path)
    assert panel.volume_view.window is w
    assert w.model.quantity == "rss_dbm/A"


def test_rapid_metric_changes_discard_old_result_and_coalesce_reads(panel_view, qtbot, monkeypatch):
    from threading import Event

    from visualizer.src.coverage import volume_window

    viz, panel, controller = panel_view
    started, release = Event(), Event()
    read = volume_window.load_coverage_metric_volume
    requests = []

    def delayed(path, metric):
        requests.append(metric)
        if len(requests) == 1:
            started.set()
            assert release.wait(timeout=10)
        return read(path, metric)

    monkeypatch.setattr(volume_window, "load_coverage_metric_volume", delayed)
    try:
        controller.handle_coverage_toggled(True)
        qtbot.waitUntil(started.is_set)
        controller.handle_coverage_metric_changed("sinr_db/A")
        controller.handle_coverage_metric_changed("serving_tx")
        assert requests == ["rss_dbm/A"]
    finally:
        release.set()
    w = loaded(qtbot, panel)
    assert requests == ["rss_dbm/A", "serving_tx"]
    assert w.model.quantity == "serving_tx"
    assert not w.poll.isActive()


def test_projection_reuses_slice_texture(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    values = w.template.values
    w.mode.setCurrentIndex(w.mode.findData("mip"))
    assert w.template.values is values
    w.mode.setCurrentIndex(0)
    assert w.template.values is values


def test_transmitter_changes_keep_threshold_and_colour_scale(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.threshold.setValue(-65)
    w.low.setValue(-80)
    w.high.setValue(-45)
    for metric in ("rss_dbm/B", "best_rss_dbm", "rss_dbm/A"):
        controller.handle_coverage_metric_changed(metric)
        loaded(qtbot, panel)
        assert (w.threshold.value(), w.low.value(), w.high.value()) == (-65, -80, -45)
        assert panel.widgets["coverage_tx_combo"].currentData() == metric
    controller.handle_coverage_metric_changed("sinr_db/A")
    loaded(qtbot, panel)
    w.threshold.setValue(6)
    controller.handle_coverage_metric_changed("rss_dbm/B")
    loaded(qtbot, panel)
    assert w.threshold.value() == -65
    controller.handle_coverage_metric_changed("sinr_db")
    loaded(qtbot, panel)
    assert w.threshold.value() == 6


def test_threshold_survives_view_and_transmitter_changes(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.filter_enabled.setChecked(True)
    w.threshold.setValue(-65)
    for mode in ("slice", "iso", "mip"):
        w.mode.setCurrentIndex(w.mode.findData(mode))
        controller.handle_coverage_metric_changed("rss_dbm/B")
        loaded(qtbot, panel)
        assert w.threshold.value() == -65
        assert w.filter_enabled.isChecked()
        assert w._filter_mask is not None
    assert w._view_state()["filter_enabled"] is True


def test_filters_region_and_contours_round_trip_in_place(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.filter_enabled.setChecked(True)
    w.threshold.setValue(-80)
    w.colour_mode.setCurrentIndex(1)
    w.solid_colour.setCurrentIndex(2)
    w.region_enabled.setChecked(True)
    w.region_bounds[0][1].setValue(15)
    w.smoothing.setCurrentIndex(1)
    w.contours.setChecked(True)
    state = w._view_state()
    path = tmp_path / "slice-view.json"
    w._save_view(path)
    w.filter_enabled.setChecked(False)
    w.region_enabled.setChecked(False)
    w.colour_mode.setCurrentIndex(0)
    w.smoothing.setCurrentIndex(0)
    assert w._load_view(path) is w
    restored = w._view_state()
    for key in (
        "filter_enabled",
        "threshold",
        "colour_mode",
        "solid_colour",
        "region_enabled",
        "region_bounds",
        "smoothing",
        "contours",
    ):
        assert restored[key] == state[key]
    assert "selected region" in w.status.text()


def test_compact_controls_use_automatic_colours_and_progressive_details(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert "Transmitter labels" not in w.sections
    assert w.sections["View"].isAncestorOf(w.region_enabled)
    assert w.sections["View"].isAncestorOf(w.region_bound_rows[0])
    assert w.sections["View"].isAncestorOf(w.sections["Display details"])
    assert not w.sections["Display details"].is_expanded()
    assert w.colour_mode.isHidden() and w.solid_colour.isHidden() and w.uniform_colour.isHidden()
    assert w.threshold.isHidden()
    w.filter_enabled.setChecked(True)
    assert not w.threshold.isHidden()
    values, template = w.model.values, w.template.values
    w.colour_mode.setCurrentIndex(1)  # Legacy scripted setting cannot override the main GUI.
    assert w.colour_mode.currentData() == "metric"
    assert w.template.values is template and w.model.values is values
    assert viz.renderer.legend.colorbar_range is not None
    controller.handle_coverage_metric_changed("serving_tx")
    loaded(qtbot, panel)
    assert w.filter_enabled.isHidden() and w.threshold.isHidden()
    assert viz.renderer.legend.colorbar_range is None
    assert {name for name, _ in viz.renderer.legend.swatches} == {"A", "B"}


def test_linear_threshold_edit_preserves_native_filter_and_bookmark(
    panel_view, qtbot, tmp_path, monkeypatch
):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    controller.handle_coverage_metric_changed("rss_w/A")
    w = loaded(qtbot, panel)
    values = w.model.values
    w.filter_enabled.setChecked(True)
    w.threshold.setValue(1e-12)
    monkeypatch.setattr(
        CoverageHDF5Reader, "read_metric", Mock(side_effect=AssertionError("Unexpected read"))
    )
    w.threshold.stepUp()
    expected = 1e-12 * 10**0.05
    assert w.threshold.value() == pytest.approx(expected, rel=1e-9, abs=1e-30)
    np.testing.assert_array_equal(w._filter_mask, np.isfinite(values) & (values >= expected))
    state = w._view_state()
    assert state["threshold"] == pytest.approx(expected, rel=1e-9, abs=1e-30)
    path = tmp_path / "linear.json"
    w._save_view(path)
    w.threshold.setValue(1e-10)
    w._load_view(path)
    assert w.threshold.value() == pytest.approx(expected, rel=1e-9, abs=1e-30)
    assert w.model.values is values


@pytest.mark.parametrize("width", [360, 420])
def test_compact_coverage_controls_fit_narrow_panels(panel_view, qtbot, width):
    from visualizer.src.app.theme import generate_application_stylesheet

    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    host = w.window()
    host.setStyleSheet(generate_application_stylesheet())
    host.resize(width, 1100)
    host.show()
    w.sections["Display details"].expand()
    w.contours.setChecked(True)
    w.cell_edges.setChecked(True)
    w.filter_enabled.setChecked(True)

    def controls_fit():
        for widget in (w.low, w.high, w.rate, w.loop, w.contour_count, w.unavailable_edges):
            assert widget.isVisible(), widget.accessibleName()
            child = widget
            while child is not host:
                parent = child.parentWidget()
                assert parent.rect().contains(child.geometry()), (
                    widget.accessibleName(),
                    parent,
                    parent.rect(),
                    child.geometry(),
                )
                child = parent
        assert host.width() == width

    qtbot.waitUntil(controls_fit)
    assert w.low.geometry().center().y() == w.high.geometry().center().y()
    assert w.play.geometry().center().y() == w.rate.geometry().center().y()
    w._play()
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert not w.sweep.isActive() and not w.sweep_controls.isHidden()
    assert w.cut_axis.isVisible()
    assert w.contour_controls.isHidden() and w.show_nonfinite.isHidden()
    controller.handle_coverage_metric_changed("serving_tx")
    loaded(qtbot, panel)
    assert w.scale_controls.isHidden() and w.colormap_controls.isHidden()
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    assert not w.sweep_controls.isHidden()
    assert not w.show_nonfinite.isHidden() and w.contour_controls.isHidden()


def test_coverage_legend_keeps_context_and_moves_diagnostics_to_hover(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert viz.renderer.legend.lines == ("A · Z = 7 m",)
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    w.filter_enabled.setChecked(True)
    w.threshold.setValue(-80)
    w.region_enabled.setChecked(True)
    w._select_slice("xy", 15, -17)
    assert len(viz.renderer.legend.lines) == 2
    assert "Filter ≥ -80.00 dBm" in viz.renderer.legend.lines[1]
    assert "Selected region" in viz.renderer.legend.lines[1]
    assert "unavailable cells" in w.status.toolTip()
    assert len(w.analysis_summary.text().splitlines()) == 2
    assert "denominator" in w.analysis_summary.toolTip()


def test_region_arrows_step_by_each_recorded_axis_spacing(panel_view, qtbot, monkeypatch):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    w.region_enabled.setChecked(True)
    values = w.model.values
    monkeypatch.setattr(
        CoverageHDF5Reader, "read_metric", Mock(side_effect=AssertionError("Unexpected read"))
    )
    for axis, bounds, step in zip(w.model.axes, w.region_bounds, (2, 3, 4), strict=True):
        low, high = bounds
        assert low.singleStep() == high.singleStep() == step
        low.stepUp()
        assert low.value() == pytest.approx(axis[1])
        low.stepDown()
        high.stepDown()
        assert high.value() == pytest.approx(axis[-2])
        high.stepUp()
    assert w.model.values is values


def test_single_transmitter_hides_comparison_without_reading_cells(qtbot, tmp_path, monkeypatch):
    from visualizer.src.coverage.cell_inspection import CoverageCellTable

    parent = QWidget()
    qtbot.addWidget(parent)
    table = CoverageCellTable(tmp_path / "unused.h5", (), parent, tx_count=1)
    reader = Mock(side_effect=AssertionError("Unexpected cell read"))
    monkeypatch.setattr(table, "_read", reader)
    table.request((0, 0, 0))
    assert table.enabled.isHidden() and not table.enabled.isEnabled()
    assert table.table.isHidden() and table.note.isHidden()
    assert table.pending is None and not table.timer.isActive()
    assert not table.charts.enabled.isHidden()
    table.shutdown()


def test_smoothing_picks_map_back_to_saved_normal_index(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.orientation.setCurrentIndex(w.orientation.findData(0))
    w.coordinate.setCurrentIndex(3)
    w.smoothing.setCurrentIndex(1)
    w._select_volume_cell(w.OBJECT_ID, (0, 2, 1), "slice")
    assert w.selected_indices == (3, 2, 1)
    assert "Source index XYZ: 3, 2, 1" in w.cell_note.toolTip()


def test_per_transmitter_table_reads_only_on_request_and_caches_cell(
    panel_view, qtbot, monkeypatch
):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    original = CoverageHDF5Reader.read_cell
    calls = []

    def observed(self, indices, **kwargs):
        calls.append(indices)
        return original(self, indices, **kwargs)

    monkeypatch.setattr(CoverageHDF5Reader, "read_cell", observed)
    w._draw()
    assert not calls
    w.cell_table.enabled.setChecked(True)
    assert not calls and w.cell_table.key is None
    w._select_slice("xy", 15, -17)
    qtbot.waitUntil(lambda: w.cell_table.loaded == w.cell_table.key)
    assert len(calls) == 1 and w.cell_table.table.rowCount() == 2
    w.opacity.setValue(0.45)
    w.low.setValue(-105)
    w.cell_table.enabled.setChecked(False)
    w.cell_table.enabled.setChecked(True)
    w.cell_table.charts.enabled.setChecked(True)
    qtbot.waitUntil(lambda: w.cell_table.charts.canvas is not None)
    assert len(calls) == 1


def test_region_edges_exclude_cells_outside_bounds(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.region_enabled.setChecked(True)
    w.region_bounds[0][1].setValue(15)
    w.cell_edges.setChecked(True)
    w.unavailable_edges.setChecked(True)
    for obj in w.edge_objects:
        # The selected last X centre is 15 m; its right boundary is 16 m.
        assert obj.payload.points[:, 0].max() <= 16
    assert "1 unavailable cells" in " ".join(w.edge_lines)


@pytest.mark.parametrize(
    "field,value",
    [
        ("filter_enabled", "yes"),
        ("contour_count", 30),
        ("colour_mode", "random"),
        ("region_bounds", [[1000, 2000]] * 3),
    ],
)
def test_invalid_analysis_bookmark_is_atomic(panel_view, qtbot, tmp_path, field, value):
    import json

    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    original = w._view_state()
    state = dict(
        original,
        region_enabled=True,
        region_bounds=[[float(axis.min()), float(axis.max())] for axis in w.model.axes],
    )
    state[field] = value
    path = tmp_path / "bad-analysis.json"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        w._load_view(path)
    assert w._view_state() == original


def test_serving_regions_default_to_all_and_filter_both_views(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    loaded(qtbot, panel)
    controller.handle_coverage_metric_changed("serving_tx")
    w = loaded(qtbot, panel)
    assert w.region_tx.currentText() == "All transmitters"
    assert w.region_tx.currentData() == -1
    source = w.model.values
    for mode in ("slice", "iso"):
        w.mode.setCurrentIndex(w.mode.findData(mode))
        w.region_tx.setCurrentIndex(w.region_tx.findData(-1))
        np.testing.assert_array_equal(w.template.values, np.where(source >= 0, source + 2, 0))
        assert len(viz.renderer.legend.swatches) == 2
        assert (
            sum(
                key.startswith("analysis:coverage_site:") and key.endswith(":label")
                for key in viz.renderer.objects
            )
            == 2
        )
        for index in (0, 1):
            w.region_tx.setCurrentIndex(w.region_tx.findData(index))
            np.testing.assert_array_equal(w.template.values > w.template.threshold, source == index)
            assert len(viz.renderer.legend.swatches) == 1
            assert (
                sum(
                    key.startswith("analysis:coverage_site:") and key.endswith(":label")
                    for key in viz.renderer.objects
                )
                == 2
            )
    w.region_tx.setCurrentIndex(w.region_tx.findData(-1))
    path = tmp_path / "all-transmitters.json"
    w._save_view(path)
    w.region_tx.setCurrentIndex(w.region_tx.findData(0))
    w._load_view(path)
    assert w.region_tx.currentData() == -1
    assert w.model.values is source


@pytest.mark.parametrize("version", (2, 3))
def test_existing_region_bookmarks_keep_transmitter_identity(panel_view, qtbot, tmp_path, version):
    import json

    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    loaded(qtbot, panel)
    controller.handle_coverage_metric_changed("serving_tx")
    w = loaded(qtbot, panel)
    state = w._view_state()
    state.update(schema=f"orchav.coverage-view.v{version}", mode="iso", region_tx=0)
    path = tmp_path / "older-view.json"
    path.write_text(json.dumps(state))
    w._load_view(path)
    assert w.region_tx.currentData() == 0
    assert w.region_tx.currentText() == "A"
    np.testing.assert_array_equal(w.template.values, (w.model.values == 0).astype(np.float32))


def test_volume_handover_requests_one_forced_frame_refresh(panel_view, qtbot):
    viz, panel, controller = panel_view
    submitted = []

    def refresh(step):
        submitted.append(
            (step, getattr(viz, "coverage_volume_active", False), viz.force_update_next_frame)
        )
        viz.force_update_next_frame = False

    viz._process_frame_step = refresh
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert submitted[-1] == (0, True, True)
    before = len(submitted)
    w.orientation.setCurrentIndex(w.orientation.findData(1))
    w.coordinate.setCurrentIndex(2)
    w.opacity.setValue(0.7)
    assert len(submitted) == before


def test_grey_no_data_availability_depends_on_view_not_metric(panel_view, qtbot):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert w.show_nonfinite.isEnabled()
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    assert not w.show_nonfinite.isEnabled()
    assert "3D volume" in w.show_nonfinite.toolTip()
    w.mode.setCurrentIndex(w.mode.findData("mip"))
    assert w.show_nonfinite.isEnabled()


def test_selected_cell_highlight_moves_and_restores(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert w.highlight_cell.isChecked()
    samples = w.model.values
    w._select_volume_cell(w.OBJECT_ID, (3, 2, 1), "slice")
    obj = viz.renderer.objects[w.SELECTION_PREFIX + ":outline"]
    np.testing.assert_allclose(obj.payload.points.min(axis=0), [16, -14, 11])
    np.testing.assert_allclose(obj.payload.points.max(axis=0), [18, -11, 11])
    w.mode.setCurrentIndex(w.mode.findData("iso"))
    obj = viz.renderer.objects[w.SELECTION_PREFIX + ":outline"]
    np.testing.assert_allclose(obj.payload.points.min(axis=0), [16, -14, 9])
    np.testing.assert_allclose(obj.payload.points.max(axis=0), [18, -11, 13])
    assert len(obj.payload.lines) == 12
    assert not obj.metadata["depth_test"]
    path = tmp_path / "selection.json"
    w._save_view(path)
    w.highlight_cell.setChecked(False)
    assert not any(w.SELECTION_PREFIX in key for key in viz.renderer.objects)
    w._select_slice("xy", 11, -18.5)
    w._load_view(path)
    assert w.selected_indices == (3, 2, 1) and w.highlight_cell.isChecked()
    controller.handle_coverage_toggled(False)
    assert not any(w.SELECTION_PREFIX in key for key in viz.renderer.objects)
    assert w.model.values is samples


def test_lazy_charts_fit_nested_inspection_form(panel_view, qtbot):
    from visualizer.src.app.theme import generate_application_stylesheet

    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    w.window().resize(680, 1000)
    w.window().setStyleSheet(generate_application_stylesheet())
    w.window().show()
    w.sections["Inspect cells"].expand()
    w.show_slices.setChecked(True)
    charts = w.cell_table.charts
    charts.enabled.setChecked(True)
    w._select_slice("xy", 15, -17)
    qtbot.waitUntil(lambda: charts.canvas is not None and w.cell_table.loaded is not None)

    def fully_contained():
        child = charts.canvas
        while child is not w:
            parent = child.parentWidget()
            if not parent.rect().contains(child.geometry()):
                return False
            child = parent
        return True

    qtbot.waitUntil(fully_contained)
    charts.enabled.setChecked(False)


def test_default_coverage_has_no_selected_cell_and_uses_shared_labels(panel_view, qtbot, tmp_path):
    viz, panel, controller = panel_view
    controller.handle_coverage_toggled(True)
    w = loaded(qtbot, panel)
    assert w.selected_indices is None and not w._selection_objects
    assert w.coordinate.currentIndex() == 0  # Saved starting height, not a cell pick.
    assert w.readable_transmitter.isChecked()
    label = viz.renderer.objects["analysis:coverage_site:A:label"]
    assert "label_background_rgba" not in label.metadata
    assert label.payload.text == "A"
    assert label.material_payload.base_color == (1, 0, 0, 1)
    w.show_slices.setChecked(True)
    w.cell_table.enabled.setChecked(True)
    w.cell_table.charts.enabled.setChecked(True)
    w.orientation.setCurrentIndex(w.orientation.findData(1))
    w.coordinate.setCurrentIndex(2)
    for view in w.slice_view.views.values():
        assert view.plot is not None and not view.marker.isVisible()
        assert not view.horizontal.isVisible() and not view.vertical.isVisible()
    assert w.selected_indices is None and not w._selection_objects
    assert w.cell_table.pending is None and w.cell_table.charts.canvas is None
    path = tmp_path / "no-cell.json"
    w._save_view(path)
    w._select_slice("xy", 15, -17)
    assert w.selected_indices is not None and w._selection_objects
    assert all(view.marker.isVisible() for view in w.slice_view.views.values())
    qtbot.waitUntil(lambda: w.cell_table.charts.canvas is not None)
    w._load_view(path)
    assert w.selected_indices is None and not w._selection_objects
    assert "Click a cell" in w.cell_note.text()
    assert w.cell_table.table.isHidden() and w.cell_table.charts.canvas.isHidden()
    controller.handle_coverage_metric_changed("best_rss_dbm")
    loaded(qtbot, panel)
    assert w.selected_indices is None and not w._selection_objects
