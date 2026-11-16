"""Raw-input views remain useful without selecting a detector output."""

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect

from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit import test_sensing_dashboard_layout as dashboard_layout
from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from tests.visualizer.unit.test_sensing_processing_inspection import signal_recording
from visualizer.src.sensing import processing_inspection as inspection
from visualizer.src.sensing.processing_panel import ProcessingPanel
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle

dashboard_tab = dashboard_layout.dashboard_tab
pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


def test_cir_default_uses_saved_input_energy_without_detections(qtbot, monkeypatch):
    sensing = signal_recording()
    sensing["detections"] = []
    sensing["samples"][..., 6] *= 4
    monkeypatch.setattr(
        inspection, "compute_range_doppler", lambda **_: pytest.fail("Input display reran an FFT")
    )
    model = inspection.inspect_processing(sensing, "inputs")
    assert model.default_coordinate == 6
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    selected = []
    panel.cell_selected.connect(lambda *args: selected.append(args))
    panel.set_model(model, "inputs")
    panel.apply_cell(None, cleared=True)
    assert panel.range_bin.value() == 6
    assert panel.element.currentIndex() == 0
    np.testing.assert_allclose(panel.before_curve.yData, abs(model.samples[0, :, 6]))
    assert "Automatic" in panel.coordinate.toolTip()
    assert selected == []


def test_cfr_default_uses_retained_samples_without_detections(qtbot):
    sensing = native_cfr(detection=False, elements=2)
    model = inspection.inspect_processing(sensing, "frequency")
    expected = int(np.argmax(np.sum(abs(model.samples) ** 2, axis=(0, 2))))
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    selected = []
    panel.cell_selected.connect(lambda *args: selected.append(args))
    panel.set_model(model, "frequency")
    panel.apply_cell(None, cleared=True)
    assert panel.range_bin.value() == expected
    np.testing.assert_allclose(panel.before_curve.yData, abs(model.samples[0, expected]))
    assert panel.coordinate.text() == f"{model.times[expected]:.6g} s"
    assert selected == []


def test_nonfinite_cfr_has_no_automatic_coordinate():
    sensing = native_cfr(detection=False)
    sensing["cfr_by_pair_rx_element"][..., 3, 0] = np.nan
    model = inspection.inspect_processing(sensing, "frequency")
    assert model.samples is None
    assert model.default_coordinate is None
    assert "nonfinite" in model.notice


def test_input_coordinates_and_receivers_remain_independent_across_views_and_acquisitions(qtbot):
    sensing = native_cfr(elements=2)
    cir = inspection.inspect_processing(sensing, "inputs")
    cfr = inspection.inspect_processing(sensing, "frequency")
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.set_model(cir, "inputs")
    panel.range_bin.setValue(7)
    panel.element.setCurrentIndex(1)
    panel.set_model(cfr, "frequency")
    panel.range_bin.setValue(3)
    assert panel.element.currentIndex() == 0
    panel.set_model(None, "inputs")  # A pending read does not discard user choices.
    panel.set_model(replace(cir, times=cir.times + 1), "inputs")
    assert panel.range_bin.value() == 7
    assert panel.element.currentIndex() == 1
    assert not panel.coordinate.toolTip()
    panel.set_model(replace(cfr, times=cfr.times + 1), "frequency")
    assert panel.range_bin.value() == 3
    assert panel.element.currentIndex() == 0
    np.testing.assert_allclose(panel.before_curve.yData, abs(cfr.samples[0, 3]))


def test_new_scope_or_changed_range_grid_retires_input_choices(qtbot):
    cir = inspection.inspect_processing(signal_recording(), "inputs")
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.set_model(cir, "inputs")
    panel.range_bin.setValue(3)
    panel.element.setCurrentIndex(1)
    panel.set_model(replace(cir, ranges=cir.ranges + 1, default_coordinate=4), "inputs")
    assert panel.range_bin.value() == 4
    panel.range_bin.setValue(2)
    panel.set_model(replace(cir, scale_scope=("different-run",), default_coordinate=5), "inputs")
    assert panel.range_bin.value() == 5
    assert panel.element.currentIndex() == 0


def test_clearing_dashboard_selection_keeps_current_cir_coordinate_and_curves(qtbot):
    request = _request()
    source = inspection.inspect_processing(signal_recording(), "inputs")
    model = replace(
        source, ranges=np.arange(3.0), samples=source.samples[..., :3], default_coordinate=2
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("diagnostics")
    window._support_box.setCurrentIndex(window._support_box.findData("inputs"))
    window.update_from_request(replace(request, processing_inspection=model))
    panel = window.processing_panel
    assert window._linked_selection is None
    assert panel.range_bin.value() == 2
    assert len(panel.before_curve.yData) == len(model.times)
    window._select_inspection_coordinate(1, 1)
    assert panel.range_bin.value() == 1
    window.set_linked_selection(None, notify=True)
    assert window._linked_selection is None
    assert panel.range_bin.value() == 1
    np.testing.assert_allclose(panel.before_curve.yData, abs(model.samples[0, :, 1]))
    assert panel.coordinate.text() != "No cell selected"


@pytest.mark.parametrize("mode", ["inputs", "frequency"])
def test_input_magnitude_and_phase_fit_after_detector_view(dashboard_tab, qtbot, mode):
    _, tabs, _, window = dashboard_tab
    window.figure_options.expand()
    window.table_button.setChecked(False)
    window._graph_height_box.setValue(180)
    window._open_result_detail("diagnostics", "cfar")
    request = _request()
    window.update_from_request(
        replace(request, detection_stage_bundle=build_detection_stage_bundle(_typed_sensing()))
    )
    qtbot.waitUntil(lambda: window.evidence_scroll.verticalScrollBar().maximum() > 0)
    window._open_result_detail("diagnostics", mode)
    window.update_from_request(
        replace(request, processing_inspection=inspection.inspect_processing(native_cfr(), mode))
    )
    panel = window.processing_panel
    viewport = window.evidence_scroll.viewport()

    def both_rows_fit():
        return window.evidence_scroll.verticalScrollBar().maximum() == 0 and all(
            viewport.rect().contains(QRect(plot.mapTo(viewport, QPoint()), plot.size()))
            for plot in (panel.slice, panel.phase)
        )

    qtbot.waitUntil(both_rows_fit)
    for plot in (panel.slice, panel.phase):
        item = plot.getPlotItem()
        scene = plot.sceneRect().adjusted(-1, -1, 1, 1)
        assert scene.contains(item.getAxis("bottom").sceneBoundingRect())
        assert scene.contains(item.getAxis("left").sceneBoundingRect())
        assert plot.getViewBox().height() >= 180
    original = [p.getViewBox().height() for p in (panel.slice, panel.phase)]
    # The tall dashboard can already allocate more than 420 px to each plot.
    # Request growth relative to that allocation, including axes and titles.
    requested_height = max(420, max(p.height() for p in (panel.slice, panel.phase)) + 80)
    window._graph_height_box.setValue(requested_height)
    qtbot.waitUntil(
        lambda: all(
            p.getViewBox().height() > height + 60
            for p, height in zip((panel.slice, panel.phase), original, strict=True)
        )
    )
    qtbot.waitUntil(lambda: window.evidence_scroll.verticalScrollBar().maximum() > 0)
    window._graph_height_box.setValue(180)
    qtbot.waitUntil(both_rows_fit)
    tabs.resize(440, 950)
    qtbot.waitUntil(lambda: panel.width() < 400)
    controls = (
        panel.range_label,
        panel.range_bin,
        panel.coordinate,
        panel.element,
        panel.details_toggle,
    )
    qtbot.waitUntil(lambda: all(c.geometry().bottom() < panel.notice.y() for c in controls))
    assert all(panel.rect().contains(c.geometry()) for c in controls)
    assert panel.notice.geometry().bottom() < panel.slice.y()
    assert window.evidence_scroll.horizontalScrollBar().maximum() == 0
    tabs.resize(1300, 1340)
    qtbot.waitUntil(both_rows_fit)
