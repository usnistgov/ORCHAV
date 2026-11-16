"""Detector inspection can retain its maps without a selected saved cell."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit.test_sensing_stage_dashboard import (
    _replay_event_for,
    _request,
    _with_detection,
)
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing import sensing_window as sensing_window_module
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.stage_inspector import build_detection_stage_bundle


@pytest.fixture
def detector(qtbot):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    request = _request(sensing, bundle, _replay_event_for(bundle))
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(request)
    return window, request


def test_clear_is_available_without_explanation_and_persists_during_playback(detector, qtbot):
    window, request = detector
    assert window.stage_clear_selection_button.isVisible()
    assert not window.selection_tools.isVisible()
    assert window._linked_selection is None
    assert len(window.stage_cell.getData()[0]) == 0
    window._select_inspection_coordinate(2, 1)
    assert len(window.stage_cell.getData()[0]) == 1
    image = window.cfar_rd_image.image.copy()
    qtbot.mouseClick(window.stage_clear_selection_button, Qt.MouseButton.LeftButton)
    assert window._linked_selection is None
    assert window._cell_selection_cleared()
    assert len(window.stage_cell.getData()[0]) == 0
    assert not window.stage_slice_line.isVisible()
    assert not window.stage_range_line.isVisible()
    assert "No cell selected" in window.detection_stage_summary.text()
    np.testing.assert_array_equal(window.cfar_rd_image.image, image)

    following = deepcopy(request.display_frame["sensing"])
    following["temporal"]["acquisition"]["acquisition_id"] = "6411913b-405d-4e2c-80aa-c3cdc6af5acd"
    _rematerialize_products(following)
    bundle = build_detection_stage_bundle(following)
    next_request = _request(following, bundle, _replay_event_for(bundle))
    for current in (request, next_request, request):
        window.update_from_request(current)
        assert window._cell_selection_cleared()
        assert window._linked_selection is None
        assert len(window.stage_cell.getData()[0]) == 0
        assert "No cell selected" in window.detection_stage_summary.text()

    window.stage_comparison_scope.setCurrentIndex(window.stage_comparison_scope.findData("reports"))
    assert window._cell_selection_cleared()
    assert window.stage_report_input.getData()[0].size == 1
    assert len(window.stage_cell.getData()[0]) == 0
    window._detection_stage_range_box.setValue(1)
    assert not window._cell_selection_cleared()
    assert window._linked_selection.cell.range_bin == 1
    assert len(window.stage_cell.getData()[0]) == 1


def test_new_recording_also_starts_without_forced_selection(detector, qtbot):
    window, request = detector
    qtbot.mouseClick(window.stage_clear_selection_button, Qt.MouseButton.LeftButton)
    window.update_from_request(replace(request, source_identity=("another recording",)))
    assert window._cell_selection_cleared()
    assert len(window.stage_cell.getData()[0]) == 0
    np.testing.assert_array_equal(
        window.cfar_rp_power.getData()[0], request.detection_stage_bundle.range_axis_m
    )
    assert window.cfar_rd_image.image is not None


def test_unselected_detector_maps_fit_saved_grid_before_scroll_or_paint(detector):
    window, request = detector
    bundle = request.detection_stage_bundle
    y = window._doppler_to_velocity(bundle.doppler_axis_hz, request.carrier_freq_hz)
    rect = window._rd_rect(bundle.range_axis_m, y)
    expected = ((rect.left(), rect.right()), (rect.top(), rect.bottom()))
    maps = (window.cfar_rd_plot, *window._stage_maps.values())
    # Maps below the scroll viewport must not depend on a Qt paint or selected
    # marker to replace the default [0, 1] ViewBox limits.
    for change in (
        lambda: None,
        lambda: window.select_detection_index(0),
        lambda: window.set_linked_selection(None, notify=True),
    ):
        change()
        for plot in maps:
            np.testing.assert_allclose(plot.viewRange(), expected)
        np.testing.assert_allclose(window.cfar_rp_plot.viewRange()[0], expected[0])


@pytest.mark.parametrize("has_eligible_cells", [True, False])
def test_empty_detection_result_displays_saved_slice_without_selecting_a_cell(
    qtbot, has_eligible_cells
):
    sensing = _with_detection(_typed_sensing())
    sensing["detections"] = []
    bundle = build_detection_stage_bundle(sensing)
    assert not bundle.detection_links
    input_power = np.array([[[1000.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 100, 12]]])
    threshold_power = np.full_like(input_power, 70.0)
    eligible = np.zeros(input_power.shape, dtype=bool)
    eligible[0, 2] = has_eligible_cells
    arrays = {
        "range_doppler_detection_input_power": input_power,
        "range_doppler_cfar_threshold_power": threshold_power,
        "range_doppler_cfar_eligible_mask": eligible,
    }
    originals = {kind: values.copy() for kind, values in arrays.items()}
    bundle = replace(
        bundle,
        products=tuple(
            replace(product, values=arrays[product.kind]) if product.kind in arrays else product
            for product in bundle.products
        ),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("diagnostics")
    selections = []
    window.selection_changed.connect(selections.append)
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))

    row = 2 if has_eligible_cells else 0
    assert window._detection_stage_doppler_box.value() == row
    assert window._linked_selection is None
    assert selections == []
    assert not window.detection_table.selectionModel().hasSelection()
    assert not window.selection_tools_toggle.isChecked()
    assert len(window.stage_cell.getData()[0]) == 0
    np.testing.assert_array_equal(window.cfar_rp_power.getData()[0], bundle.range_axis_m)
    np.testing.assert_allclose(
        window.cfar_rp_power.getData()[1], 10 * np.log10(input_power[0, row])
    )
    np.testing.assert_allclose(
        window.cfar_rp_thresh.getData()[1],
        np.where(eligible[0, row], 10 * np.log10(threshold_power[0, row]), np.nan),
        equal_nan=True,
    )
    assert f"Doppler bin {row}" in window.cfar_rp_plot.getPlotItem().titleLabel.text

    # An explicitly inspected row stays visible after its cell is deselected.
    window._select_inspection_coordinate(1, 1)
    window.stage_clear_selection_button.click()
    assert window._linked_selection is None
    assert window._detection_stage_doppler_box.value() == 1
    assert not window.stage_range_line.isVisible()
    assert len(window.stage_cell.getData()[0]) == 0
    np.testing.assert_allclose(window.cfar_rp_power.getData()[1], 10 * np.log10(input_power[0, 1]))
    for kind, values in arrays.items():
        np.testing.assert_array_equal(values, originals[kind])


def test_clicking_selected_detection_table_row_clears_it(detector, qtbot):
    window, _ = detector
    index = window.detection_table_proxy.index(0, 0)

    def click_row():
        qtbot.mouseClick(
            window.detection_table.viewport(),
            Qt.MouseButton.LeftButton,
            pos=window.detection_table.visualRect(index).center(),
        )

    click_row()
    assert window._linked_selection.detection_index == 0
    assert window.detection_table.selectionModel().hasSelection()
    click_row()
    assert window._linked_selection is None
    assert window._cell_selection_cleared()
    assert not window.detection_table.selectionModel().hasSelection()
    assert len(window.stage_cell.getData()[0]) == 0
    window.select_detection_index(0)
    window.select_detection_index(0)  # Programmatic synchronization remains idempotent.
    assert window._linked_selection.detection_index == 0


def test_replay_export_does_not_include_a_cleared_cell(detector, qtbot, monkeypatch, tmp_path):
    window, request = detector
    destination = tmp_path / "replay-without-cell"
    monkeypatch.setattr(
        sensing_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), ""),
    )
    calls = []

    def export(event, output, **options):
        calls.append((event, options))
        return SimpleNamespace(output_directory=output)

    monkeypatch.setattr(sensing_window_module, "export_replay_session", export)
    qtbot.mouseClick(window.stage_clear_selection_button, Qt.MouseButton.LeftButton)
    window.detection_stage_export_button.click()
    assert calls == [
        (
            request.replay_event,
            {"stage_bundle": None, "stage_selection": None, "stage_kinds": None},
        )
    ]
    assert "no cell selected; replay-only package" in window.detection_stage_text.toPlainText()
