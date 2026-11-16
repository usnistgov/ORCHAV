"""Headless dashboard tests for persisted detector-stage inspection."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QApplication

from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing import sensing_window as sensing_window_module
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    DetectionStageBundle,
    InspectionDisposition,
    build_detection_stage_bundle,
    load_detection_stage_bundle,
)
from visualizer.src.sensing.update_request import SensingUpdateRequest

_APP = QApplication.instance() or QApplication([])


def _with_detection(sensing: dict[str, object]) -> dict[str, object]:
    products = sensing["products"]
    assert isinstance(products, dict)
    slow_time_key = "slow_time_by_pair_rx_element"
    slow_time_descriptor = deepcopy(products[slow_time_key])
    if "combine" not in slow_time_descriptor:
        slow_time_descriptor["combine"] = {
            "policy": "rx_broadside_unitary_voltage_sum",
            "input_axis": "rx_element",
            "normalization": "one_over_sqrt_element_count",
        }
        products[slow_time_key] = slow_time_descriptor
        _rematerialize_products(sensing)
    sensing["detections"] = [
        {
            "cluster_id": 0,
            "doppler_bin": 1,
            "range_bin": 2,
            "range_m": 2.0,
            "doppler_hz": 0.0,
            "snr_db": 3.0102999566,
        }
    ]
    return sensing


def _replay_event_for(
    bundle: DetectionStageBundle,
    **overrides: Any,
) -> SimpleNamespace:
    event = bundle.event
    assert event is not None
    values = {
        "clock_id": event.clock_id,
        "acquisition_id": event.acquisition_id,
        "processing_run_id": event.processing_run_id,
        "processing_recipe_id": event.processing_recipe_id,
        "pair_id": event.pair_id,
        "reference_time_s": event.reference_time_s,
    }
    values.update(overrides)
    key = SimpleNamespace(
        clock_id=values["clock_id"],
        acquisition_id=values["acquisition_id"],
        processing_run_id=values["processing_run_id"],
        pair_id=values["pair_id"],
        reference_time_s=values["reference_time_s"],
    )
    recipe = SimpleNamespace(
        processing_run_id=values["processing_run_id"],
        processing_recipe_id=values["processing_recipe_id"],
    )
    return SimpleNamespace(key=key, recipe=recipe)


def _request(
    sensing: dict[str, object],
    bundle: DetectionStageBundle | None,
    replay_event: Any,
) -> SensingUpdateRequest:
    return SensingUpdateRequest(
        display_frame={"sensing": sensing},
        state_frame={},
        state_step=0,
        replay_event=replay_event,
        detection_stage_bundle=bundle,
    )


def _select_stage(window: SensingWindow, kind: str) -> None:
    window._detection_stage_kind = kind
    if window._last_request.display_frame["sensing"].get("detections"):
        window.select_detection_index(0)
    window._update_detection_stage_inspector_from_controls()


def test_diagnostics_inspects_linked_persisted_cell_and_lineage() -> None:
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    replay_event = _replay_event_for(bundle)
    request = _request(sensing, bundle, replay_event)
    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")

    window._set_view_mode("diagnostics")
    window.update_from_request(request)
    _select_stage(window, "range_doppler_snr_db")

    text = window.detection_stage_text.toPlainText()
    assert window._detection_stage_doppler_box.value() == 1
    assert window._detection_stage_range_box.value() == 2
    assert "Source: immutable typed Phase-4 products; no sensing reprocessing performed." in text
    assert "eligibility=eligible" in text
    assert "CFAR=passed" in text
    assert "NMS=retained" in text
    assert "cluster=clustered" in text
    assert "cluster_id=0" in text
    assert "doppler_bin=1 -> 0 Hz" in text
    assert "range_bin=2 -> 2 m" in text
    expected_snr = bundle.inspect(window._current_detection_stage_selection(bundle)).snr_db
    assert expected_snr is not None
    assert f"value={format(expected_snr, '.17g')} dB" in text
    assert bundle.event is not None
    assert f"acquisition_id={bundle.event.acquisition_id}" in text
    assert f"processing_run_id={bundle.event.processing_run_id}" in text
    assert f"processing_recipe_id={bundle.event.processing_recipe_id}" in text
    assert f"pair_id={bundle.event.pair_id}" in text
    lineage = text.split("Ordered lineage (root first):", maxsplit=1)[1].split(
        "Stage inventory", maxsplit=1
    )[0]
    lineage_lines = [line.strip() for line in lineage.splitlines() if line.strip()]
    assert "kind=range_doppler_snr_db" in lineage_lines[0]
    assert any("kind=range_doppler" in line for line in lineage_lines)
    for kind in DETECTION_STAGE_KINDS:
        assert kind in text
    window.close()


def test_detector_map_uses_stage_lineage_independently_of_displayed_rd() -> None:
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    products = sensing["products"]
    assert isinstance(products, dict)
    rd_key, rd_descriptor = next(
        (key, descriptor)
        for key, descriptor in products.items()
        if isinstance(descriptor, dict) and descriptor.get("kind") == "range_doppler"
    )
    rd_values = np.asarray(sensing[rd_key])
    rd_result = SimpleNamespace(
        power_linear=np.abs(rd_values[0]) ** 2,
        doppler_axis_hz=np.asarray(sensing["doppler_axis_hz"]),
        range_axis_m=np.asarray(sensing["range_axis_m"]),
        range_axis_label="Monostatic range (m)",
        metadata={"product_id": rd_descriptor["product_id"], "range_mode": "monostatic"},
    )
    request = replace(
        _request(sensing, bundle, _replay_event_for(bundle)),
        rd_result=rd_result,
    )
    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")

    window.update_from_request(request)

    expected = np.asarray(sensing["range_doppler_cfar_mask_by_pair"])[0].T
    np.testing.assert_array_equal(window._persisted_cfar_mask_for_rd(rd_result).T, expected)
    np.testing.assert_array_equal(window.cfar_rd_mask.image, expected)
    assert "Detection input power" in window.cfar_rd_plot.getPlotItem().titleLabel.text
    assert window.rd_plot.isHidden()

    mismatched = replace(
        request,
        rd_result=SimpleNamespace(**{**vars(rd_result), "metadata": {"product_id": "0" * 64}}),
    )
    window.update_from_request(mismatched)
    assert window._persisted_cfar_mask_for_rd(mismatched.rd_result) is None
    np.testing.assert_array_equal(window.cfar_rd_mask.image, expected)
    assert "unavailable" not in window.cfar_rd_plot.getPlotItem().titleLabel.text
    window.close()


def test_diagnostics_exposes_missing_and_invalid_stages() -> None:
    missing_sensing = _with_detection(_typed_sensing())
    missing_sensing.pop("range_doppler_snr_db_by_pair")
    products = missing_sensing["products"]
    assert isinstance(products, dict)
    products.pop("range_doppler_snr_db_by_pair")
    missing_bundle = build_detection_stage_bundle(missing_sensing)
    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")

    window.update_from_request(
        _request(missing_sensing, missing_bundle, _replay_event_for(missing_bundle))
    )
    _select_stage(window, "range_doppler_snr_db")

    missing_text = window.detection_stage_text.toPlainText()
    assert "kind=range_doppler_snr_db" in missing_text
    assert "disposition=unavailable (descriptor_absent)" in missing_text
    assert "value=unavailable dB" in missing_text
    assert "range_doppler_snr_db] | unavailable |" in missing_text

    valid_sensing = _with_detection(_typed_sensing())
    valid_bundle = build_detection_stage_bundle(valid_sensing)
    contradictory = deepcopy(valid_sensing)
    invalid_products = contradictory["products"]
    assert isinstance(invalid_products, dict)
    key = "range_doppler_cfar_noise_power_by_pair"
    descriptor = deepcopy(invalid_products[key])
    descriptor["lineage"]["processing_run_id"] = "wrong-run"
    invalid_products[key] = descriptor
    invalid_bundle = load_detection_stage_bundle(contradictory)
    assert invalid_bundle.disposition is InspectionDisposition.INVALID

    window.update_from_request(
        _request(valid_sensing, invalid_bundle, _replay_event_for(valid_bundle))
    )

    invalid_text = window.detection_stage_text.toPlainText()
    assert "Persisted detector-stage inspection invalid:" in invalid_text
    assert "lineage.processing_run_id must match temporal result" in invalid_text
    assert "No sensing reprocessing is performed." in invalid_text
    assert not window._detection_stage_pair_box.isEnabled()
    assert not window._detection_stage_doppler_box.isEnabled()
    assert not window._detection_stage_range_box.isEnabled()
    assert invalid_text.count(" | invalid | ") == len(DETECTION_STAGE_KINDS)

    assert not window.detection_stage_export_button.isEnabled()
    window._export_detection_stage_replay()
    export_text = window.detection_stage_text.toPlainText()
    assert "Replay export status:\nblocked: detector-stage bundle is invalid" in export_text
    window.close()


def test_replay_bundle_event_mismatch_fails_before_ui_mutation() -> None:
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    good_request = _request(sensing, bundle, _replay_event_for(bundle))
    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(good_request)
    previous_text = window.detection_stage_text.toPlainText()

    bad_event = _replay_event_for(bundle, processing_recipe_id="different-recipe")
    bad_request = _request(sensing, bundle, bad_event)

    with pytest.raises(ValueError, match="identity mismatch: recipe"):
        window.update_from_request(bad_request)

    assert window._last_request is good_request
    assert window.detection_stage_text.toPlainText() == previous_text
    window.close()


def test_stage_inspection_is_window_order_and_rerender_invariant() -> None:
    first_sensing = _with_detection(_typed_sensing())
    first_bundle = build_detection_stage_bundle(first_sensing)
    first_request = _request(first_sensing, first_bundle, _replay_event_for(first_bundle))

    second_sensing = _with_detection(_typed_sensing())
    temporal = second_sensing["temporal"]
    assert isinstance(temporal, dict)
    result = temporal["result"]
    assert isinstance(result, dict)
    result["processing_run_id"] = "5e425f06-1b9d-4f00-92ad-9ac786f809d6"
    _rematerialize_products(second_sensing)
    second_bundle = build_detection_stage_bundle(second_sensing)
    second_request = _request(
        second_sensing,
        second_bundle,
        _replay_event_for(second_bundle),
    )

    ordered = SensingWindow()
    ordered.update_from_request(first_request)
    ordered.update_from_request(second_request)
    ordered_text = ordered.detection_stage_text.toPlainText()

    fresh = SensingWindow()
    fresh.update_from_request(second_request)
    assert fresh.detection_stage_text.toPlainText() == ordered_text

    ordered._rerender_last_request()
    assert ordered.detection_stage_text.toPlainText() == ordered_text
    ordered.close()
    fresh.close()


def test_explicit_export_action_uses_dialog_and_one_composite_exporter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    replay_event = _replay_event_for(bundle)
    request = _request(sensing, bundle, replay_event)
    destination = tmp_path / "chosen-replay-package"
    calls: dict[str, Any] = {}

    def choose_new_package(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        calls["dialog"] = True
        return str(destination), "Replay package directory (*)"

    def export_replay(
        event: Any,
        output_directory: Path,
        **stage_options: Any,
    ) -> SimpleNamespace:
        calls["replay_count"] = int(calls.get("replay_count", 0)) + 1
        calls["replay"] = (event, output_directory, stage_options)
        assert not output_directory.exists()
        return SimpleNamespace(output_directory=output_directory)

    monkeypatch.setattr(
        sensing_window_module.QFileDialog,
        "getSaveFileName",
        choose_new_package,
    )
    monkeypatch.setattr(sensing_window_module, "export_replay_session", export_replay)

    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(request)
    _select_stage(window, "range_doppler_snr_db")
    window.detection_stage_export_button.click()

    assert calls["dialog"] is True
    exported_event, export_root, stage_options = calls["replay"]
    assert exported_event is replay_event
    assert export_root == destination.resolve()
    assert stage_options["stage_bundle"] is bundle
    selection = stage_options["stage_selection"]
    assert selection.pair_id == bundle.event.pair_id
    assert selection.doppler_bin == 1
    assert selection.range_bin == 2
    assert selection.cluster_id == 0
    assert stage_options["stage_kinds"] == ("range_doppler_snr_db",)
    assert "Replay export status:\ncomplete:" in window.detection_stage_text.toPlainText()
    assert "selected persisted detector cell included" in (
        window.detection_stage_text.toPlainText()
    )

    destination.mkdir()
    window.detection_stage_export_button.click()
    assert calls["replay_count"] == 1
    assert "blocked: destination already exists" in window.detection_stage_text.toPlainText()
    window.close()


def test_explicit_export_action_preserves_replay_only_when_stage_bundle_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sensing = _with_detection(_typed_sensing())
    identity_bundle = build_detection_stage_bundle(sensing)
    replay_event = _replay_event_for(identity_bundle)
    request = _request(sensing, None, replay_event)
    destination = tmp_path / "replay-only-package"
    calls: list[tuple[Any, Path, dict[str, Any]]] = []

    monkeypatch.setattr(
        sensing_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "Replay package directory (*)"),
    )

    def export_replay_only(
        event: Any,
        output_directory: Path,
        **stage_options: Any,
    ) -> SimpleNamespace:
        calls.append((event, output_directory, stage_options))
        return SimpleNamespace(output_directory=output_directory)

    monkeypatch.setattr(
        sensing_window_module,
        "export_replay_session",
        export_replay_only,
    )

    window = SensingWindow()
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(request)
    window.detection_stage_export_button.click()

    assert calls == [
        (
            replay_event,
            destination.resolve(),
            {
                "stage_bundle": None,
                "stage_selection": None,
                "stage_kinds": None,
            },
        )
    ]
    assert "replay-only package" in window.detection_stage_text.toPlainText()
    window.close()


def test_all_stage_maps_show_saved_evidence_without_changing_arrays(qtbot, monkeypatch):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    saved = {p.kind: np.array(p.values, copy=True) for p in bundle.products if p.available}
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))

    def no_processing(*args, **kwargs):
        raise AssertionError("Inspecting saved evidence must not run signal processing")

    for name in (
        "compute_micro_doppler",
        "compute_range_collapsed_micro_doppler",
        "compute_range_tracked_micro_doppler",
    ):
        monkeypatch.setattr(sensing_window_module, name, no_processing)
    for kind in ("range_doppler_detection_input_power", "range_doppler_cfar_threshold_power"):
        _select_stage(window, kind)
        image = (
            window.cfar_rd_image
            if kind.endswith("input_power")
            else window.stage_threshold_plot.image
        )
        np.testing.assert_allclose(image.image, 10 * np.log10(np.maximum(saved[kind][0].T, 1e-30)))
        assert window.cfar_rd_mask.isVisible()
        assert window.cfar_pass_borders.isVisible()
        assert "selected" in window.detection_stage_summary.text().lower()
    window._show_cfar_cb.setChecked(False)
    assert not window.cfar_rd_mask.isVisible()
    assert not window.cfar_pass_borders.isVisible()
    window._show_cfar_cb.setChecked(True)
    _select_stage(window, "range_doppler_cfar_mask")
    np.testing.assert_array_equal(
        window.stage_cfar_plot.image.image, saved["range_doppler_cfar_mask"][0].T
    )
    assert window.stage_cfar_plot.image.getLevels() == pytest.approx([0, 1])
    assert not window._rd_dynamic_range_box.isHidden()
    _select_stage(window, "range_doppler_snr_db")
    snr = saved["range_doppler_snr_db"][0].T.astype(float)
    snr[~saved["range_doppler_snr_valid_mask"][0].T.astype(bool)] = np.nan
    np.testing.assert_allclose(window.stage_snr_plot.image.image, snr, equal_nan=True)
    _select_stage(window, "range_doppler_cluster_labels")
    clusters = saved["range_doppler_cluster_labels"][0].T.astype(float)
    clusters[clusters < 0] = np.nan
    np.testing.assert_allclose(window.stage_cluster_plot.image.image, clusters, equal_nan=True)
    assert not hasattr(window, "_detection_stage_box")
    assert "range_doppler_cfar_eligible_mask" not in window._stage_maps
    assert "range_doppler_cfar_training_cell_count" not in window._stage_maps
    assert "range_doppler_snr_valid_mask" not in window._stage_maps
    assert not hasattr(window, "stage_valid_plot")
    for plot in window._stage_maps.values():
        assert not plot.isHidden()
    for product in bundle.products:
        if product.available:
            np.testing.assert_array_equal(product.values, saved[product.kind])


def test_detector_maps_share_cell_selection_and_use_two_columns(qtbot):
    from PySide6.QtCore import QPointF, Qt

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window._doppler_hz_cb.setChecked(True)
    _APP.processEvents()
    assert window._grid.getItemPosition(window._grid.indexOf(window.cfar_rp_plot)) == (1, 0, 1, 2)
    for row, pair in enumerate(
        (
            (window.cfar_rd_plot, window.stage_threshold_plot),
            (window.stage_noise_plot, window.stage_snr_plot),
            (window.stage_nms_plot, window.stage_cluster_plot),
        ),
        3,
    ):
        for column, plot in enumerate(pair):
            assert window._grid.getItemPosition(window._grid.indexOf(plot)) == (row, column, 1, 1)
    assert window._grid.indexOf(window.detection_table) == -1
    assert window.result_browser.isVisible()
    assert window._grid.getItemPosition(window._grid.indexOf(window.stage_cfar_plot)) == (
        2,
        0,
        1,
        2,
    )
    plot = window.stage_snr_plot
    window.evidence_scroll.ensureWidgetVisible(plot)
    _APP.processEvents()
    pos = plot.getViewBox().mapViewToScene(QPointF(1.0, float(bundle.doppler_axis_hz[0])))
    image = plot.image.image
    window._select_stage_map_cell(
        SimpleNamespace(
            button=lambda: Qt.MouseButton.LeftButton,
            scenePos=lambda: pos,
            accept=lambda: None,
        ),
        plot,
        "range_doppler_snr_db",
    )
    assert window._detection_stage_kind == "range_doppler_snr_db"
    assert window._detection_stage_range_box.value() == 1
    assert window._detection_stage_doppler_box.value() == 0
    assert plot.image.image is image  # Cell inspection does not repaint all arrays.
    for stage in window._stage_maps.values():
        np.testing.assert_allclose(stage.cell.getData()[0], [1.0])
    window.set_linked_selection(None, notify=True)
    for stage in window._stage_maps.values():
        assert stage.cell.getData()[0] is None or len(stage.cell.getData()[0]) == 0
    # A full-width slice must use the same coordinates as the narrower maps,
    # including after a resize and a pan started in either plot.
    window.cfar_rd_plot.setXRange(0.5, 2.5, padding=0)
    qtbot.waitUntil(lambda: window.cfar_rp_plot.viewRange()[0] == [0.5, 2.5])
    window.resize(950, 900)
    _APP.processEvents()
    assert window.cfar_rp_plot.viewRange()[0] == pytest.approx([0.5, 2.5])
    window.cfar_rp_plot.setXRange(0.25, 2.75, padding=0)
    qtbot.waitUntil(lambda: window.cfar_rd_plot.viewRange()[0] == [0.25, 2.75])


@pytest.mark.parametrize("use_hz", [False, True], ids=["velocity", "doppler"])
def test_detector_grid_and_pass_borders_match_clickable_saved_cells(qtbot, use_hz):
    from PySide6.QtCore import QPointF, Qt

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window._doppler_hz_cb.setChecked(use_hz)
    _APP.processEvents()
    original_image = window.cfar_rd_image.image.copy()
    view = window.cfar_rd_plot.getViewBox()
    original_bounds = view.viewRange()
    assert not window.stage_bin_grid.isVisible()
    window._show_bin_grid_cb.click()
    assert window.stage_bin_grid.isVisible()
    assert window.bin_spacing_label.isVisible()
    np.testing.assert_allclose(view.viewRange(), original_bounds)
    assert "range 1 m" in window.bin_spacing_label.text()
    assert ("Doppler" if use_hz else "velocity") in window.bin_spacing_label.text()

    # Grid edges coincide with image edges and sit halfway between saved centres.
    grid_x, grid_y = window.stage_bin_grid.getData()
    rect = window.cfar_rd_image.mapRectToParent(window.cfar_rd_image.boundingRect())
    assert [min(grid_x), max(grid_x), min(grid_y), max(grid_y)] == pytest.approx(
        [rect.left(), rect.right(), rect.top(), rect.bottom()]
    )
    np.testing.assert_allclose(np.unique(grid_x), [-0.5, 0.5, 1.5, 2.5, 3.5])
    dy = rect.height() / 3
    np.testing.assert_allclose(np.unique(grid_y), np.array([-1.5, -0.5, 0.5, 1.5]) * dy)

    # Only saved cell (Doppler 1, range 2) passed CFAR in this fixture.
    border_x, border_y = window.cfar_pass_borders.getData()
    segments = np.column_stack((border_x, border_y)).reshape(-1, 2, 2)
    np.testing.assert_allclose(
        segments,
        [
            [[1.5, -dy / 2], [1.5, dy / 2]],
            [[2.5, -dy / 2], [2.5, dy / 2]],
            [[1.5, -dy / 2], [2.5, -dy / 2]],
            [[1.5, dy / 2], [2.5, dy / 2]],
        ],
    )

    # Clicking across a visible edge selects the adjacent bin on either axis.
    view.setRange(xRange=(-0.5, 3.5), yRange=(-1.5 * dy, 1.5 * dy), padding=0)
    for x, y, expected in ((1.49, -0.49 * dy, (1, 1)), (1.51, -0.51 * dy, (0, 2))):
        clicked = view.mapViewToScene(QPointF(x, y))
        window._select_stage_map_cell(
            SimpleNamespace(
                button=lambda: Qt.MouseButton.LeftButton,
                scenePos=lambda: clicked,
                accept=lambda: None,
            )
        )
        assert (
            window._detection_stage_doppler_box.value(),
            window._detection_stage_range_box.value(),
        ) == expected
    np.testing.assert_array_equal(window.cfar_rd_image.image, original_image)

    window._show_bin_grid_cb.click()
    assert not window.stage_bin_grid.isVisible()
    assert window.cfar_pass_borders.isVisible()  # Independent of the full-grid overlay.
    window._show_bin_grid_cb.click()
    window._set_view_mode("analysis")
    assert not window._show_bin_grid_cb.isVisible()
    assert not window.bin_spacing_label.isVisible()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, None, None))
    assert not window.stage_bin_grid.isVisible()
    assert not window.bin_spacing_label.isVisible()
    assert window.cfar_pass_borders.getData()[0] is None


def test_cell_and_detection_selection_drive_the_exact_threshold_slice(qtbot):
    from PySide6.QtCore import QPointF, Qt

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window._doppler_hz_cb.setChecked(True)
    _APP.processEvents()
    view = window.cfar_rd_plot.getViewBox()
    view.setRange(
        xRange=(-1, 4),
        yRange=(float(bundle.doppler_axis_hz[0]) - 1, float(bundle.doppler_axis_hz[-1]) + 1),
        padding=0,
    )
    clicked = view.mapViewToScene(QPointF(bundle.range_axis_m[1], bundle.doppler_axis_hz[0]))
    window._select_stage_map_cell(
        SimpleNamespace(
            button=lambda: Qt.MouseButton.LeftButton, scenePos=lambda: clicked, accept=lambda: None
        )
    )
    assert window._detection_stage_doppler_box.value() == 0
    assert window._detection_stage_range_box.value() == 1
    assert window.stage_slice_detections.getData()[0] is None
    window.select_detection_index(0)
    assert window._detection_stage_doppler_box.value() == 1
    assert window._detection_stage_range_box.value() == 2
    assert "Doppler bin 1" in window.cfar_rp_plot.getPlotItem().titleLabel.text
    np.testing.assert_allclose(
        window.cfar_rp_power.getData()[1],
        10
        * np.log10(
            np.maximum(bundle.product("range_doppler_detection_input_power").values[0, 1], 1e-30)
        ),
    )
    np.testing.assert_allclose(window.stage_slice_detections.getData()[0], [2])
    assert window.stage_range_line.value() == 2
    window._zoom_data_cb.setChecked(True)
    window._zoom_data_cb.setChecked(False)
    rect = window._rd_rect(bundle.range_axis_m, bundle.doppler_axis_hz)
    np.testing.assert_allclose(
        window.cfar_rd_plot.viewRange(), [[rect.left(), rect.right()], [rect.top(), rect.bottom()]]
    )
    window._zoom_data_cb.setChecked(True)
    assert window.cfar_rd_plot.getViewBox().autoRangeEnabled() == [False, False]


def test_threshold_slice_full_view_keeps_weak_input_and_threshold_peaks(qtbot):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    products = []
    for product in bundle.products:
        if product.kind == "range_doppler_detection_input_power":
            values = np.broadcast_to(
                np.array([1e-20, 1e-8, 1.0, 1e-15]), product.values.shape
            ).copy()
            product = replace(product, values=values)
        elif product.kind == "range_doppler_cfar_threshold_power":
            product = replace(product, values=np.full_like(product.values, 100.0))
        elif product.kind == "range_doppler_cfar_eligible_mask":
            product = replace(product, values=np.ones_like(product.values))
        products.append(product)
    bundle = replace(bundle, products=tuple(products))
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window.select_detection_index(0)
    assert not window._zoom_data_cb.isChecked()

    def fits_curves():
        low, high = window.cfar_rp_plot.viewRange()[1]
        return low < -200 and high > 20

    qtbot.waitUntil(fits_curves)
    for curve in (window.cfar_rp_power, window.cfar_rp_thresh):
        x, y = curve.getData()
        assert len(x) == len(bundle.range_axis_m)
        assert np.isfinite(y).all()
    window._zoom_data_cb.setChecked(True)
    window._zoom_data_cb.setChecked(False)
    qtbot.waitUntil(fits_curves)
    # The heatmap still uses its independently limited colour scale.
    assert window.cfar_rd_image.getLevels()[0] > -200
    window.close()


@pytest.mark.parametrize("hertz", [False, True])
def test_stage_cell_deselect_clears_details_and_preserves_slice_across_redraws(qtbot, hertz):
    from PySide6.QtCore import QPointF, Qt

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    request = _request(sensing, bundle, _replay_event_for(bundle))
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(request)
    window._doppler_hz_cb.setChecked(hertz)
    window.select_detection_index(0)
    power = window.cfar_rd_image.image.copy()
    assert len(window.stage_cell.getData()[0]) == 1
    _APP.processEvents()
    y = (
        bundle.doppler_axis_hz
        if hertz
        else window._doppler_to_velocity(bundle.doppler_axis_hz, request.carrier_freq_hz)
    )
    window.evidence_scroll.ensureWidgetVisible(window.cfar_rd_plot)
    _APP.processEvents()
    pos = window.cfar_rd_plot.getViewBox().mapViewToScene(QPointF(bundle.range_axis_m[2], y[1]))
    window._select_stage_map_cell(
        SimpleNamespace(
            button=lambda: Qt.MouseButton.LeftButton,
            scenePos=lambda: pos,
            accept=lambda: None,
        )
    )
    assert window._linked_selection is None
    for redraw in (lambda: None, lambda: window.update_from_request(request)):
        redraw()
        assert len(window.stage_cell.getData()[0]) == 0
        assert not window.stage_slice_line.isVisible()
        assert not window.stage_range_line.isVisible()
        np.testing.assert_array_equal(window.cfar_rp_power.getData()[0], bundle.range_axis_m)
        np.testing.assert_array_equal(window.cfar_rp_thresh.getData()[0], bundle.range_axis_m)
        assert window._detection_stage_doppler_box.value() == 1
        assert "Doppler bin 1" in window.cfar_rp_plot.getPlotItem().titleLabel.text
        assert "No cell selected" in window.detection_stage_summary.text()
        assert window.detection_stage_text.toPlainText().startswith("No cell selected.")
        np.testing.assert_array_equal(window.cfar_rd_image.image, power)
    window._detection_stage_range_box.setValue(1)
    assert window._linked_selection.cell.range_bin == 1
    assert window.stage_range_line.isVisible()
    assert window.cfar_rp_power.getData()[0] is not None
    window.close()


def test_detector_maps_replace_comparison_shortcut_and_keep_source_details(qtbot):
    from PySide6.QtCore import QPoint

    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    _APP.processEvents()
    assert window.detection_stage_details_toggle.isVisible()
    assert window.detection_stage_summary.mapTo(window, QPoint(0, 0)).y() > (
        window.cfar_rd_plot.mapTo(window, QPoint(0, 0)).y()
    )
    assert not hasattr(window, "stage_progression")
    assert window.cfar_rd_plot.isVisible() and window.cfar_rp_plot.isVisible()
    assert all(plot.isVisible() for plot in window._stage_maps.values())
    view = window.cfar_rd_plot.getViewBox()
    window._doppler_hz_cb.setChecked(True)
    view.setRange(xRange=(0.75, 2.75), yRange=(-0.5, 0.75), padding=0)
    assert window.cfar_rp_plot.viewRange()[0] == pytest.approx((0.75, 2.75))
    # The automatic overview can already display row 0 without a selection.
    # Change the control first to exercise an explicit user cell inspection.
    window._detection_stage_doppler_box.setValue(1)
    window._detection_stage_doppler_box.setValue(0)
    assert "Doppler bin 0" in window.cfar_rp_plot.getPlotItem().titleLabel.text
    assert window.cfar_rd_plot.isVisible() and window.cfar_rp_plot.isVisible()
    assert window.detection_stage_details_toggle.isVisible()
    window.detection_stage_details_toggle.click()
    assert window.stage_technical.isVisible()


def test_snr_without_saved_validity_does_not_display_unqualified_values(qtbot):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    bundle = replace(
        bundle,
        products=tuple(
            (
                replace(
                    p,
                    disposition=InspectionDisposition.UNAVAILABLE,
                    values=None,
                    reason="descriptor_absent",
                )
                if p.kind == "range_doppler_snr_valid_mask"
                else p
            )
            for p in bundle.products
        ),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    _select_stage(window, "range_doppler_snr_db")
    assert window.stage_snr_plot.image.image is None
    assert "validity mask not recorded" in window.stage_snr_plot.getPlotItem().titleLabel.text
    assert "value: not available" in window.detection_stage_summary.text()


def test_all_report_comparison_uses_each_own_doppler_threshold(qtbot):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    first = bundle.detection_links[0]
    bundle = replace(
        bundle, detection_links=(first, replace(first, doppler_bin=0, range_bin=1, cluster_id=1))
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window.stage_comparison_scope.setCurrentIndex(1)
    points = window.stage_report_input.points()
    assert len(points) == 2
    for point, (d, r) in zip(points, [(0, 1), (1, 2)], strict=True):
        assert point.pos().x() == bundle.range_axis_m[r]
        expected = 10 * np.log10(
            max(bundle.product("range_doppler_detection_input_power").values[0, d, r], 1e-30)
        )
        assert point.pos().y() == pytest.approx(expected)
        assert f"Doppler bin {d}" in point.data()
    assert window.cfar_rp_power.getData()[0] is None
    window.stage_comparison_scope.setCurrentIndex(0)
    assert len(window.stage_report_input.points()) == 0
    assert window._linked_selection is None
    np.testing.assert_array_equal(window.cfar_rp_power.getData()[0], bundle.range_axis_m)
    window.select_detection_index(0)
    assert window.cfar_rp_power.getData()[0] is not None


def test_surface_scope_controls_enable_only_the_active_inspection(qtbot):
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("diagnostics")
    window.update_from_request(_request(sensing, bundle, _replay_event_for(bundle)))
    window.stage_comparison_scope.setCurrentIndex(0)
    window.stage_surface_toggle.setChecked(True)
    assert not window.stage_comparison_scope.isEnabled()
    assert window.stage_comparison_scope.currentData() == "reports"
    assert not window._detection_stage_doppler_box.isEnabled()
    assert not window._detection_stage_range_box.isEnabled()
    window.stage_surface.scope.setCurrentText("Selected neighborhood")
    assert window._detection_stage_doppler_box.isEnabled()
    assert window.stage_comparison_scope.currentData() == "slice"
    assert not window.stage_comparison_scope.isEnabled()
    window.stage_surface.scope.setCurrentText("Whole grid")
    assert not window._detection_stage_range_box.isEnabled()
    window.stage_surface_toggle.setChecked(False)
    assert window.stage_comparison_scope.isEnabled()
    assert window.stage_comparison_scope.currentData() == "slice"
    assert window._detection_stage_range_box.isEnabled()


def test_range_power_fallback_is_explicit_and_never_replaces_saved_profile(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    power = np.array([[1e-20, 2e-20, 4e-20], [3e-20, 1e-20, 2e-20]])
    rd = SimpleNamespace(
        power_linear=power, range_axis_m=np.arange(3), range_axis_label="Range (m)"
    )
    window._update_range_profile(None, "", rd_result=rd)
    np.testing.assert_allclose(window.rp_curve.getData()[1], 10 * np.log10(power.sum(axis=0)))
    assert "Doppler sum" in window.rp_plot.getPlotItem().titleLabel.text
    assert "Saved RD power summed over Doppler" in window.rp_plot.toolTip()
    rp = SimpleNamespace(power_linear=np.array([1.0, 2.0, 3.0]), range_axis_m=np.arange(3))
    window._update_range_profile(rp, "", rd_result=rd)
    np.testing.assert_allclose(window.rp_curve.getData()[1], 10 * np.log10(rp.power_linear))
    assert "Range profile" in window.rp_plot.getPlotItem().titleLabel.text
    window._update_range_profile(None, "")
    assert window.rp_curve.getData()[0] is None
    assert window.rp_samples.getData()[0] is None
    assert "unavailable" in window.rp_plot.getPlotItem().titleLabel.text


def test_isolated_range_bins_show_zero_baseline_and_preserve_weak_samples(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    rd = SimpleNamespace(
        power_linear=np.array([[0.0, 1e-20, 0.0, 0.0, 1e-10]]), range_axis_m=np.arange(5)
    )
    window._update_range_profile(None, "", rd_result=rd)
    x, y = window.rp_curve.getData()
    np.testing.assert_array_equal(x, np.arange(5))
    np.testing.assert_allclose(y[[1, 4]], [-200, -100])
    assert np.all(y[[0, 2, 3]] < -200)
    assert window.rp_curve.opts["stepMode"] is None
    np.testing.assert_array_equal(window.rp_samples.getData()[0], [1, 4])
    assert "display floor" in window.rp_plot.toolTip()
    np.testing.assert_array_equal(rd.power_linear, [[0.0, 1e-20, 0.0, 0.0, 1e-10]])
    np.testing.assert_allclose(window.rp_plot.viewRange()[1], [-180, -97])
    window._profile_full_dynamic_range_cb.setChecked(True)
    ymin, ymax = window.rp_plot.viewRange()[1]
    assert ymin < -200 and ymax > -100


def test_profile_power_axis_follows_visible_range_without_dropping_samples(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    window.show()
    axis = np.arange(6, dtype=float)
    power_db = np.array([-250, -180, -125, -100, -120, -190], dtype=float)
    window._update_range_profile(
        SimpleNamespace(power_linear=10 ** (power_db / 10), range_axis_m=axis), ""
    )
    window.rp_plot.setXLink(None)
    window.rp_plot.setXRange(2, 4, padding=0)
    np.testing.assert_allclose(window.rp_plot.viewRange()[1], [-180, -97])
    window._profile_full_dynamic_range_cb.setChecked(True)
    low, high = window.rp_plot.viewRange()[1]
    assert -140 < low < -125 and high > -100
    np.testing.assert_allclose(window.rp_curve.getData()[1], power_db)
    window.rp_plot.setXRange(0, 5, padding=0)
    assert window.rp_plot.viewRange()[1][0] < -250
    window._profile_full_dynamic_range_cb.setChecked(False)
    np.testing.assert_allclose(window.rp_plot.viewRange()[1], [-180, -97])
    np.testing.assert_allclose(window.rp_curve.getData()[1], power_db)


@pytest.mark.parametrize("bins", [1, 4])
def test_sparse_profile_zoom_shows_neighboring_empty_cells(qtbot, bins):
    window = SensingWindow()
    qtbot.addWidget(window)
    axis = np.arange(128) * 299792458 / 1e9
    power = np.zeros((3, 128))
    power[1, 15 : 15 + bins] = np.geomspace(1e-11, 1e-13, bins)
    rd = SimpleNamespace(power_linear=power, range_axis_m=axis)
    window._update_range_profile(None, "", rd_result=rd)
    with np.errstate(divide="ignore"):
        window._apply_zoom_to_data(window.rd_plot, 10 * np.log10(power), axis, np.arange(3))
    lo, hi = window.rd_plot.viewRange()[0]
    assert lo < axis[14]
    assert hi > axis[15 + bins]
    assert np.isfinite(window.rp_curve.getData()[1]).all()


@pytest.mark.parametrize("fft_size", [128, 512])
@pytest.mark.parametrize("frequency_window", ["boxcar", "hann"])
def test_cfr_profile_keeps_peak_and_sidelobes_at_saved_coordinates(
    qtbot, fft_size, frequency_window
):
    from shared.sensing.range_response import (
        RangeResponseConfig,
        frequency_to_delay,
        paths_to_frequency,
    )

    config = RangeResponseConfig("cfr_ifft", 128, frequency_window)
    observation, _ = paths_to_frequency(
        np.ones((1, 1, 4), complex),
        np.array([2 * 4.55 / 299792458]),
        np.array([True]),
        np.arange(4) / 1000,
        bandwidth=500e6,
        config=config,
    )
    response, axes, _ = frequency_to_delay(
        observation, bandwidth=500e6, config=config, fft_size=fft_size, range_mode="monostatic"
    )
    power = abs(response[0, -1]) ** 2
    original = power.copy()
    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_range_profile(
        SimpleNamespace(power_linear=power, range_axis_m=axes["range_axis_m"]), ""
    )
    x, y = window.rp_samples.getData()
    np.testing.assert_array_equal(x, axes["range_axis_m"])
    np.testing.assert_allclose(y, 10 * np.log10(original))
    np.testing.assert_allclose(window.rp_curve.getData()[1], y)
    np.testing.assert_allclose(window.rp_plot.viewRange()[1], [y.max() - 80, y.max() + 3])
    window._profile_full_dynamic_range_cb.setChecked(True)
    assert window.rp_plot.viewRange()[1][0] < y.min()
    np.testing.assert_allclose(window.rp_curve.getData()[1], y)
    np.testing.assert_array_equal(power, original)
    assert x[np.argmax(y)] == pytest.approx(4.55, abs=np.diff(x)[0])
    assert "display floor" not in window.rp_plot.getPlotItem().titleLabel.text


def test_zero_profile_is_visible_and_nonuniform_sample_coordinates_are_exact(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    axis = np.array([0.0, 0.2, 0.8, 1.5])
    for power in (np.array([0.0, 1e-12, 0.0, 1e-14]), np.zeros(4)):
        window._update_range_profile(SimpleNamespace(power_linear=power, range_axis_m=axis), "")
        np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)
        if power.any():
            np.testing.assert_array_equal(window.rp_samples.getData()[0], axis[power > 0])
        else:
            assert np.isfinite(window.rp_curve.getData()[1]).all()
            assert window.rp_samples.getData()[0] is None
            assert "all bins zero" in window.rp_plot.getPlotItem().titleLabel.text


def test_range_profile_markers_follow_zoom_without_resampling(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window._set_view_mode("analysis")
    window.show()
    axis = np.arange(512) * 0.075
    power = np.exp(-axis) + 1e-20
    window._update_range_profile(SimpleNamespace(range_axis_m=axis, power_linear=power), "")
    window.rp_plot.setXLink(None)
    window.rp_plot.setXRange(0, axis[-1], padding=0)
    assert not window.rp_samples.isVisible()
    window.rp_plot.setXRange(4, 5, padding=0)
    assert window.rp_samples.isVisible()
    np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)
    np.testing.assert_allclose(window.rp_curve.getData()[1], 10 * np.log10(power))
    assert window.rp_curve.opts["stepMode"] is None


def test_saved_profile_identifies_last_sample_and_preserves_power(qtbot):
    from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
    from visualizer.src.services.sensing_service import SensingService

    sensing = native_cfr(profile=True)
    power = sensing["range_profile_by_pair"].copy()
    service = SensingService(SimpleNamespace())
    result = service._range_profile_from_validated(sensing, "primary")
    assert result.metadata["temporal_support"]["sample_index"] == 15
    assert result.metadata["time_span"]["reference_time_s"] == 0.015
    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_range_profile(result, "")
    assert "last sample" in window.rp_plot.getPlotItem().titleLabel.text
    assert "t = 0.015 s" in window.rp_plot.toolTip()
    assert "Range window: rectangular" in window.rp_plot.toolTip()
    assert "before Doppler processing" in window.rp_plot.toolTip()
    rd = service._range_doppler_from_validated(sensing, "primary")
    window._show_heatmaps_cb.setChecked(True)
    window._update_range_velocity(rd, {"sensing": sensing}, "", 28e9)
    assert window.rd_plot.getPlotItem().titleLabel.text == "Range–Velocity"
    assert "Range window: rectangular" in window.rd_plot.toolTip()
    assert "Doppler window: Hann" in window.rd_plot.toolTip()
    np.testing.assert_allclose(
        window.rd_image.image, (10 * np.log10(np.maximum(rd.power_linear, 1e-30))).T
    )
    np.testing.assert_array_equal(sensing["range_profile_by_pair"], power)
    np.testing.assert_array_equal(window.rp_curve.getData()[0], result.range_axis_m)


def test_scenario_view_bounds_preserve_profile_samples_and_full_range_access(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window._zoom_data_cb.setChecked(True)
    axis = np.arange(512) * 0.075
    power = np.ones((3, 512))
    window._update_range_profile(SimpleNamespace(power_linear=power[0], range_axis_m=axis), "")
    window._last_request = SimpleNamespace(
        display_frame={
            "sensing": {
                "config_resolved": {
                    "display_range_xlim": [3.5, 5.5],
                    "display_velocity_ylim": [-0.4, 2.3],
                }
            }
        }
    )
    window._apply_zoom_to_data(window.rd_plot, 10 * np.log10(power), axis, np.arange(3))
    np.testing.assert_allclose(window.rd_plot.viewRange()[0], [3.5, 5.5])
    np.testing.assert_allclose(window.rd_plot.viewRange()[1], [-0.4, 2.3])
    window._doppler_hz_cb.blockSignals(True)
    window._doppler_hz_cb.setChecked(True)
    window._apply_zoom_to_data(window.rd_plot, 10 * np.log10(power), axis, np.arange(3) * 100)
    assert window.rd_plot.viewRange()[1][1] > 100
    np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)
    window._zoom_data_cb.blockSignals(True)
    window._zoom_data_cb.setChecked(False)
    window._apply_zoom_to_data(window.rd_plot, 10 * np.log10(power), axis, np.arange(3))
    assert window.rd_plot.getViewBox().autoRangeEnabled() == [False, True]
    np.testing.assert_allclose(window.rd_plot.viewRange()[0], [0, axis[-1] + 0.075 / 2])
    np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)


def test_full_profile_span_is_view_only_and_control_follows_profile(qtbot, monkeypatch):
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    window.show()
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    axis = np.arange(5, dtype=float)
    power = 10 ** (np.array([-290, -140, -90, -150, -230], dtype=float) / 10)
    window._update_range_profile(SimpleNamespace(power_linear=power, range_axis_m=axis), "")
    window.rd_plot.setXRange(0, 4, padding=0)
    before = tuple(a.copy() for a in window.rp_curve.getData())
    map_limits = window.rd_plot.viewRange()
    reloads = []
    window.view_changed.connect(lambda: reloads.append(True))
    monkeypatch.setattr(window, "_rerender_last_request", lambda: pytest.fail("unexpected replay"))
    assert window._advanced_widget.isVisible()
    assert window._profile_full_dynamic_range_cb.isVisible()
    window._profile_full_dynamic_range_cb.click()
    assert window.rp_plot.viewRange()[1][0] < -290
    window._profile_full_dynamic_range_cb.click()
    np.testing.assert_allclose(window.rp_plot.viewRange()[1], [-170, -87])
    np.testing.assert_allclose(window.rd_plot.viewRange(), map_limits)
    for actual, expected in zip(window.rp_curve.getData(), before):
        np.testing.assert_array_equal(actual, expected)
    assert not reloads
    monkeypatch.undo()
    window._set_view_mode("summary")
    assert not window._profile_full_dynamic_range_cb.isVisible()


@pytest.mark.parametrize("axis", [np.arange(512) * 0.075, np.array([0.0, 0.2, 0.8, 1.5])])
def test_full_range_excludes_overlay_padding_and_keeps_linked_samples(qtbot, axis):
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    window.show()
    power_db = np.linspace(-290, -90, axis.size)
    window._update_range_profile(
        SimpleNamespace(power_linear=10 ** (power_db / 10), range_axis_m=axis), ""
    )
    window.rd_det_scatter.setData([axis[-1] + 50], [0])
    window._zoom_data_cb.setChecked(False)
    window._apply_zoom_to_data(window.rd_plot, power_db[None, :], axis, np.array([0.0]))
    QApplication.processEvents()
    right = axis[-1] + (axis[-1] - axis[-2]) / 2
    np.testing.assert_allclose(window.rd_plot.viewRange()[0], [0, right])
    np.testing.assert_allclose(window.rp_plot.viewRange()[0], [0, right])
    np.testing.assert_array_equal(window.rp_curve.getData()[0], axis)
    np.testing.assert_allclose(window.rp_curve.getData()[1], power_db)


@pytest.mark.parametrize(
    "sensing, override, expected",
    [
        ({"enabled": True}, None, True),
        ({"enabled": False}, None, False),
        ({}, None, False),
        ({"enabled": True}, {"enabled": False}, False),
        ({"enabled": True}, False, False),
        ({"enabled": True}, {"enabled": True}, True),
    ],
)
def test_sensing_panel_infers_scenario_role_and_honors_override(sensing, override, expected):
    from types import SimpleNamespace

    from visualizer.src.sensing.runtime import _panel_enabled

    panels = {} if override is None else {"sensing": override}
    viz = SimpleNamespace(
        scenario_config=SimpleNamespace(sensing=sensing, visualizer_cfg={"panels": panels})
    )
    assert _panel_enabled(viz) is expected


def test_export_availability_tracks_accepted_request_after_unavailable():
    sensing = _with_detection(_typed_sensing())
    bundle = build_detection_stage_bundle(sensing)
    request = _request(sensing, bundle, _replay_event_for(bundle))
    window = SensingWindow()
    try:
        window.show()
        window._set_view_mode("diagnostics")
        window.set_unavailable("No result", scene_frame_id=0)
        assert not window.detection_stage_export_button.isEnabled()
        window.update_from_request(request)
        assert window.detection_stage_export_button.isEnabled()
        window.set_unavailable("No result", scene_frame_id=1)
        assert not window.detection_stage_export_button.isEnabled()
        window.update_from_request(replace(request, replay_event=None, detection_stage_bundle=None))
        assert not window.detection_stage_export_button.isEnabled()
    finally:
        window.close()
