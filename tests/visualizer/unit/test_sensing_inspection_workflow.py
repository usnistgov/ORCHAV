"""Exact-bin inspection semantics, bounded reads and shared display contracts."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from tests.visualizer.unit.test_sensing_processing_inspection import signal_recording
from tests.visualizer.unit.test_sensing_stage_inspector import _with_detection
from visualizer.src.sensing.dashboard_diagnostics import (
    cell_explanation,
    stage_counts,
    threshold_margin,
)
from visualizer.src.sensing.inspection_display import PowerScale
from visualizer.src.sensing.processing_inspection import ProcessingInspection, inspect_processing
from visualizer.src.sensing.processing_panel import ProcessingPanel
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.spectrum_surface import SpectrumSurface
from visualizer.src.sensing.stage_inspector import CellSelection, build_detection_stage_bundle
from visualizer.src.services.sensing_service import SensingService


@pytest.mark.parametrize(
    "p,t,eligible,saved,value,note",
    [
        (100, 10, True, True, 10, "linear"),
        (10, 10, True, False, 0, "equality"),
        (10, 10, True, True, 0, "CONSISTENCY ERROR"),
        (1e300, 1e-300, True, True, 6000, "linear"),
        (1, 0, False, False, None, "not tested"),
        (1, 0, None, None, None, "eligibility unavailable"),
        (1, 0, True, True, np.inf, "zero threshold"),
        (0, 1, True, False, -np.inf, "zero input"),
        (0, 0, True, False, None, "undefined"),
        (np.nan, 1, True, False, None, "invalid"),
        (np.inf, 1, True, True, None, "invalid"),
        (-1, 1, True, False, None, "invalid"),
        (1, None, True, None, None, "not recorded"),
    ],
)
def test_threshold_margin_is_not_a_display_floor(p, t, eligible, saved, value, note):
    actual = threshold_margin(p, t, eligible, saved)
    assert actual.value_db == value
    assert note in actual.note


def test_scoped_counts_and_cluster_member_is_not_representative():
    bundle = build_detection_stage_bundle(_with_detection(_typed_sensing()))
    summary = stage_counts(bundle, 1)
    assert "winning-beam projection" in summary and "Published reports in this result: 1" in summary
    cell = bundle.inspect(CellSelection(bundle.event.pair_id, 1, 2))
    # Keep authoritative cluster link, inspect a different member coordinate.
    member = replace(
        cell,
        selection=CellSelection(bundle.event.pair_id, 1, 1),
        cfar_eligible=True,
        cfar_passed=True,
        nms_retained=True,
    )
    assert "not a published representative" in cell_explanation(member)


def test_auto_common_reference_lock_visibility_and_incompatible_scope():
    scale = PowerScale()
    before, after = np.array([1.0, 100]), np.array([1.0, 10])
    assert scale.levels((before, after), identity=("one",), scope=("same",), span=60) == (-40, 20)
    scale.locked = True
    assert scale.levels((after,), identity=("two",), scope=("same",), span=40) == (-20, 20)
    assert scale.levels((after,), identity=("two",), scope=("other",), span=40) == (-30, 10)
    scale.locked = False
    assert scale.levels((before,), identity=("three",), scope=("other",), span=40) == (-20, 20)


@pytest.mark.parametrize("kind", ["clutter", "inputs"])
def test_authoritative_path_coordinate_and_signal_workspace(qtbot, kind):
    sensing = signal_recording(raw=True)
    sensing["config_resolved"]["range_mode"] = "bistatic"
    model = inspect_processing(sensing, kind)
    assert model.range_label == "Total path length"
    np.testing.assert_array_equal(model.ranges, sensing["ranges"])
    window = SensingWindow()
    qtbot.addWidget(window)
    window._set_view_mode("diagnostics")
    window._support_box.setCurrentIndex(window._support_box.findData(kind))
    assert window.view_description.isHidden()
    assert window.workspace_box.currentData() == "current"
    assert window.workspace_tabs.tabText(window.workspace_tabs.currentIndex()) == (
        "Detector" if kind == "clutter" else "Inputs"
    )
    window.close()


def test_detection_coordinate_continuity_and_source_recipe_invalidation(qtbot):
    request = replace(_request(), source_identity=("recording-A",))
    sensing = request.display_frame["sensing"]
    sensing["detections"] = [
        dict(range_m=1.4, velocity_m_s=0, doppler_hz=0, range_bin=1, doppler_bin=2)
    ]
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window.update_from_request(request)
    window.select_detection_index(0)
    assert window._linked_selection.cell.range_bin == 1  # representative, not centroid 1.4
    before = np.ones((4, 3))
    before[:, 2] = 100  # strongest bin is a different return
    model = ProcessingInspection(
        "clutter",
        ranges=np.arange(3.0),
        doppler=np.array([-2.0, -1.0, 0.0, 1.0]),
        before=before,
        after=before / 2,
        identity=("model",),
    )
    window._set_view_mode("diagnostics")
    window._support_box.setCurrentIndex(window._support_box.findData("clutter"))
    request = replace(request, processing_inspection=model)
    window.update_from_request(request)
    assert window.processing_panel.range_bin.value() == 1
    window.processing_panel.surface_toggle.setChecked(True)
    assert window.processing_panel.surface._data[2] == (2, 1)
    window.processing_panel.range_bin.setValue(0)
    assert (
        window._linked_selection.cell.range_bin == 0
        and window._linked_selection.cell.origin == "cell"
    )
    invalid = deepcopy(request.display_frame)
    invalid["sensing"]["temporal"]["result"]["processing_recipe_id"] = "changed"
    # Check linking independently: this intentionally malformed changed contract
    # must not be accepted as a real update by the strict reader.
    window._last_request = replace(request, display_frame=invalid)
    window._sync_selected_stage_cell(window._linked_selection)
    assert window.processing_panel._blocked_selection
    assert window.processing_panel.before_image.image is None
    window._last_request = replace(request, source_identity=("recording-B",))
    window.apply_linked_selection()
    assert window._linked_selection is None
    window.close()


def test_processing_maps_toggle_cell_and_clear_companion_slices(qtbot):
    from PySide6.QtCore import QPointF, Qt

    request = replace(_request(), source_identity=("recording",))
    model = ProcessingInspection(
        "clutter",
        ranges=request.rd_result.range_axis_m,
        doppler=request.rd_result.doppler_axis_hz,
        before=np.ones((4, 3)),
        after=np.ones((4, 3)) / 2,
        identity=("model",),
    )
    request = replace(request, processing_inspection=model)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("diagnostics")
    window._support_box.setCurrentIndex(window._support_box.findData("clutter"))
    window.update_from_request(request)
    panel = window.processing_panel

    def click(plot):
        qtbot.wait(1)
        pos = plot.getViewBox().mapViewToScene(QPointF(model.ranges[1], model.doppler[2]))
        panel._select_range(
            SimpleNamespace(
                button=lambda: Qt.MouseButton.LeftButton,
                scenePos=lambda: pos,
                accept=lambda: None,
            ),
            plot,
        )

    click(panel.before_plot)
    assert window._linked_selection.cell.range_bin == 1
    assert panel.before_curve.getData()[0] is not None
    click(panel.after_plot)
    assert window._linked_selection is None
    for redraw in (lambda: None, lambda: window.update_from_request(request)):
        redraw()
        assert panel.before_curve.getData()[0] is None
        assert panel.after_curve.getData()[0] is None
        assert panel.coordinate.text() == "No cell selected"
        assert all(not line.isVisible() for line in panel._lines)
        assert panel.before_image.image is not None and panel.after_image.image is not None
    panel.range_bin.setValue(2)
    assert window._linked_selection.cell.range_bin == 2
    assert panel.before_curve.getData()[0] is not None
    assert all(line.isVisible() for line in panel._lines)
    window.close()


def test_probe_never_transplants_indices_to_changed_grid(qtbot):
    from visualizer.src.sensing.selection import inspection_cell

    req = _request()
    cell = inspection_cell(req, 1, 2)
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    model = ProcessingInspection(
        "clutter",
        ranges=np.array([0.0, 2.0, 4.0]),
        doppler=req.rd_result.doppler_axis_hz,
        after=np.ones((4, 3)),
    )
    panel.set_model(model, "clutter")
    panel.apply_cell(cell, explicit=True)
    assert panel._blocked_selection and panel.after_image.image is None
    panel.range_bin.setValue(2)
    assert not panel._blocked_selection
    panel.release()
    panel.close()


def test_surface_controls_keep_scale_probe_and_holes(qtbot):
    surface = SpectrumSurface()
    qtbot.addWidget(surface)
    p = np.ones((3, 4))
    p[1, 2] = 100
    t = p / 10
    eligible = np.ones_like(p, dtype=bool)
    eligible[1, 2] = False
    surface.set_spectra(
        np.array([0.0, 1.0, 3.0, 8.0]),
        np.array([-2.0, 0.0, 3.0]),
        p,
        t,
        selected=(1, 2),
        labels=("Input", "Threshold"),
        threshold=True,
        eligible=eligible,
        levels=(-30, 20),
        range_label="Total path length",
    )
    assert surface.axes.get_xlabel() == "Total path length (m)"
    assert "not tested" in surface.last_probe[1]
    assert surface.caption.isHidden()
    assert "not tested" in surface.canvas.toolTip()
    surface.preset.setCurrentText("Top-down")
    assert surface.axes.elev == 90
    surface.layers[0].setChecked(False)
    surface.layers[1].setChecked(False)
    assert "Both layers hidden" in surface.caption.text()
    assert not surface.caption.isHidden()
    assert surface.axes.get_zlim() == (-30, 21)
    surface.reset_button.click()
    assert surface.axes.elev == 30
    surface.layers[0].setChecked(True)
    assert surface._sources[0] is p
    assert surface.caption.isHidden()
    surface.clear("Waiting for the current saved spectrum")
    assert not surface.caption.isHidden()
    assert surface.canvas.toolTip() == "Waiting for the current saved spectrum"
    surface.release()
    surface.close()
    surface.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_singleton_surface_falls_back_to_slice(qtbot):
    surface = SpectrumSurface()
    qtbot.addWidget(surface)
    surface.set_spectra(
        np.arange(3.0),
        np.array([0.0]),
        np.ones((1, 3)),
        None,
        selected=(0, 2),
        labels=("Input", ""),
    )
    assert surface.canvas is None and "1D slice" in surface.caption.text()
    assert not surface.caption.isHidden()


def test_oversized_input_declined_before_provider_dense_read(monkeypatch):
    reads = []
    provider = SimpleNamespace(load_frame_projection=lambda *args: reads.append(args))
    monkeypatch.setattr(
        "visualizer.src.services.sensing_service.visualizer_frame_provider", lambda _: provider
    )
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    service._sensing_metadata_index[(int, 0)] = {
        "sensing": {
            "products": {
                "samples": {
                    "kind": "delay_domain_slow_time",
                    "shape": [1, 64, 1024, 1024],
                    "dtype": "<c16",
                }
            }
        }
    }
    with pytest.raises(ValueError, match="before allocation"):
        service._load_sensing_candidate(0, ("delay_domain_slow_time",))
    assert reads == []


def test_cfar_request_uses_provider_stage_projection_without_cir_or_angles(monkeypatch):
    from tests.visualizer.unit.test_services_sensing import (
        _CompleteFrameOnlyProvider,
        _detector_stage_sensing,
        build_standard_mpc_frame,
    )
    from visualizer.src.sensing.dashboard_data import load_dashboard_request

    sensing = _detector_stage_sensing(scene_frame_id=11, sample_time_s=1.1)
    frame = replace(build_standard_mpc_frame(frame_idx=11), sensing=sensing)
    provider = _CompleteFrameOnlyProvider(frame)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    # Test the ordinary dashboard route, including service descriptor and stage reads.
    monkeypatch.setattr(
        service,
        "dashboard_scene_frame",
        lambda _: {
            "sensing": {},
            "_source": {"timestamp": 1.1},
            "rx_positions": np.zeros((1, 3)),
            "tx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "tx_orientations": np.zeros((1, 3)),
        },
    )
    request = load_dashboard_request(service, 11, "diagnostics", "primary", False, "auto")
    assert request.detection_stage_bundle.available
    assert all(r.sensing_products for r in provider.requests if r.components)
    requested = set().union(*(r.sensing_products for r in provider.requests))
    assert "range_doppler_cfar_mask_by_pair" in requested
    assert "slow_time_by_pair_rx_element" not in requested
    assert "range_doppler_angle_power_by_pair" not in requested


def test_coasting_track_cannot_supply_a_new_cell(qtbot):
    request = _request()
    sensing = request.display_frame["sensing"]
    sensing["detections"] = [
        dict(track_id=9, range_m=1, velocity_m_s=0, doppler_hz=0, range_bin=1, doppler_bin=2)
    ]
    window = SensingWindow()
    qtbot.addWidget(window)
    window._initial_view_selected = True
    window.update_from_request(request)
    window.select_detection_index(0)
    model = ProcessingInspection(
        "clutter",
        ranges=request.rd_result.range_axis_m,
        doppler=request.rd_result.doppler_axis_hz,
        after=np.ones((4, 3)),
    )
    window.processing_panel.set_model(model, "clutter")
    window._last_request = replace(_request(1), processing_inspection=model)
    window._sync_selected_stage_cell(window._linked_selection)
    assert window.processing_panel._blocked_selection
    assert window.processing_panel.after_image.image is None
    assert window._linked_selection.track_id == 9
    window.close()


def test_input_magnitude_does_not_inherit_db_limits(qtbot):
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    sensing = signal_recording(raw=True)
    panel.set_model(inspect_processing(sensing, "clutter"), "clutter")
    panel.set_model(inspect_processing(sensing, "inputs"), "inputs")
    assert panel.before_curve.yData.min() > 0
    low, high = panel.slice.viewRange()[1]
    assert low == 0
    assert panel.before_curve.yData.max() < high < 1.2 * panel.before_curve.yData.max()
    panel.release()
    panel.close()


@pytest.mark.parametrize("tracked", [False, True])
def test_raw_cell_retires_at_next_acquisition_but_track_identity_follows(qtbot, tracked):
    """A new result must not display a previous acquisition's coordinate probe."""
    first, second = _request(), _request(1)
    for request in (first, second):
        request.display_frame["sensing"]["detections"] = [
            dict(track_id=9, range_m=1, velocity_m_s=0, doppler_hz=0, range_bin=1, doppler_bin=2)
        ]
    window = SensingWindow()
    qtbot.addWidget(window)
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window.update_from_request(first)
    if tracked:
        window.select_track(9)
    else:
        window._select_inspection_coordinate(0, 2)
    selected = window._linked_selection
    assert selected is not None
    assert (
        selected.acquisition_id
        != second.display_frame["sensing"]["temporal"]["acquisition"]["acquisition_id"]
    )
    cleared = []
    window.selection_changed.connect(cleared.append)
    window.update_from_request(second)
    if tracked:
        assert window._linked_selection.track_id == 9
        assert "Track 9" in window.selection_label.text()
        assert len(window._selected_rd_cells.points()) == 1
        assert cleared == []
    else:
        assert window._linked_selection is None
        assert len(window._selected_rd_cells.points()) == 0
        assert window.selection_label.text() == "No selection"
        assert not window.locate_selection_button.isEnabled()
        assert cleared == [None]
    window.close()
