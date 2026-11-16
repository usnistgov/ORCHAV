"""Saved-product dashboard regressions: visit order never becomes a signal clock."""

import subprocess
import sys
from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QApplication

from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from visualizer.src.app.theme import get_theme_manager
from visualizer.src.sensing.dashboard_data import (
    DashboardSession,
    HistoryRecord,
    RecordingHistory,
    history_cells,
    load_dashboard_request,
    load_recording_history,
    result_context,
)
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.update_request import SensingUpdateRequest
from visualizer.src.services.sensing_service import SensingService

_APP = QApplication.instance() or QApplication([])


def _frame(step=0):
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "range_doppler_power_by_pair": np.arange(1.0, 13.0).reshape(1, 4, 3),
                "config_resolved": {
                    "track_confirmation_enabled": False,
                    "processing_mode": "coherent_cpi",
                    "acquisition": {
                        "sample_source": "coherent_one_solve",
                        "sample_period_s": 0.0625,
                        "sample_rate_hz": 16.0,
                        "phase_contract": "analytic_bulk_motion",
                        "cpi_length_samples": 4,
                        "cpi_hop_samples": 4,
                        "rt_solve_stride_scene_steps": 4,
                    },
                    "processing": {
                        "mode": "coherent_cpi",
                        "strategy": "coherent_cpi_fft",
                        "result_stride_samples": 4,
                        "product_semantics": "coherent_slow_time_fft",
                        "evidence_status": "supported",
                    },
                },
                "processing_strategy": "coherent_cpi_fft",
                "range_axis_m": np.arange(3.0),
                "doppler_axis_hz": np.array([-2.0, -1.0, 0.0, 1.0]),
            },
            scene_frame_id=step,
            sample_time_s=step * 0.25,
        )
    }
    return frame


def _request(step=0, history=None):
    frame = _frame(step)
    cfg = frame["sensing"]["config_resolved"]
    service = SensingService(SimpleNamespace())
    rd = service.compute_range_doppler(
        frame=frame,
        sampling_frequency=cfg["sampling_frequency"],
        prf_hz=cfg["prf_hz"],
        variant="primary",
    )
    return SensingUpdateRequest(
        display_frame=frame,
        state_frame=frame,
        state_step=step,
        rd_result=rd,
        variant="primary",
        recording_managed=True,
        recording_history=history,
    )


def _history():
    return RecordingHistory(
        tuple(
            HistoryRecord(
                time_s=i * 0.25,
                identity=("clock", "run", "pair", str(i)),
                spectrum=np.array([1.0, 3.0, 9.0, 2.0]) * (i + 1),
                doppler_hz=np.array([-2.0, -1.0, 0.0, 1.0]),
                range_m=None,
                range_power=None,
                detections=(),
                truth=None,
                truth_positions=(),
                cadence_s=0.25,
            )
            for i in range(6)
        )
    )


@pytest.fixture
def window():
    w = SensingWindow()
    w.show()
    w._set_view_mode("micro-doppler")
    yield w
    w.close()


def test_recording_spectrum_ignores_seek_order_and_retains_saved_power(window, monkeypatch):
    history = _history()

    def unexpected(*args, **kwargs):
        raise AssertionError("Saved RD overview must not recompute an FFT or hidden angular view")

    monkeypatch.setattr(np.fft, "fft", unexpected)
    monkeypatch.setattr(window, "_update_angular_products", unexpected)
    for step in (5, 1, 4, 0, 5):
        window.update_from_request(_request(step, history))
        np.testing.assert_allclose(
            window.collapsed_doppler_image.image,
            10 * np.log10(np.stack([r.spectrum for r in history.records])),
        )
        assert window.collapsed_doppler_cursor.value() == step * 0.25
    # No repeated image upload when only the selected time changes.
    monkeypatch.setattr(window.collapsed_doppler_image, "setImage", unexpected)
    window.update_from_request(_request(2, history))


def test_advanced_motion_returns_from_gate_to_full_recording(window):
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    request = _request(2, _history())
    request.display_frame["sensing"]["detections"] = [{"range_m": 1.0, "power_linear": 10.0}]
    window.update_from_request(request)
    overview = window.collapsed_doppler_image.image.copy()
    saved_power = request.rd_result.power_linear.copy()
    assert window._advanced_motion_cb.isVisible()
    assert not window.motion_controls.isVisible()

    window._advanced_motion_cb.click()
    assert window.motion_controls.isVisible()
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("gated"))
    window._micro_gate_box.setValue(0)
    assert window.micro_plot.isVisible()
    np.testing.assert_allclose(window.micro_image.image, 10 * np.log10(saved_power[:, 1])[None, :])

    # Changing workspaces retains the deliberate gate and exposes its controls.
    window._set_view_mode("analysis")
    assert not window._advanced_motion_cb.isVisible()
    assert not window.motion_controls.isVisible()
    window._set_view_mode("micro-doppler")
    assert window.motion_controls.isVisible()
    assert window._selected_micro_doppler_mode() == "gated"

    # Hiding advanced choices restores all results, not a hidden single-result gate.
    window._advanced_motion_cb.click()
    assert not window.motion_controls.isVisible()
    assert window._selected_micro_doppler_mode() == "auto"
    assert window.collapsed_doppler_plot.isVisible()
    assert not window.micro_plot.isVisible()
    np.testing.assert_array_equal(window.collapsed_doppler_image.image, overview)
    np.testing.assert_array_equal(request.rd_result.power_linear, saved_power)
    assert window.collapsed_doppler_cursor.value() == 0.5


@pytest.mark.parametrize("mode", ["gated", "fixed-bin"])
def test_motion_preset_exposes_gate_and_new_recording_resets_it(window, mode):
    with QSignalBlocker(window.figure_options):
        window.figure_options.expand()
    window.apply_display_preset({"view_mode": "micro-doppler", "micro_doppler_mode": mode})
    assert window._advanced_motion_cb.isChecked()
    assert window.motion_controls.isVisible()
    assert window._selected_micro_doppler_mode() == mode
    window.reset_recording(preserve_options=True)
    assert window._selected_micro_doppler_mode() == mode
    assert window._advanced_motion_cb.isChecked()
    window.reset_recording()
    window.update_from_request(_request(2, _history()))
    window._set_view_mode("micro-doppler")
    assert not window._advanced_motion_cb.isChecked()
    assert not window.motion_controls.isVisible()
    assert window._selected_micro_doppler_mode() == "auto"


def test_recording_load_uses_saved_results_not_cir(monkeypatch):
    frames = [_frame(0), _frame(2), _frame(5)]
    service = SensingService(SimpleNamespace())
    metadata = [service._metadata_only_sensing_frame(frame) for frame in frames]
    monkeypatch.setattr(service, "dashboard_result_rows", lambda: list(enumerate(metadata)))
    requests = []

    def load(index, products):
        requests.append(products)
        return frames[index]

    monkeypatch.setattr(service, "dashboard_load_row", load)
    monkeypatch.setattr(service, "dashboard_product_kinds", lambda _: frozenset({"range_doppler"}))
    monkeypatch.setattr(service, "dashboard_scene_frame", lambda _: {})
    monkeypatch.setattr(service, "dashboard_world_detections", lambda _: [])
    history = load_recording_history(
        service, result_context(frames[0]["sensing"]), "primary", Event()
    )
    assert [r.time_s for r in history.records] == [0, 0.5, 1.25]
    for record in history.records:
        np.testing.assert_array_equal(record.spectrum, [6.0, 15.0, 24.0, 33.0])
        np.testing.assert_array_equal(record.doppler_hz, [-2.0, -1.0, 0.0, 1.0])
    assert all(set(p) == {"range_doppler"} for p in requests)


def test_history_budget_keeps_spectra_without_copying_sample_provenance(monkeypatch):
    from copy import deepcopy

    from shared.cache_sizing import estimate_retained_bytes
    from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr

    sensing = native_cfr(samples=64, detection=True, profile=True, recording="reprocessing")
    service = SensingService(SimpleNamespace())
    rp, rd = service.dashboard_products({"sensing": sensing}, "primary")
    frames = []
    for step in range(8):
        payload = deepcopy(sensing)
        payload["temporal"]["acquisition"]["acquisition_id"] = f"acquisition-{step}"
        payload["temporal"]["result"]["reference_time_s"] = step * 0.064
        payload["temporal"]["display_binding"]["scene_frame_id"] = step
        frames.append({"sensing": payload})
    monkeypatch.setattr(service, "dashboard_result_rows", lambda: list(enumerate(frames)))
    monkeypatch.setattr(
        service, "dashboard_history_products", lambda step, _: (frames[step], rp, rd)
    )
    monkeypatch.setattr(service, "dashboard_scene_frame", lambda _: {})
    monkeypatch.setattr(
        service, "dashboard_world_detections", lambda frame: frame["sensing"]["detections"]
    )
    from visualizer.src.sensing.cache_policy import SensingCachePolicy

    policy = SensingCachePolicy(history_bytes=256 * 1024)
    context = result_context(sensing)
    history = load_recording_history(service, context, "primary", Event(), cache_policy=policy)
    assert not history.truncated
    assert len(history.records) == len(frames)
    assert estimate_retained_bytes(history.records) < policy.history_bytes
    for step, record in enumerate(history.records):
        assert record.time_s == step * 0.064
        assert record.scene_frame_id == step
        np.testing.assert_array_equal(record.spectrum, np.sum(rd.power_linear, axis=-1))
        np.testing.assert_array_equal(record.range_power, rp.power_linear)
        assert len(frames[step]["sensing"]["temporal"]["acquisition"]["samples"]) == 64

    # Compaction must preserve the cap for genuinely larger overviews.
    budget = 2 * max(estimate_retained_bytes(record) for record in history.records)
    bounded = load_recording_history(
        service, context, "primary", Event(), cache_policy=replace(policy, history_bytes=budget)
    )
    assert bounded.truncated
    assert 0 < len(bounded.records) < len(frames)
    assert sum(estimate_retained_bytes(record) for record in bounded.records) <= budget
    assert bounded.records[-1].time_s == history.records[-1].time_s


def test_history_gaps_are_empty_not_interpolated():
    edges, cells = history_cells(
        np.array([0.0, 1.0, 3.0]), np.ones(3), np.array([[1.0], [2.0], [4.0]])
    )
    np.testing.assert_array_equal(edges, [-0.5, 0.5, 1.5, 2.5, 3.5])
    assert np.isnan(cells[2, 0])
    np.testing.assert_array_equal(cells[[0, 1, 3], 0], [1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="increase"):
        history_cells(np.array([1.0, 1.0]), np.ones(2), np.ones((2, 1)))


def test_combined_products_revalidate_mutated_caller_frames():
    """Optimization must not leave a cached validation stamp on mutable input."""
    service = SensingService(SimpleNamespace())
    frame = _frame()
    _, rd = service.dashboard_products(frame, "primary")
    np.testing.assert_array_equal(rd.power_linear, np.arange(1.0, 13.0).reshape(4, 3))
    frame["sensing"]["range_doppler_power_by_pair"][0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        service.dashboard_products(frame, "primary")


def test_history_product_read_still_validates_provider_metadata():
    from tests.visualizer.unit.test_sensing_panel_fallback import _PanelFrameProvider

    frame = _frame()
    provider = _PanelFrameProvider({0: frame["sensing"]})
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    loaded, _, rd = service.dashboard_history_products(0, "primary")
    assert loaded is not None and rd is not None
    provider.frames[0].sensing["temporal"]["acquisition"]["acquisition_id"] = "bad-identity"
    with pytest.raises(ValueError, match="acquisition_id"):
        service.dashboard_history_products(0, "primary")


def test_combined_products_do_not_substitute_a_missing_variant():
    service = SensingService(SimpleNamespace())
    assert service.dashboard_products(_frame(), "raw") == (None, None)


def test_motion_cursor_reuses_exact_history_without_product_io(window, monkeypatch):
    from tests.visualizer.unit.test_sensing_panel_fallback import _PanelFrameProvider

    provider = _PanelFrameProvider({i: _frame(i)["sensing"] for i in (0, 2, 5)})
    session = DashboardSession(provider)
    try:
        history = load_recording_history(
            session.service, result_context(_frame()["sensing"]), "primary", Event()
        )
        session.history = history
        session.history_inventory = session.service.dashboard_inventory_token()
        request = _request(0, history)
        window.update_from_request(request)
        assert window.can_move_recording_cursor(history)
        image = window.collapsed_doppler_mesh.z.copy()

        def unexpected(*args, **kwargs):
            raise AssertionError("Cached cursor must not read or re-upload products")

        monkeypatch.setattr(provider, "load_frame_projection", unexpected)
        monkeypatch.setattr(provider, "iter_frame_projections", unexpected)
        monkeypatch.setattr(window.collapsed_doppler_mesh, "setData", unexpected)
        for step in (5, 0, 2):
            record = session.motion_cursor(step)
            assert record is not None and record.scene_frame_id == step
            window.move_recording_cursor(record, "primary")
            assert window.result_time_s() == step * 0.25
        assert window._last_request is request  # No synthetic replacement contract.
        np.testing.assert_array_equal(window.collapsed_doppler_mesh.z, image)
        assert session.motion_cursor(1) is None  # No nearest-frame substitution.
        provider.frames[7] = provider.frames[5]
        assert session.motion_cursor(5) is None  # Growing inventories need the worker.
        window._table_requested = True
        assert window.can_move_recording_cursor(history)
        # The shared browser uses compact rows and does not force dense product reads.
        window.move_recording_cursor(history.records[-1], "primary")
        assert window.result_browser.context.text().endswith("1.250 s")
        window.set_unavailable("No result", scene_frame_id=1)
        assert window.result_time_s() == 0.0
    finally:
        session.close()


def test_motion_cursor_does_not_substitute_another_processing_run(monkeypatch):
    service = SensingService(SimpleNamespace())
    metadata = service._metadata_only_sensing_frame(_frame())
    from copy import deepcopy

    newer = deepcopy(metadata)
    newer["sensing"]["temporal"]["result"]["processing_run_id"] = "zzzz"
    service._sensing_index_token = ("inventory",)
    service._sensing_result_index = ((0, metadata), (8, newer))
    monkeypatch.setattr(service, "_current_metadata_index_token", lambda _: ("inventory",))
    # The higher-ranked result wins even if the caller's history has only row 0.
    assert service.dashboard_exact_result_id(0) == 8


def test_pinned_motion_cursor_requires_the_same_physical_time_winner(monkeypatch):
    service = SensingService(SimpleNamespace())
    rows = tuple((step, service._metadata_only_sensing_frame(_frame(step))) for step in (0, 2, 5))
    service._sensing_index_token = ("inventory",)
    service._sensing_result_index = rows
    monkeypatch.setattr(service, "_current_metadata_index_token", lambda _: ("inventory",))
    assert service.dashboard_exact_result_id(2, pin_freshest=True, scene_time_s=0.5) == 2
    # An exact binding alone cannot override the actual pinned-time policy.
    assert service.dashboard_exact_result_id(2, pin_freshest=True, scene_time_s=0.4) is None
    assert service.dashboard_exact_result_id(2, pin_freshest=True, scene_time_s=1.25) is None
    assert service.dashboard_exact_result_id(2, pin_freshest=True) is None
    assert service.dashboard_exact_result_id(2, pin_freshest=True, scene_time_s=np.nan) is None
    assert service.dashboard_exact_result_id(1, pin_freshest=True, scene_time_s=0.25) is None
    monkeypatch.setattr(service, "_current_metadata_index_token", lambda _: ("changed",))
    assert service.dashboard_exact_result_id(2, pin_freshest=True, scene_time_s=0.5) is None


def test_sensing_only_motion_playback_uses_prepared_cursor(window, monkeypatch):
    from tests.visualizer.unit.test_sensing_panel_fallback import _PanelFrameProvider
    from visualizer.src.panels import sensing_panel

    provider = _PanelFrameProvider({i: _frame(i)["sensing"] for i in (0, 2, 5)})
    parent = SimpleNamespace(
        animation_running=True,
        sensing_playback=SimpleNamespace(timestamps={i: i * 0.25 for i in (0, 2, 5)}),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._sensing_window = window
    session = panel._dashboard_session = DashboardSession(provider)
    try:
        context = result_context(_frame()["sensing"])
        history = load_recording_history(session.service, context, "primary", Event())
        session.history = history
        session.history_key = (*context, "primary")
        session.history_inventory = session.service.dashboard_inventory_token()
        window.update_from_request(_request(0, history))

        def unexpected(*args, **kwargs):
            raise AssertionError("Prepared Motion playback must not request a full CPI")

        monkeypatch.setattr(session, "request", unexpected)
        monkeypatch.setattr(panel, "_poll_dashboard", unexpected)
        monkeypatch.setattr(panel, "_update_result_navigation", lambda: None)
        monkeypatch.setattr(provider, "load_frame_projection", unexpected)
        monkeypatch.setattr(provider, "iter_frame_projections", unexpected)
        for step in (5, 0, 2):
            panel._refresh_dashboard_async(provider, step, force=False)
            assert window.result_time_s() == step * 0.25
            assert panel._completed_dashboard_key[0] == step
    finally:
        panel._retire_dashboard_session()


def test_error_or_source_change_clears_previous_recording(window):
    window.update_from_request(_request(5, _history()))
    assert window.collapsed_doppler_image.image is not None
    window.update_from_request(_request(5, RecordingHistory((), error="bad product")))
    assert window.collapsed_doppler_image.image is None
    assert not window.collapsed_doppler_mesh.isVisible()
    window.reset_recording()
    assert window._last_request is None
    assert not window._plots_widget.isVisible()


def test_optional_controls_stay_hidden_until_requested(window):
    window._toggle_detection_table(False)
    assert not window.detection_stage_inspector.isVisible()
    assert not window.detection_table.isVisible()
    window._set_view_mode("diagnostics")
    assert window.detection_stage_inspector.isVisible()
    window._set_view_mode("tracking")
    assert not window.detection_stage_inspector.isVisible()
    assert not window.detection_table.isVisible()
    window._toggle_detection_table(True)
    assert window.detection_table.isVisible()
    window._set_view_mode("micro-doppler")
    assert window.detection_table.isVisible()
    assert window.table_button.isVisible()


def test_incompatible_doppler_axes_cannot_leave_previous_image(window):
    history = _history()
    window.update_from_request(_request(5, history))
    records = list(history.records)
    records[-1] = replace(records[-1], doppler_hz=np.arange(4.0))
    window.update_from_request(_request(5, RecordingHistory(tuple(records))))
    assert window.collapsed_doppler_image.image is None
    assert "Incompatible" in window.collapsed_doppler_plot.plotItem.titleLabel.text


def test_theme_switch_and_gt_toggle_do_not_change_power(window):
    manager = get_theme_manager()
    original = manager.mode
    window.update_from_request(_request(5, _history()))
    image = window.collapsed_doppler_image.image.copy()
    try:
        for mode in ("dark", "light"):
            manager.set_mode(mode, persist=False)
            assert (
                window.rd_plot.backgroundBrush().color().name()
                == manager.current_theme.bg_secondary
            )
            window._show_gt_cb.setChecked(not window._show_gt_cb.isChecked())
            np.testing.assert_array_equal(window.collapsed_doppler_image.image, image)
    finally:
        manager.set_mode(original, persist=False)


def test_tracking_seek_excludes_future_and_keeps_all_saved_ids(window):
    window._set_view_mode("tracking")
    records = tuple(
        replace(
            r,
            detections=(
                {
                    "track_id": 7,
                    "world_track_position": [r.time_s, 1.0, 0.0],
                    "range_m": 1.0,
                    "velocity_m_s": 2.0,
                },
                {
                    "track_id": 9,
                    "world_track_position": [r.time_s, 2.0, 0.0],
                    "range_m": 2.0,
                    "velocity_m_s": 3.0,
                },
            ),
            truth_positions=(("person", (r.time_s, 3.0, 0.0)),),
        )
        for r in _history().records
    )
    history = RecordingHistory(records)
    request = _request(5, history)
    request.display_frame.update(rx_positions=np.zeros((1, 3)), rx_orientations=np.zeros((1, 3)))
    request.display_frame["sensing"]["detections"] = [
        {
            **row,
            **{
                f"track_{axis}_m": value
                for axis, value in zip("xyz", row["world_track_position"], strict=True)
            },
        }
        for row in records[-1].detections
    ]
    window.update_from_request(request)
    assert set(window._tracking_curves) == {"7", "9"}
    curve = window._tracking_curves["7"]
    curve.sigPointsClicked.emit(curve, [], None)
    assert window._linked_selection.track_id == 7
    assert window.locate_selection_button.isEnabled()
    assert "saved spatial position" in window.locate_selection_button.toolTip()
    window.set_linked_selection(None)
    assert not window.locate_selection_button.isEnabled()
    assert "range shell" in window.locate_selection_button.toolTip()
    assert all(curve.opacity() == 1.0 for curve in window._tracking_curves.values())
    for scatter in (window._gt_range_det_scatter, window._gt_speed_det_scatter):
        assert all(point.pen().color().alphaF() == 1.0 for point in scatter.points())
    window.update_from_request(_request(1, history))
    for curve in window._tracking_curves.values():
        np.testing.assert_array_equal(curve.xData, [0.0, 0.25])
    np.testing.assert_array_equal(window._tracking_truth_curves["person"].yData, [3.0, 3.0])


def test_detection_defers_xy_until_tracking_opens_and_respects_rewind(window):
    history = RecordingHistory(
        tuple(
            replace(
                row,
                detections=(
                    dict(
                        track_id=7,
                        world_track_position=[row.time_s, 1.0, 0.0],
                        range_m=1.0,
                        velocity_m_s=2.0,
                    ),
                ),
                truth_positions=(("person", (row.time_s, 3.0, 0.0)),),
            )
            for row in _history().records
        )
    )
    window._set_view_mode("analysis")
    window.update_from_request(_request(5, history))
    assert not window._tracking_curves
    assert not window._tracking_truth_curves
    np.testing.assert_array_equal(window._gt_range_det_scatter.getData()[0], np.arange(6) * 0.25)

    window.update_from_request(_request(1, history))
    window._set_view_mode("tracking")
    window.update_from_request(_request(1, history))
    np.testing.assert_array_equal(window._tracking_curves["7"].xData, [0.0, 0.25])
    np.testing.assert_array_equal(window._tracking_truth_curves["person"].xData, [0.0, 0.25])


def test_session_coalesces_requests_and_cancels_retired_jobs(monkeypatch):
    session = DashboardSession(SimpleNamespace())
    submitted = []
    future = Future()
    monkeypatch.setattr(session._executor, "submit", lambda *args: submitted.append(args) or future)
    session.request((0, "analysis", "primary", False, "auto"))
    session.request((64, "analysis", "primary", False, "auto"))
    assert len(submitted) == 1
    session.close()
    session.pending = None
    session.request((128, "analysis", "primary", False, "auto"))
    assert len(submitted) == 1
    assert session._cancelled.is_set()


def test_active_refresh_is_faster_and_pending_reads_remain_bounded(window, monkeypatch):
    from visualizer.src.panels import sensing_panel

    provider = SimpleNamespace()
    panel = sensing_panel.SensingPanel(SimpleNamespace())
    panel._sensing_window = window
    panel._start_auto_refresh()
    idle_interval = panel._refresh_timer.interval()
    monkeypatch.setattr(panel, "_poll_dashboard", lambda: None)
    monkeypatch.setattr(panel, "_prefer_fresh_sensing_display", lambda: False)
    session = DashboardSession(provider)
    panel._dashboard_session = session
    submitted = []
    monkeypatch.setattr(
        session._executor, "submit", lambda *args, **kwargs: submitted.append(args) or Future()
    )
    try:
        for step in (0, 64, 128):
            panel._refresh_dashboard_async(provider, step, force=False)
        assert 0 < panel._refresh_timer.interval() <= 35 < idle_interval
        assert panel._dashboard_poll_timer.interval() <= 10
        assert len(submitted) == 1
        # The next idle request reads the newest step, without replaying 64/128.
        session.pending = None
        panel._refresh_dashboard_async(provider, 192, force=False)
        assert session.pending_key[0] == 192
        assert len(submitted) == 2
        panel._retire_dashboard_session()
        assert panel._refresh_timer.interval() == idle_interval
    finally:
        panel._retire_dashboard_session()
        panel._refresh_timer.stop()


def test_playback_publishes_completed_profiles_while_scene_keeps_advancing(window, monkeypatch):
    from visualizer.src.panels import sensing_panel

    provider = SimpleNamespace()
    parent = SimpleNamespace(
        animation_running=True,
        animation_step=65,
        get_available_animation_steps=lambda: list(range(192)),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._sensing_window = window
    window._set_view_mode("analysis")
    session = panel._dashboard_session = DashboardSession(provider)
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider)
    try:
        assert not panel._prefer_fresh_sensing_display()
        assert panel._dashboard_follows_results()
        for result_step, current_step in ((63, 65), (127, 129), (191, 191)):
            parent.animation_step = current_step
            request = _request(result_step)
            request = replace(
                request,
                rd_result=replace(
                    request.rd_result,
                    power_linear=request.rd_result.power_linear * (result_step + 1),
                ),
            )
            session.pending = Future()
            session.pending_key = (result_step, "analysis", "primary", True, "auto")
            session.pending.set_result(request)
            panel._poll_dashboard()
            assert window._last_request is not None
            assert window._last_request.effective_display_step == result_step
            np.testing.assert_allclose(
                window.rp_samples.getData()[1],
                10 * np.log10(request.rd_result.power_linear.sum(axis=0)),
            )
        parent.animation_running = False
        assert not panel._dashboard_follows_results()
    finally:
        panel._retire_dashboard_session()


def test_paused_tracking_keeps_prior_result_when_scrubbing_between_updates(window):
    from visualizer.src.panels.sensing_panel import SensingPanel

    frames = {
        i: {
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "timestamp_s": i * 0.25,
            **(_frame(i) if i in (1, 3, 5) else {}),
        }
        for i in range(7)
    }
    parent = SimpleNamespace(
        animation_running=False,
        cache_service=SimpleNamespace(
            get_frame=frames.get, frame_cache=SimpleNamespace(keys=lambda: [5, 3, 1])
        ),
    )
    panel = SensingPanel(parent)
    panel._sensing_window = window
    window._set_view_mode("tracking")
    service = SensingService(parent)
    history = RecordingHistory(
        tuple(
            replace(
                r,
                detections=(
                    {
                        "track_id": 7,
                        "world_track_position": [r.time_s, 1.0, 0.0],
                    },
                ),
            )
            for i, r in enumerate(_history().records)
            if i in (1, 3, 5)
        )
    )
    assert not panel._prefer_fresh_sensing_display()
    assert panel._dashboard_follows_results()
    for step, saved_steps in ((4, [1, 3]), (6, [1, 3, 5]), (2, [1])):
        request = load_dashboard_request(
            service, step, "tracking", "primary", panel._dashboard_follows_results(), "auto"
        )
        assert request is not None
        assert request.state_step == step
        assert request.effective_display_step == saved_steps[-1]
        assert request.display_from_cache
        window.update_from_request(replace(request, recording_history=history))
        assert window._plots_widget.isVisible()
        np.testing.assert_allclose(window._tracking_curves["7"].xData, np.array(saved_steps) * 0.25)
    assert load_dashboard_request(service, 0, "tracking", "primary", True, "auto") is None
    window._set_view_mode("analysis")
    assert not panel._dashboard_follows_results()
    assert load_dashboard_request(service, 4, "analysis", "primary", False, "auto") is None


@pytest.mark.parametrize("playing,current_step", [(True, 5), (False, 65)])
def test_late_playback_result_is_rejected_after_rewind_or_pause(
    window, monkeypatch, playing, current_step
):
    from visualizer.src.panels import sensing_panel

    provider = SimpleNamespace()
    parent = SimpleNamespace(
        animation_running=playing,
        animation_step=current_step,
        get_available_animation_steps=lambda: list(range(192)),
    )
    panel = sensing_panel.SensingPanel(parent)
    panel._sensing_window = window
    window._set_view_mode("analysis")
    session = panel._dashboard_session = DashboardSession(provider)
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider)
    try:
        session.pending = Future()
        session.pending_key = (63, "analysis", "primary", True, "auto")
        session.pending.set_result(_request(63))
        panel._poll_dashboard()
        assert window._last_request is None
    finally:
        panel._retire_dashboard_session()


def test_range_time_zoom_fits_all_saved_returns_and_can_show_full_range(window):
    window._zoom_data_cb.setChecked(True)
    axis = np.arange(128) * 0.3
    records = []
    for index, record in enumerate(_history().records):
        power = np.zeros(128)
        power[15 + index] = 1e-11
        records.append(replace(record, range_m=axis, range_power=power))
    history = RecordingHistory(tuple(records))
    window.update_from_request(_request(5, history))
    _APP.processEvents()
    lo, hi = window.range_time_plot.viewRange()[1]
    assert 3.0 < lo < axis[15]
    assert axis[20] < hi < 8.0
    assert window._zoom_data_cb.isVisibleTo(window._advanced_widget)
    window._zoom_data_cb.setChecked(False)
    _APP.processEvents()
    lo, hi = window.range_time_plot.viewRange()[1]
    assert lo <= axis[0]
    assert hi >= axis[-1]


def test_incremental_range_time_zoom_preserves_weak_return(window):
    window._zoom_data_cb.setChecked(True)
    axis = np.arange(128) * 0.3
    power = np.zeros(128)
    power[15] = 1e-20
    window._update_range_time_waterfall(
        SimpleNamespace(power_linear=power, range_axis_m=axis),
        sensing_block=_frame()["sensing"],
        variant_suffix="",
    )
    lo, hi = window.range_time_plot.viewRange()[1]
    assert 3.0 < lo < 4.5 < hi < 6.0
    assert window.range_time_raster.image[0, 15] == pytest.approx(-200.0)


@pytest.mark.parametrize("irregular_axis", [None, "time", "range"])
def test_range_time_native_geometry_preserves_edges_values_and_orientation(window, irregular_axis):
    from PySide6.QtCore import QPointF

    times = np.array([1.0, 1.5, 2.0])
    ranges = np.array([10.0, 12.0, 14.0, 16.0])
    if irregular_axis == "time":
        times[-1] = 2.25
    if irregular_axis == "range":
        ranges[-1] = 17.0
    values = np.arange(6.0).reshape(2, 3)
    window._set_range_time_image(times, ranges, values, (-10.0, 5.0))
    if irregular_axis is None:
        item = window.range_time_raster
        assert item.isVisible() and not window.range_time_image.isVisible()
        np.testing.assert_array_equal(item.image, values)
        np.testing.assert_array_equal(item.getLevels(), [-10.0, 5.0])
        for i, time in enumerate(times):
            for j, distance in enumerate(ranges):
                point = item.mapToParent(QPointF(i, j))
                assert point.x() == pytest.approx(time)
                assert point.y() == pytest.approx(distance)
    else:
        item = window.range_time_image
        assert item.isVisible() and not window.range_time_raster.isVisible()
        np.testing.assert_array_equal(item.z, values)
        np.testing.assert_array_equal(item.x[:, 0], times)
        np.testing.assert_array_equal(item.y[0, :], ranges)
    window._clear_recording_images()
    assert not window.range_time_raster.isVisible()
    assert not window.range_time_image.isVisible()


@pytest.mark.parametrize("explicit_inventory", [False, True])
def test_angle_capability_is_disabled_without_products_and_routes_to_angle_when_available(
    window, monkeypatch, explicit_inventory
):
    index = window._view_mode_box.findData("angular")
    item = window._view_mode_box.model().item(index)
    assert not item.isEnabled()
    request = _request()
    if explicit_inventory:
        request = replace(request, available_product_kinds=frozenset({"range_doppler"}))
    window.update_from_request(request)
    assert not item.isEnabled()
    assert "no saved angular product" in item.toolTip()
    modes = []
    monkeypatch.setattr(window, "_update_angular_products", lambda *_args: modes.append("angular"))
    request = replace(request, available_product_kinds=frozenset({"range_doppler_angle"}))
    window.update_from_request(request)
    assert item.isEnabled()
    window._view_mode_box.setCurrentIndex(index)
    assert window._selected_view_mode() == "angular"
    assert window.request_view_mode() == "angular"
    assert modes == ["angular"]
    assert window.range_azimuth_plot.isVisible()
    assert not window.detection_stage_inspector.isVisible()


def test_tracking_supports_saved_range_tracks_without_cartesian_positions(window):
    window._initial_view_selected = True
    window._set_view_mode("tracking")
    history = RecordingHistory(
        tuple(
            replace(
                r,
                truth_positions=(("person", (r.time_s, 0.1, 0.8)),),
                detections=({"track_id": 1, "range_m": 5.0, "velocity_m_s": 1.0},),
            )
            for r in _history().records
        )
    )
    request = _request(5, history)
    for key in ("config", "config_resolved"):
        request.display_frame["sensing"][key]["track_confirmation_enabled"] = True
        request.display_frame["sensing"][key]["tracker_type"] = "kalman"
    window.update_from_request(request)
    item = window._view_mode_box.model().item(window._view_mode_box.findData("tracking"))
    assert item.isEnabled()
    assert window._selected_view_mode() == "tracking"
    assert window.tracking_plot.isHidden()
    assert window.gt_range_plot.isVisible() and window.gt_speed_plot.isVisible()


@pytest.mark.parametrize("separate_tracks", [False, True])
def test_tracking_remains_available_during_confirmation_when_later_cartesian_tracks_exist(
    window, separate_tracks
):
    history = _history()
    records = list(history.records)
    records[-1] = replace(
        records[-1],
        detections=(
            {
                "track_id": 1,
                "world_track_position": (1.0, 2.0, 0.0),
                "range_m": 3.0,
                "velocity_m_s": 1.0,
            },
        ),
    )
    history = replace(history, records=tuple(records))
    request = _request(0, _history() if separate_tracks else history)
    if separate_tracks:
        request = replace(request, tracking_history=history)
    window.update_from_request(request)
    item = window._view_mode_box.model().item(window._view_mode_box.findData("tracking"))
    assert item.isEnabled()
    window._set_view_mode("tracking")
    assert window._selected_view_mode() == "tracking"
    assert not window._tracking_curves
    window.update_from_request(
        replace(request, recording_history=_history(), tracking_history=None)
    )
    assert not item.isEnabled()
    assert window._selected_view_mode() == "analysis"


def test_first_result_starts_with_summary_and_keeps_detection_details_available(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    request = _request()
    for key in ("config", "config_resolved"):
        request.display_frame["sensing"][key]["track_confirmation_enabled"] = True
    window.update_from_request(request)
    assert window._selected_view_mode() == "summary"
    assert window.run_summary_panel.isVisible()
    assert not window.rd_plot.isVisible()
    window._set_view_mode("analysis")
    assert window.rd_plot.isVisible() and window.rp_plot.isVisible()
    assert not hasattr(window, "response_badge")
    assert not window.selection_tools.isVisible()
    assert window.provenance_label.isVisible()
    assert window.figure_options.isVisible()
    assert not window.selection_tools_toggle.isVisible()
    assert not hasattr(window, "response_details_toggle")
    window._select_inspection_coordinate(1, 2)
    assert window.selection_tools_toggle.isVisible()
    window.selection_tools_toggle.click()
    assert window.selection_tools.isVisible()


def test_retired_provider_generation_cannot_publish_a_late_result(window, monkeypatch):
    from visualizer.src.panels import sensing_panel

    provider = SimpleNamespace(_generation_epoch=0)
    panel = sensing_panel.SensingPanel(SimpleNamespace())
    panel._sensing_window = window
    session = DashboardSession(provider)
    panel._dashboard_session = session
    session.pending = Future()
    session.pending.set_result(_request(5, _history()))
    session.pending_key = (5, "micro-doppler", "primary", False, "auto")
    monkeypatch.setattr(sensing_panel, "visualizer_frame_provider", lambda _: provider)
    provider._generation_epoch = 1
    panel._poll_dashboard()
    assert panel._dashboard_session is None
    assert window._last_request is None
    assert session._cancelled.is_set()


def test_reopening_reuses_completed_data_but_not_another_source():
    provider = SimpleNamespace()
    first = DashboardSession(provider)
    first.history = _history()
    first.history_key = ("clock", "run", "pair", "primary")
    first.close()
    reopened = DashboardSession(provider, previous=first)
    other = DashboardSession(SimpleNamespace(), previous=first)
    try:
        assert reopened.service is first.service
        assert reopened.history is first.history
        assert reopened.pending is None and reopened.history_pending is None
        assert other.service is not first.service
        assert other.history is None
    finally:
        reopened.close()
        other.close()


def test_history_refreshes_when_recorded_inventory_grows(monkeypatch):
    session = DashboardSession(SimpleNamespace())
    submitted = []
    token = [0]
    monkeypatch.setattr(session.service, "dashboard_inventory_token", lambda: tuple(token))
    monkeypatch.setattr(
        session._executor, "submit", lambda *args, **kwargs: submitted.append(args) or Future()
    )
    sensing = _frame()["sensing"]
    session.request_history(sensing, "primary")
    session.history_pending = None
    session.request_history(sensing, "primary")
    assert len(submitted) == 1
    token.append(1)
    session.request_history(sensing, "primary")
    assert len(submitted) == 2
    session.close()


def test_cyclic_plot_cleanup_is_scheduled_on_gui_thread():
    script = """
import gc
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
from visualizer.src.sensing.qt_lifetime import ensure_gui_garbage_collection
app = QApplication([])
gc.collect()
ensure_gui_garbage_collection()
collector = app._orchav_plot_collector
ensure_gui_garbage_collection()
assert collector is app._orchav_plot_collector
assert collector.timer.parent() is app
assert not gc.isenabled()
finalizers = []
class Cycle:
    def __init__(self): self.cycle = self
    def __del__(self): finalizers.append(QThread.currentThread() == app.thread())
cycle = Cycle()
del cycle
def allocate():
    values = [dict(i=i) for i in range(2000)]
    assert not finalizers
with ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(allocate).result()
collector.threshold = (0, 0, 0)
collector.check()
assert finalizers == [True], finalizers
collector.timer.stop()
gc.enable()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
