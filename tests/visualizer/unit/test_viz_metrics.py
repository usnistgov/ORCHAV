from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QFont

pytest.importorskip("pyqtgraph")

from shared.statistics import FrameStats
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog
from visualizer.src.metrics.viz_metrics import MetricsWindow, _default_ui_font


def _canonical_frame():
    return SimpleNamespace(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        lines=np.array([[0, 1], [2, 3]], dtype=np.int64),
        path_start_indices=np.array([0, 2], dtype=np.int64),
        path_orders=np.array([0, 2], dtype=np.int64),
        path_delays=np.array([10.0, 20.0], dtype=float),
        path_losses=np.array([80.0, 90.0], dtype=float),
        path_tx=np.array([0, 1], dtype=np.int64),
        path_rx=np.array([0, 1], dtype=np.int64),
        path_aod_az=np.array([5.0, 55.0], dtype=float),
        path_aod_el=np.array([2.0, 22.0], dtype=float),
        path_aoa_az=np.array([15.0, 65.0], dtype=float),
        path_aoa_el=np.array([4.0, 24.0], dtype=float),
        aod_az=np.array([5.0, 5.0, 55.0, 55.0], dtype=float),
        aod_el=np.array([2.0, 2.0, 22.0, 22.0], dtype=float),
        aoa_az=np.array([15.0, 15.0, 65.0, 65.0], dtype=float),
        aoa_el=np.array([4.0, 4.0, 24.0, 24.0], dtype=float),
    )


def _view_model():
    return SimpleNamespace(
        canonical_data=_canonical_frame(),
        path_mask=np.array([False, True], dtype=bool),
    )


def _frame_stats():
    return FrameStats(
        total_paths=1,
        orders_hist={2: 1},
        delay_range_ns=(20.0, 20.0),
        path_loss_range=(90.0, 90.0),
        binned_power_delay_profile=(np.array([20.0]), np.array([-90.0])),
    )


def _deep_canonical_frame():
    return SimpleNamespace(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.4, 2.0, 0.0],
                [0.7, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
        lines=np.array(
            [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]],
            dtype=np.int64,
        ),
        path_id=np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64),
        path_start_indices=np.array([0, 2, 5], dtype=np.int64),
        path_orders=np.array([0, 1, 2], dtype=np.int64),
        path_delays=np.array([5.0, 20.0, 30.0], dtype=float),
        path_losses=np.array([60.0, 80.0, 90.0], dtype=float),
        path_tx=np.array([0, 0, 1], dtype=np.int64),
        path_rx=np.array([0, 1, 1], dtype=np.int64),
        path_aod_az=np.array([0.0, 20.0, 45.0], dtype=float),
        path_aod_el=np.array([0.0, 5.0, 15.0], dtype=float),
        path_aoa_az=np.array([180.0, 210.0, 240.0], dtype=float),
        path_aoa_el=np.array([0.0, -5.0, -15.0], dtype=float),
        material_ids=np.array([0, 0, 0, 1, 0, 0, 2, 1, 0], dtype=np.int64),
        material_id_to_name={0: "", 1: "concrete", 2: "glass"},
        aod_az=np.array([0.0, 0.0, 20.0, 20.0, 20.0, 45.0, 45.0, 45.0, 45.0]),
        aod_el=np.array([0.0, 0.0, 5.0, 5.0, 5.0, 15.0, 15.0, 15.0, 15.0]),
        aoa_az=np.array([180.0, 180.0, 210.0, 210.0, 210.0, 240.0, 240.0, 240.0, 240.0]),
        aoa_el=np.array([0.0, 0.0, -5.0, -5.0, -5.0, -15.0, -15.0, -15.0, -15.0]),
    )


def _deep_view_model():
    return SimpleNamespace(
        canonical_data=_deep_canonical_frame(),
        path_mask=np.array([True, True, True], dtype=bool),
    )


def _deep_frame_stats():
    return FrameStats(
        total_paths=3,
        orders_hist={0: 1, 1: 1, 2: 1},
        delay_range_ns=(5.0, 30.0),
        path_loss_range=(60.0, 90.0),
        delay_spread_ns=2.0,
        angular_spread_deg=8.0,
        binned_power_delay_profile=(
            np.array([5.0, 20.0, 30.0]),
            np.array([-60.0, -80.0, -90.0]),
        ),
    )


def test_metrics_window_uses_platform_default_ui_font(qapp):
    window = MetricsWindow(update_hz=8)

    expected_family = window.font().family()
    for label in (
        window.title_label,
        window.context_label,
        window.channel_summary_label,
        window.pair_count_title_label,
        window.data_status_label,
        window.stats_label,
    ):
        assert label.font().family() == expected_family

    assert window.title_label.font().pointSize() == 13
    assert window.title_label.font().bold() is True
    window.close()


def test_default_ui_font_begins_from_qfont(qapp) -> None:
    baseline = QFont().resolve(qapp.font())
    font = _default_ui_font(12, weight=QFont.Weight.Bold)

    assert font.family() == baseline.family()
    assert font.pointSize() == 12
    assert font.weight() == QFont.Weight.Bold


def test_metrics_window_uses_selected_path_mask(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(
        _view_model(),
        _frame_stats(),
        context={
            "step": 3,
            "selected_tx": 1,
            "selected_rx": 1,
            "visible_paths": 1,
            "total_paths": 2,
            "filters": ["orders", "AoA"],
        },
    )
    window._refresh()

    np.testing.assert_allclose(
        window._selected_path_values("path_delays", "delay"), np.array([20.0])
    )
    np.testing.assert_allclose(
        window._selected_path_values("path_losses", "loss"), np.array([90.0])
    )
    assert "Frame 4" in window.context_label.text()
    assert "TX: TX2" in window.context_label.text()
    assert "RX: RX2" in window.context_label.text()
    assert "MPCs: 1/2" in window.context_label.text()

    window.close()


def test_metrics_window_angle_selector_and_export_rows_follow_selection(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_view_model(), _frame_stats(), context={})
    window.angle_combo.setCurrentIndex(window.angle_combo.findData("aoa_az"))
    window._refresh()

    np.testing.assert_allclose(window._selected_path_values("", "aoa_az"), np.array([65.0]))
    rows = window._selected_path_rows()
    assert rows == [
        {
            "path_id": 1,
            "tx": 1,
            "rx": 1,
            "order": 2,
            "delay_ns": 20.0,
            "path_loss_db": 90.0,
            "aod_az_deg": 55.0,
            "aod_el_deg": 22.0,
            "aoa_az_deg": 65.0,
            "aoa_el_deg": 24.0,
        }
    ]

    window.close()


def test_metrics_window_catalog_preserves_unavailable_angle_vs_real_zero(qapp):
    canonical = _canonical_frame()
    canonical.path_aoa_az = np.array([0.0, np.nan], dtype=np.float32)
    # Point-level zeros cannot stand in for missing whole-path angle metadata.
    canonical.aoa_az = np.zeros(canonical.points.shape[0], dtype=np.float32)
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(2, dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _frame_stats(), context={})

    angles = window._path_array("", "aoa_az", 2)
    assert angles[0] == pytest.approx(0.0)
    assert np.isnan(angles[1])
    np.testing.assert_array_equal(
        window._selected_path_values("", "aoa_az"),
        np.array([0.0], dtype=np.float32),
    )
    rows = window._selected_path_rows()
    assert rows[0]["aoa_az_deg"] == pytest.approx(0.0)
    assert rows[1]["aoa_az_deg"] == ""

    window.close()


def test_metrics_window_incomplete_legacy_fixture_keeps_narrow_path_fallback(qapp):
    canonical = SimpleNamespace(
        path_orders=np.array([0, 1], dtype=np.int64),
        path_delays=np.array([10.0, 20.0], dtype=float),
        path_losses=np.array([80.0, 90.0], dtype=float),
        path_tx=np.array([0, 0], dtype=np.int64),
        path_rx=np.array([0, 1], dtype=np.int64),
    )
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.array([False, True], dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _frame_stats(), context={})

    assert window._path_count(canonical) == 2
    np.testing.assert_allclose(
        window._selected_path_values("path_delays", "delay"),
        np.array([20.0]),
    )
    assert window._catalog_attempted is True
    assert window._catalog_instance is None

    window.close()


def test_metrics_window_enqueue_is_coalesced_until_timer_refresh(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_view_model(), _frame_stats(), context={"step": 4})
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={"step": 6})

    assert window._pending_refresh is True
    assert window._timer.isActive()
    assert window._timer.isSingleShot()
    assert "Frame 7" not in window.context_label.text()

    window._periodic_refresh()

    assert window._pending_refresh is False
    assert not window._timer.isActive()
    assert "Frame 7" in window.context_label.text()
    window.close()


def test_provider_statistics_are_computed_once_for_latest_queued_frame(qapp):
    first = _view_model()
    latest = _deep_view_model()
    calls = []

    def provide(view_model):
        calls.append(view_model)
        return _deep_frame_stats()

    window = MetricsWindow(update_hz=8, frame_stats_provider=provide)
    window.enqueue(first, context={"step": 0})
    window.enqueue(latest, context={"step": 6})

    window._periodic_refresh()

    assert calls == [latest]
    assert window._last_stats is not None
    assert "Frame 7" in window.context_label.text()

    window.tabs.setCurrentWidget(window._channel_tab)
    window._periodic_refresh()
    assert calls == [latest]
    window.close()


def test_update_rate_accepts_fixed_hz_and_adaptive_maximum(qapp):
    window = MetricsWindow(update_hz=8)

    assert window._fixed_refresh_period_ms() == 125
    window.update_rate_spin.setValue(20)
    assert window._fixed_refresh_period_ms() == 50
    window.update_rate_spin.setValue(0)
    assert window.update_rate_spin.text() == "Maximum (adaptive)"
    assert window._fixed_refresh_period_ms() is None
    assert "Intermediate Metrics frames may be skipped" in window.update_rate_spin.toolTip()
    window.close()


def test_adaptive_maximum_cooldown_scales_with_refresh_cost(qapp):
    window = MetricsWindow(update_hz=0)

    window._last_refresh_duration_ms = 4.0
    assert window._adaptive_cooldown_ms() == 16

    window._last_refresh_duration_ms = 40.2
    assert window._adaptive_cooldown_ms() == 41
    window.close()


def test_adaptive_maximum_rearms_once_after_measured_refresh_cost(qapp):
    window = MetricsWindow(update_hz=0)
    clock_values = iter((10.0, 10.0404, 10.0404))
    window._clock = lambda: next(clock_values)
    window._refresh = lambda: None

    window.enqueue(_view_model(), _frame_stats(), context={"step": 0})
    window._periodic_refresh()
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={"step": 6})

    assert window._last_refresh_duration_ms == pytest.approx(40.4)
    assert window._timer.isActive()
    assert window._timer.interval() == 41
    assert window._last_vm is not None
    assert window._last_context["step"] == 6
    window.close()


def test_fixed_refresh_overrun_yields_instead_of_catching_up(qapp):
    window = MetricsWindow(update_hz=8)
    window._last_refresh_started_s = 10.0
    window._last_refresh_finished_s = 10.150
    window._last_refresh_duration_ms = 150.0

    assert window._next_refresh_delay_ms(now_s=10.150) == 150
    window.close()


def test_fixed_refresh_near_budget_still_reserves_ui_time(qapp):
    window = MetricsWindow(update_hz=60)
    window._last_refresh_started_s = 10.0
    window._last_refresh_finished_s = 10.015
    window._last_refresh_duration_ms = 15.0

    assert window._next_refresh_delay_ms(now_s=10.015) == 16
    window.close()


def test_manual_refresh_consumes_pending_timer_and_records_cost(qapp):
    window = MetricsWindow(update_hz=0)
    refresh_calls = []
    clock_values = iter((10.0, 10.020))
    window._clock = lambda: next(clock_values)
    window._refresh = lambda: refresh_calls.append(True)
    window.enqueue(_view_model(), _frame_stats(), context={"step": 0})

    window._manual_refresh()

    assert refresh_calls == [True]
    assert not window._pending_refresh
    assert not window._timer.isActive()
    assert window._last_refresh_duration_ms == pytest.approx(20.0)
    window.close()


def test_pause_retains_latest_input_and_unpause_arms_one_refresh(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_view_model(), _frame_stats(), context={"step": 0})

    window.pause_cb.setChecked(True)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={"step": 6})

    assert not window._pending_refresh
    assert not window._timer.isActive()
    assert window._last_context["step"] == 6

    window.pause_cb.setChecked(False)
    assert window._pending_refresh
    assert window._timer.isActive()

    window._periodic_refresh()
    assert "Frame 7" in window.context_label.text()
    assert not window._timer.isActive()
    window.close()


def test_metrics_window_log_y_transforms_bar_counts_without_log_axis(qapp):
    window = MetricsWindow(update_hz=8)
    counts = np.array([0.0, 1.0, 3.0])

    np.testing.assert_allclose(window._count_heights(counts), counts)

    window.log_y_cb.setChecked(True)

    np.testing.assert_allclose(window._count_heights(counts), np.log10(counts + 1.0))
    assert not window.delay_plot.getPlotItem().ctrl.logYCheck.isChecked()
    window.close()


def test_metrics_window_material_breakdown_uses_selected_frame_materials(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    rows, depth_matrix, material_labels, depths = window._selected_material_breakdown()

    assert [row["material"] for row in rows] == ["concrete", "glass"]
    assert [row["hits"] for row in rows] == [2, 1]
    assert [row["path_count"] for row in rows] == [2, 1]
    assert material_labels == ["concrete", "glass"]
    assert depths == [1, 2]
    np.testing.assert_allclose(depth_matrix, np.array([[1.0, 1.0], [1.0, 0.0]]))

    window.tabs.setCurrentWidget(window._materials_tab)
    window._refresh()

    model = window.material_table.model()
    assert model.rowCount() == 2
    assert model.data(model.index(0, 0)) == "concrete"

    layout = window._materials_tab.layout()
    assert layout.getItemPosition(layout.indexOf(window.material_table)) == (0, 0, 1, 2)
    assert layout.getItemPosition(layout.indexOf(window.material_depth_plot)) == (1, 0, 1, 1)
    assert layout.getItemPosition(layout.indexOf(window.material_power_plot)) == (1, 1, 1, 1)

    expected_display = np.array([[1.0, 1.0], [1.0, np.nan]])
    np.testing.assert_allclose(
        window.material_depth_heatmap.image,
        expected_display,
        equal_nan=True,
    )
    np.testing.assert_allclose(window.material_depth_heatmap.levels, np.array([0.0, 2.0]))
    assert window.material_depth_heatmap.axisOrder == "row-major"
    assert window.material_depth_colorbar.levels() == pytest.approx((0.0, 2.0))
    assert window.material_depth_colorbar.getAxis("right").labelText == "Hit count"
    assert window.material_depth_colorbar.isVisible()
    color_sample = window.material_depth_heatmap.lut[64, :3]
    assert len(set(color_sample.tolist())) > 1

    power_values = np.asarray([row["power_db"] for row in rows], dtype=float)
    bar_floor = float(window.material_power_bars.opts["y0"])
    np.testing.assert_allclose(
        window.material_power_bars.opts["height"] + bar_floor,
        power_values,
    )
    assert bar_floor < float(np.min(power_values))
    colorbar_id = id(window.material_depth_colorbar)
    window._clear_material_plots()
    assert not window.material_depth_colorbar.isVisible()
    assert id(window.material_depth_colorbar) == colorbar_id
    window.close()


def test_metrics_window_material_breakdown_uses_catalog_sequences(qapp, monkeypatch):
    original_material_sequence = MpcPathCatalog.material_sequence
    requested_path_ids = []

    def tracked_material_sequence(self, path_id):
        requested_path_ids.append(path_id)
        if path_id == 1:
            return np.array([2], dtype=np.int64)
        return original_material_sequence(self, path_id)

    monkeypatch.setattr(
        MpcPathCatalog,
        "material_sequence",
        tracked_material_sequence,
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    rows, _, material_labels, _ = window._selected_material_breakdown()

    assert requested_path_ids == [0, 1, 2]
    assert material_labels == ["glass", "concrete"]
    assert [row["hits"] for row in rows] == [2, 1]

    window.close()


def test_metrics_window_displays_binned_pdp_values_as_path_gain_db(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    window._refresh()

    np.testing.assert_allclose(
        window.binned_pdp_stems.opts["x"],
        np.array([5.0, 20.0, 30.0]),
    )
    np.testing.assert_allclose(
        window.binned_pdp_stems.opts["height"],
        np.array([30.0, 10.0, 0.0]),
    )
    assert window.binned_pdp_stems.opts["y0"] == -90.0
    assert "Phase is not used" in window.binned_pdp_plot.toolTip()

    window.close()


def test_metrics_window_exposes_one_corrected_power_delay_profile(qapp):
    window = MetricsWindow(update_hz=8)

    assert window._plot_titles[window.binned_pdp_plot] == "Power Delay Profile (1 ns resolution)"
    assert not hasattr(window, "pdp_plot")
    axis = window.binned_pdp_plot.getAxis("left")
    assert axis.labelText == "Summed path gain"
    assert axis.labelUnits == "dB"

    window.close()


def test_metrics_window_uses_interaction_order_label_and_six_plus_bucket(qapp):
    window = MetricsWindow(update_hz=8)
    stats = FrameStats(total_paths=6, orders_hist={0: 1, 6: 2, 8: 3})
    window.enqueue(_view_model(), stats, context={})

    window._update_order_distribution()

    assert "Interaction Order" in window.order_plot.getPlotItem().titleLabel.text
    np.testing.assert_array_equal(window.order_bars.opts["x"], np.arange(7))
    np.testing.assert_array_equal(
        window.order_bars.opts["height"],
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]),
    )

    window.close()


def test_material_gain_proxy_counts_repeated_material_once_per_path(qapp, monkeypatch):
    """Repeated bounces stay visible as hits without duplicating path gain."""

    def repeated_material(self, path_id):
        if path_id == 1:
            return np.array([1, 1], dtype=np.int64)
        return np.empty((0,), dtype=np.int64)

    monkeypatch.setattr(MpcPathCatalog, "material_sequence", repeated_material)
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    rows, _, _, _ = window._selected_material_breakdown()

    assert len(rows) == 1
    assert rows[0]["hits"] == 2
    assert rows[0]["path_count"] == 1
    assert rows[0]["power_db"] == pytest.approx(-80.0)
    assert rows[0]["mean_loss"] == pytest.approx(80.0)

    window.close()


def test_material_breakdown_excludes_estimated_loss_from_gain_proxy(qapp):
    canonical = _deep_canonical_frame()
    canonical.path_loss_is_estimated = np.array([False, True, False], dtype=bool)
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(3, dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _deep_frame_stats(), context={})

    rows, _, _, _ = window._selected_material_breakdown()
    concrete = next(row for row in rows if row["material"] == "concrete")

    assert concrete["hits"] == 2
    assert concrete["path_count"] == 2
    assert concrete["power_db"] == pytest.approx(-90.0)
    assert concrete["mean_loss"] == pytest.approx(90.0)

    window.close()


def test_material_breakdown_reports_no_proxy_when_all_losses_are_estimated(qapp):
    canonical = _deep_canonical_frame()
    canonical.path_loss_is_estimated = np.ones(3, dtype=bool)
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(3, dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _deep_frame_stats(), context={})

    rows, _, _, _ = window._selected_material_breakdown()
    assert rows
    assert all(np.isnan(row["power_db"]) for row in rows)
    assert all(np.isnan(row["mean_loss"]) for row in rows)

    bottom_axis = window.material_power_plot.getAxis("bottom")
    bottom_axis.setTicks([[(0, "stale material")]])
    window._update_material_breakdown()
    assert len(window.material_power_bars.opts["x"]) == 0
    assert bottom_axis._tickLevels == []

    window.close()


def test_metrics_window_pair_matrix_and_channel_summary_are_frame_instant(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    matrix, tx_labels, rx_labels, title = window._pair_metric_matrix("count")
    np.testing.assert_allclose(matrix, np.array([[1.0, 0.0], [1.0, 1.0]]))
    assert tx_labels == [0, 1]
    assert rx_labels == [0, 1]
    assert title == "Selected MPCs by TX/RX Pair"

    metrics = window._channel_summary_metrics()
    assert metrics["paths"] == 3
    assert metrics["direct_count"] == 1
    assert metrics["interacted_count"] == 2
    assert metrics["strongest_loss"] == 60.0
    assert metrics["mean_delay_after_earliest_ns"] > 0.0
    assert metrics["direct_to_interacted_gain_db"] == pytest.approx(
        10.0 * np.log10(1e-6 / (1e-8 + 1e-9))
    )

    window._update_channel_summary()
    summary = window.channel_summary_label.text()
    assert "Direct/interacted" in summary
    assert "Aggregate path gain" in summary
    assert "LoS/NLoS" not in summary
    assert "K-factor" not in summary

    window.close()


def test_pair_count_table_is_exact_and_tx_major(qapp):
    canonical = _deep_canonical_frame()
    canonical.path_tx = np.array([0, 1, 0], dtype=np.int64)
    canonical.path_rx = np.array([0, 1, 2], dtype=np.int64)
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(3, dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _deep_frame_stats(), context={})

    matrix, tx_labels, rx_labels, _ = window._pair_metric_matrix("count")
    np.testing.assert_allclose(
        matrix,
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
    )
    assert tx_labels == [0, 1]
    assert rx_labels == [0, 1, 2]

    window._update_pair_heatmap()
    model = window.pair_count_table.model()
    assert model.rowCount() == 3
    assert [
        tuple(model.data(model.index(row, column)) for column in range(3))
        for row in range(model.rowCount())
    ] == [
        ("TX1", "RX1", "1"),
        ("TX1", "RX3", "1"),
        ("TX2", "RX2", "1"),
    ]
    assert window.pair_stack.currentWidget() is window.pair_count_panel
    assert window.pair_heatmap.image is None

    window.close()


def test_pair_metric_switch_uses_colored_matrix_and_preserves_missing_pairs(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    window._update_pair_heatmap()
    assert window.pair_count_table.model().rowCount() == 3

    window.pair_metric_combo.setCurrentIndex(window.pair_metric_combo.findData("mean_delay"))
    window._update_pair_heatmap()

    assert window.pair_count_table.model().rowCount() == 0
    assert window.pair_stack.currentWidget() is window.pair_heatmap_plot
    assert window.pair_heatmap.image is not None
    assert np.isnan(window.pair_heatmap.image[0, 1])
    assert window.pair_heatmap_plot.getAxis("bottom").labelText == "TX"
    assert window.pair_heatmap_plot.getAxis("left").labelText == "RX"
    assert window.pair_colorbar.getAxis("right").labelText == "Mean path delay"
    assert window.pair_colorbar.getAxis("right").labelUnits == "ns"
    assert window.pair_colorbar.levels() == pytest.approx((5.0, 30.0))
    assert window.pair_colorbar.isVisible()
    color_sample = window.pair_heatmap.lut[64, :3]
    assert len(set(color_sample.tolist())) > 1

    window.pair_metric_combo.setCurrentIndex(window.pair_metric_combo.findData("strongest_loss"))
    window._update_pair_heatmap()

    assert np.isnan(window.pair_heatmap.image[0, 1])
    assert window.pair_colorbar.getAxis("right").labelText == "Strongest path loss"
    assert window.pair_colorbar.getAxis("right").labelUnits == "dB"
    assert window.pair_colorbar.levels() == pytest.approx((60.0, 90.0))

    window.pair_metric_combo.setCurrentIndex(window.pair_metric_combo.findData("count"))
    window._update_pair_heatmap()
    assert window.pair_stack.currentWidget() is window.pair_count_panel
    assert not window.pair_colorbar.isVisible()

    window.close()


def test_channel_summary_excludes_estimated_delay_or_loss(qapp):
    canonical = _deep_canonical_frame()
    canonical.path_delay_is_estimated = np.array([False, False, True], dtype=bool)
    canonical.path_loss_is_estimated = np.array([False, True, False], dtype=bool)
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(3, dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _deep_frame_stats(), context={})

    metrics = window._channel_summary_metrics()

    assert metrics["paths"] == 2
    assert metrics["direct_count"] == 1
    assert metrics["interacted_count"] == 1
    assert metrics["strongest_loss"] == 60.0
    assert metrics["aggregate_path_gain_db"] == pytest.approx(10.0 * np.log10(1e-6 + 1e-9))
    assert metrics["mean_delay_after_earliest_ns"] == 0.0

    window.close()


def test_channel_summary_preserves_weak_path_gain_dynamic_range(qapp):
    canonical = _deep_canonical_frame()
    canonical.path_losses[2] = 240.0
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.array([False, False, True], dtype=bool),
    )
    window = MetricsWindow(update_hz=8)
    window.enqueue(view_model, _deep_frame_stats(), context={})

    metrics = window._channel_summary_metrics()

    assert metrics["aggregate_path_gain_db"] == pytest.approx(-240.0)

    window.close()


def test_metrics_window_deep_dive_views_refresh_without_stale_data(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})
    window.tabs.setCurrentWidget(window._channel_tab)
    window.angular_map_combo.setCurrentIndex(window.angular_map_combo.findData("aoa"))
    window.pair_metric_combo.setCurrentIndex(window.pair_metric_combo.findData("mean_delay"))

    window._refresh()

    assert "AoA" in window.angular_heatmap_plot.getPlotItem().titleLabel.text
    assert "Polar" in window.angular_heatmap_plot.getPlotItem().titleLabel.text
    assert len(window.angular_polar_scatter.points()) == 3
    assert window.pair_heatmap.image is not None
    assert len(window.delay_power_scatter.points()) == 3
    assert "Mean Delay" in window.pair_heatmap_plot.getPlotItem().titleLabel.text

    window.close()


def test_metrics_window_reuses_reflection_order_brushes(qapp):
    window = MetricsWindow(update_hz=8)
    orders = np.array([0.0, 0.0, 1.0, 6.0, np.nan])

    first = window._order_brushes(orders)
    second = window._order_brushes(orders)

    assert first[0] is first[1]
    assert first[0] is second[0]
    assert first[2] is second[2]
    assert first[3] is first[4]
    assert first[3] is second[3]
    assert first[0].color().name() == "#2f7ed8"
    assert first[2].color().name() == "#5cb85c"
    assert first[3].color().name() == "#888888"

    window.close()


def test_metrics_window_single_pair_count_remains_visible(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_view_model(), _frame_stats(), context={})
    window.tabs.setCurrentWidget(window._channel_tab)

    window._refresh()

    model = window.pair_count_table.model()
    assert model.rowCount() == 1
    assert tuple(model.data(model.index(0, column)) for column in range(3)) == (
        "TX2",
        "RX2",
        "1",
    )
    assert window.pair_stack.currentWidget() is window.pair_count_panel
    assert window.pair_heatmap.image is None
    window.close()


def test_dense_power_delay_profile_honors_configurable_marker_limit(qapp):
    marker_limit = 750
    count = marker_limit + 101
    delays = np.linspace(0.0, 1000.0, count)
    gains_db = np.linspace(-60.0, -160.0, count)
    stats = FrameStats(
        total_paths=count,
        orders_hist={0: count},
        delay_range_ns=(0.0, 1000.0),
        path_loss_range=(60.0, 160.0),
        binned_power_delay_profile=(delays, gains_db),
    )
    window = MetricsWindow(update_hz=8)
    window.plot_point_limit_spin.setValue(marker_limit)
    window._last_stats = stats

    window._update_binned_pdp()

    assert len(window.binned_pdp_dots.points()) == marker_limit
    assert len(window.binned_pdp_stems.opts["x"]) == marker_limit
    assert any("statistics use all selected paths" in msg for msg in window._plot_messages)

    window.close()


def test_marker_limit_uniformly_samples_scatter_without_changing_source_data(qapp):
    window = MetricsWindow(update_hz=8)
    marker_limit = 200
    path_count = 503
    window.plot_point_limit_spin.setValue(marker_limit)
    source = {
        "path_id": np.arange(path_count, dtype=np.int64),
        "delay": np.linspace(0.0, 50.0, path_count),
    }

    sampled = window._sample_plot_data(source)

    assert sampled["path_id"].size == marker_limit
    assert sampled["path_id"][0] == 0
    assert sampled["path_id"][-1] == path_count - 1
    assert source["path_id"].size == path_count
    assert "Statistics and CSV export" in window.plot_point_limit_spin.toolTip()
    window.close()


def test_no_marker_limit_keeps_every_scatter_and_profile_point(qapp):
    window = MetricsWindow(update_hz=8)
    window.plot_point_limit_spin.setValue(0)
    source = {
        "path_id": np.arange(750, dtype=np.int64),
        "delay": np.linspace(0.0, 50.0, 750),
    }
    delays = np.linspace(0.0, 50.0, 750)
    gains = np.linspace(-60.0, -120.0, 750)

    sampled = window._sample_plot_data(source)
    profile_delays, profile_gains = window._sample_profile_for_display(
        delays,
        gains,
        label="PDP",
    )

    assert window.plot_point_limit_spin.text() == "No limit"
    assert sampled["path_id"].size == 750
    np.testing.assert_array_equal(profile_delays, delays)
    np.testing.assert_array_equal(profile_gains, gains)
    window.close()


def test_disabling_adaptive_axes_freezes_current_overview_ranges(qapp):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})
    window._refresh()
    delay_range = np.asarray(window.delay_plot.viewRange(), dtype=float)
    pdp_range = np.asarray(window.binned_pdp_plot.viewRange(), dtype=float)

    window.auto_range_cb.setChecked(False)
    canonical = _deep_canonical_frame()
    canonical.path_delays = np.array([500.0, 600.0, 700.0])
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.ones(3, dtype=bool),
    )
    stats = FrameStats(
        total_paths=3,
        orders_hist={0: 1, 1: 1, 2: 1},
        delay_range_ns=(500.0, 700.0),
        path_loss_range=(60.0, 90.0),
        binned_power_delay_profile=(
            np.array([500.0, 600.0, 700.0]),
            np.array([-60.0, -80.0, -90.0]),
        ),
    )
    window.enqueue(view_model, stats, context={})
    window._refresh()

    np.testing.assert_allclose(window.delay_plot.viewRange(), delay_range)
    np.testing.assert_allclose(window.binned_pdp_plot.viewRange(), pdp_range)
    assert "does not scan other scenario frames" in window.auto_range_cb.toolTip()
    window.close()


def test_overview_axis_freeze_does_not_disable_channel_or_material_fitting(qapp, monkeypatch):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})
    window.auto_range_cb.setChecked(False)
    channel_ranges = []
    material_ranges = []
    monkeypatch.setattr(
        window.delay_power_plot,
        "setXRange",
        lambda minimum, maximum: channel_ranges.append((minimum, maximum)),
    )
    monkeypatch.setattr(
        window.material_power_plot,
        "setXRange",
        lambda minimum, maximum: material_ranges.append((minimum, maximum)),
    )

    window._update_delay_power_scatter()
    window._update_material_breakdown()

    assert channel_ranges
    assert material_ranges
    window.close()


def test_channel_layout_keeps_pair_view_and_summary_to_half_width(qapp):
    window = MetricsWindow(update_hz=8)
    channel_grid = window._channel_tab.layout().itemAt(0).layout()

    assert channel_grid.getItemPosition(channel_grid.indexOf(window.delay_power_plot)) == (
        0,
        0,
        1,
        1,
    )
    assert channel_grid.getItemPosition(channel_grid.indexOf(window.angular_heatmap_plot)) == (
        0,
        1,
        1,
        1,
    )
    assert channel_grid.getItemPosition(channel_grid.indexOf(window.pair_stack)) == (1, 0, 1, 1)
    assert channel_grid.getItemPosition(channel_grid.indexOf(window.channel_summary_label)) == (
        1,
        1,
        1,
        1,
    )

    window.close()


def test_metrics_window_redraws_only_the_visible_tab(qapp, monkeypatch):
    window = MetricsWindow(update_hz=8)
    assert not hasattr(window, "channel_tab_cb")
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})
    calls = []

    def record(name):
        return lambda *_args, **_kwargs: calls.append(name)

    overview_updaters = (
        "_update_delay_histogram",
        "_update_loss_histogram",
        "_update_order_distribution",
        "_update_binned_pdp",
        "_update_angle_distribution",
    )
    channel_updaters = (
        "_update_channel_summary",
        "_update_delay_power_scatter",
        "_update_angular_heatmap",
        "_update_pair_heatmap",
    )
    for name in (*overview_updaters, *channel_updaters, "_update_material_breakdown"):
        monkeypatch.setattr(window, name, record(name))

    window.tabs.setCurrentWidget(window._channel_tab)
    window._refresh()
    assert calls == list(channel_updaters)

    calls.clear()
    window.tabs.setCurrentWidget(window._overview_tab)
    window._refresh()
    assert calls == list(overview_updaters)

    calls.clear()
    window.tabs.setCurrentWidget(window._materials_tab)
    window._refresh()
    assert calls == ["_update_material_breakdown"]
    window.close()


def test_metrics_window_materials_tab_can_be_disabled(qapp, monkeypatch):
    window = MetricsWindow(update_hz=8)
    window.enqueue(_deep_view_model(), _deep_frame_stats(), context={})

    def fail(*_args, **_kwargs):
        raise AssertionError("disabled heavy tab should not refresh")

    window.tabs.setCurrentWidget(window._materials_tab)
    window.materials_tab_cb.setChecked(False)
    monkeypatch.setattr(window, "_update_material_breakdown", fail)
    window._refresh()
    assert "Materials tab disabled" in window.data_status_label.text()

    window.close()
