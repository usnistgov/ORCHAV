from __future__ import annotations

import time

import numpy as np
import pytest

from visualizer.src.coverage import analysis as coverage_analysis
from visualizer.src.coverage.analysis import (
    build_coverage_isoline,
    compute_coverage_scalar_plot_data,
    compute_coverage_slice_summary,
    compute_coverage_threshold_mask,
    compute_coverage_threshold_summary,
    compute_serving_tx_coverage_summary,
    coverage_metric_color_scale,
    coverage_metric_colormap,
    coverage_metric_comparator,
    coverage_metric_label,
    coverage_metric_valid_mask,
    coverage_values_for_height,
    default_coverage_threshold,
    format_coverage_slice_summary,
    format_coverage_threshold_summary,
    format_coverage_value,
    format_serving_tx_coverage_summary,
    supports_coverage_threshold,
)


def test_scalar_plot_data_uses_metric_direction_and_total_slice_area():
    path_loss = compute_coverage_scalar_plot_data(
        {
            "metric_name": "path_loss_db",
            "values_3d": np.array([[[80.0, 100.0, 100.0, np.nan]]], dtype=np.float32),
        },
        height_index=0,
    )
    rss = compute_coverage_scalar_plot_data(
        {
            "metric_name": "rss_dbm",
            "values_3d": np.array([[[-100.0, -80.0, -80.0, np.nan]]], dtype=np.float32),
        },
        height_index=0,
    )

    assert path_loss.comparator == "<="
    np.testing.assert_allclose(path_loss.thresholds, [80.0, 100.0])
    np.testing.assert_allclose(path_loss.qualifying_percent_total, [25.0, 75.0])
    assert rss.comparator == ">="
    np.testing.assert_allclose(rss.thresholds, [-100.0, -80.0])
    np.testing.assert_allclose(rss.qualifying_percent_total, [75.0, 50.0])


def test_scalar_plot_data_filters_invalid_logarithmic_values_and_bounds_curve():
    values = np.concatenate(
        (
            np.array([np.nan, -1.0, 0.0], dtype=np.float32),
            np.arange(1.0, 101.0, dtype=np.float32),
        )
    ).reshape(1, 1, -1)

    plot_data = compute_coverage_scalar_plot_data(
        {"metric_name": "rss_w", "values_3d": values},
        height_index=0,
        max_curve_points=8,
    )

    assert plot_data.color_scale == "logarithmic"
    assert plot_data.valid_values.size == 100
    assert plot_data.thresholds.size == 8
    assert np.all(plot_data.thresholds > 0.0)


def test_file_backed_selected_slice_uses_active_array_and_logical_height_label_index():
    active = np.array([[[10.0, 20.0]]], dtype=np.float32)
    coverage_data = {
        "coverage_file": "coverage_maps.h5",
        "_active_height_index": 3,
        "values_3d": active,
    }

    selected = coverage_values_for_height(coverage_data, height_index=3)

    np.testing.assert_array_equal(selected, active[0])


def test_path_loss_threshold_counts_lower_values_as_covered():
    coverage_data = {
        "metric_name": "best_path_loss_db",
        "grid_spacing": np.array([2.0, 3.0], dtype=np.float32),
        "values_3d": np.array([[[80.0, 110.0], [130.0, np.nan]]], dtype=np.float32),
    }

    summary = compute_coverage_threshold_summary(
        coverage_data,
        height_index=0,
        threshold=110.0,
    )

    assert summary.comparator == "<="
    assert summary.valid_cells == 3
    assert summary.covered_cells == 2
    assert summary.cell_area_m2 == pytest.approx(6.0)
    assert summary.covered_area_m2 == pytest.approx(12.0)
    assert summary.covered_percent_valid == pytest.approx(66.6666667)
    assert summary.covered_percent_total == pytest.approx(50.0)
    assert format_coverage_threshold_summary(summary) == (
        "<= 110.0: 2/3 valid cells (66.7% valid; 50.0% total), 12.0 m^2"
    )


def test_scalar_slice_summary_reports_availability_and_percentiles():
    coverage_data = {
        "metric_name": "best_path_loss_db",
        "grid_spacing": np.array([2.0, 3.0], dtype=np.float32),
        "values_3d": np.array([[[80.0, 100.0], [120.0, np.nan]]], dtype=np.float32),
    }

    summary = compute_coverage_slice_summary(coverage_data, height_index=0)

    assert summary.total_cells == 4
    assert summary.valid_cells == 3
    assert summary.no_data_cells == 1
    assert summary.valid_percent == pytest.approx(75.0)
    assert summary.no_data_percent == pytest.approx(25.0)
    assert summary.cell_area_m2 == pytest.approx(6.0)
    assert summary.percentile_10 == pytest.approx(84.0)
    assert summary.percentile_50 == pytest.approx(100.0)
    assert summary.percentile_90 == pytest.approx(116.0)
    assert format_coverage_slice_summary(summary) == (
        "Valid: 3/4 cells (75.0%); no data: 25.0% | " "P10/P50/P90: 84.0/100.0/116.0 dB"
    )


def test_serving_tx_slice_summary_reports_no_service_and_tx_area_shares():
    coverage_data = {
        "metric_name": "serving_tx",
        "grid_spacing": np.array([2.0, 2.5], dtype=np.float32),
        "tx_names": ["West", "East"],
        "serving_tx_count": 2,
        "values_3d": np.array([[[0, 1, 1], [-1, np.nan, 0]]], dtype=np.float32),
    }

    summary = compute_serving_tx_coverage_summary(coverage_data, height_index=0)

    assert summary.total_cells == 6
    assert summary.served_cells == (2, 2)
    assert summary.no_service_cells == 2
    assert summary.no_service_percent == pytest.approx(100.0 / 3.0)
    assert summary.no_service_area_m2 == pytest.approx(10.0)
    assert summary.served_percent(0) == pytest.approx(100.0 / 3.0)
    assert summary.served_area_m2(1) == pytest.approx(10.0)
    assert format_serving_tx_coverage_summary(summary) == (
        "No service: 2/6 cells (33.3%, 10.0 m^2) | "
        "West: 2/6 cells (33.3%, 10.0 m^2) | "
        "East: 2/6 cells (33.3%, 10.0 m^2)"
    )


def test_serving_tx_slice_summary_treats_out_of_range_index_as_no_service():
    coverage_data = {
        "metric_name": "serving_tx",
        "tx_names": ["West", "East"],
        "serving_tx_count": 2,
        "values_3d": np.array([[[0, 1, 2, -1]]], dtype=np.float32),
    }

    summary = compute_serving_tx_coverage_summary(coverage_data, height_index=0)

    assert summary.tx_names == ("West", "East")
    assert summary.served_cells == (1, 1)
    assert summary.no_service_cells == 2


def test_rss_threshold_counts_higher_values_as_covered():
    coverage_data = {
        "metric_name": "rss_dbm",
        "grid_spacing": np.array([1.0, 1.0], dtype=np.float32),
        "values_3d": np.array([[[-95.0, -80.0], [-75.0, -105.0]]], dtype=np.float32),
    }

    summary = compute_coverage_threshold_summary(
        coverage_data,
        height_index=0,
        threshold=-85.0,
    )

    assert summary.comparator == ">="
    assert summary.valid_cells == 4
    assert summary.covered_cells == 2
    assert summary.covered_percent_valid == pytest.approx(50.0)


def test_default_threshold_uses_selected_height_median():
    coverage_data = {
        "metric_name": "sinr_db",
        "values_3d": np.array(
            [
                [[1.0, 2.0]],
                [[10.0, 30.0]],
            ],
            dtype=np.float32,
        ),
    }

    assert default_coverage_threshold(coverage_data, height_index=1) == pytest.approx(20.0)


def test_serving_tx_is_not_thresholdable():
    assert supports_coverage_threshold("serving_tx") is False
    with pytest.raises(ValueError):
        compute_coverage_threshold_summary(
            {"metric_name": "serving_tx", "values_3d": np.array([[[0, 1]]])},
            height_index=0,
            threshold=0,
        )


def test_threshold_mask_uses_metric_comparator():
    values = np.array([[80.0, 110.0], [130.0, np.nan]], dtype=np.float32)

    path_loss_mask = compute_coverage_threshold_mask(
        values,
        metric_name="best_path_loss_db",
        threshold=110.0,
    )
    rss_mask = compute_coverage_threshold_mask(
        values,
        metric_name="rss_dbm",
        threshold=110.0,
    )

    np.testing.assert_array_equal(path_loss_mask, [[True, True], [False, False]])
    np.testing.assert_array_equal(rss_mask, [[False, True], [True, False]])


@pytest.mark.parametrize(
    ("metric_name", "expected_label", "expected_unit"),
    [
        ("path_gain_db/TX1", "Path gain (TX1)", "dB"),
        ("serving_path_gain_linear", "Serving path gain", "linear"),
        ("rss_w/TX2", "Received power (TX2)", "W"),
        ("sinr_linear", "SINR", "linear"),
    ],
)
def test_schema_scalar_metrics_have_complete_metadata(
    metric_name,
    expected_label,
    expected_unit,
):
    assert coverage_metric_label(metric_name) == (expected_label, expected_unit)
    assert coverage_metric_comparator(metric_name) == ">="


@pytest.mark.parametrize(
    ("metric_name", "expected_colormap"),
    [
        ("path_loss_db", "RdYlGn_r"),
        ("best_path_loss_db", "RdYlGn_r"),
        ("path_gain_db/TX1", "RdYlGn"),
        ("rss_dbm", "RdYlGn"),
        ("sinr_linear", "RdYlGn"),
        ("tx_margin_db", "RdYlGn"),
        ("serving_tx", None),
    ],
)
def test_metric_colormap_follows_good_value_direction(metric_name, expected_colormap):
    assert coverage_metric_colormap(metric_name) == expected_colormap


@pytest.mark.parametrize(
    ("metric_name", "expected_scale"),
    [
        ("path_gain_linear/TX1", "logarithmic"),
        ("serving_path_gain_linear", "logarithmic"),
        ("rss_w/TX1", "logarithmic"),
        ("sinr_linear", "logarithmic"),
        ("path_loss_db/TX1", "linear"),
        ("sinr_db", "linear"),
        ("serving_tx", "categorical"),
    ],
)
def test_metric_color_scale_matches_rf_quantity(metric_name, expected_scale):
    assert coverage_metric_color_scale(metric_name) == expected_scale


def test_logarithmic_threshold_helpers_exclude_nonpositive_cells():
    values = np.array([[[-1.0, 0.0, 1.0e-12, 1.0e-9]]], dtype=np.float32)
    coverage_data = {
        "metric_name": "path_gain_linear/TX1",
        "grid_spacing": np.array([1.0, 1.0], dtype=np.float32),
        "values_3d": values,
    }

    valid = coverage_metric_valid_mask(values[0], coverage_data["metric_name"])
    threshold_mask = compute_coverage_threshold_mask(
        values[0],
        metric_name=coverage_data["metric_name"],
        threshold=1.0e-10,
    )
    summary = compute_coverage_threshold_summary(
        coverage_data,
        height_index=0,
        threshold=1.0e-10,
    )

    np.testing.assert_array_equal(valid, [[False, False, True, True]])
    np.testing.assert_array_equal(threshold_mask, [[False, False, False, True]])
    assert default_coverage_threshold(coverage_data) == pytest.approx(5.005e-10)
    assert summary.valid_cells == 2
    assert summary.covered_cells == 1


def test_logarithmic_default_threshold_uses_geometric_range_center_without_values():
    coverage_data = {
        "metric_name": "rss_w/TX1",
        "value_min": 1.0e-12,
        "value_max": 1.0e-6,
        "values_3d": np.array([[[0.0, np.nan]]], dtype=np.float32),
    }

    assert default_coverage_threshold(coverage_data) == pytest.approx(1.0e-9)


def test_watt_values_use_compact_significant_digit_formatting():
    assert format_coverage_value(1.25e-6, "W") == "1.25e-06"


def test_isoline_builds_segments():
    values = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    points, lines = build_coverage_isoline(
        values,
        grid_origin=np.array([10.0, 20.0, 0.0], dtype=np.float32),
        grid_spacing=np.array([2.0, 2.0, 1.0], dtype=np.float32),
        z_level=3.0,
        level=0.5,
    )

    assert points.shape[1] == 3
    assert lines.shape[1] == 2
    assert lines.shape[0] >= 1
    np.testing.assert_allclose(points[:, 2], 3.05)


def _canonical_xy_segments(points, lines):
    return {
        frozenset(
            {
                tuple(np.round(points[int(start), :2], decimals=8)),
                tuple(np.round(points[int(end), :2], decimals=8)),
            }
        )
        for start, end in lines
    }


def test_isoline_constant_field_at_level_is_empty(monkeypatch):
    kwargs = {
        "grid_origin": np.zeros(3, dtype=np.float32),
        "grid_spacing": np.ones(3, dtype=np.float32),
        "z_level": 0.0,
        "level": 2.0,
    }
    values = np.full((3, 3), 2.0, dtype=np.float32)

    payloads = [build_coverage_isoline(values, **kwargs)]
    monkeypatch.setattr(coverage_analysis, "_contour_generator", None)
    payloads.append(build_coverage_isoline(values, **kwargs))

    for points, lines in payloads:
        assert points.shape == (0, 3)
        assert points.dtype == np.float64
        assert lines.shape == (0, 2)
        assert lines.dtype == np.int32


def test_isoline_saddle_uses_value_aware_topology():
    points, lines = build_coverage_isoline(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        grid_origin=np.zeros(3, dtype=np.float32),
        grid_spacing=np.ones(3, dtype=np.float32),
        z_level=0.0,
        level=0.75,
    )

    assert _canonical_xy_segments(points, lines) == {
        frozenset({(1.125, 0.5), (0.5, 1.125)}),
        frozenset({(1.5, 1.25), (1.25, 1.5)}),
    }


def test_isoline_fallback_preserves_saddle_topology(monkeypatch):
    monkeypatch.setattr(coverage_analysis, "_contour_generator", None)

    points, lines = build_coverage_isoline(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        grid_origin=np.zeros(3, dtype=np.float32),
        grid_spacing=np.ones(3, dtype=np.float32),
        z_level=0.0,
        level=0.75,
    )

    assert _canonical_xy_segments(points, lines) == {
        frozenset({(1.125, 0.5), (0.5, 1.125)}),
        frozenset({(1.5, 1.25), (1.25, 1.5)}),
    }


def test_isoline_fallback_handles_opposite_saddle_orientation(monkeypatch):
    monkeypatch.setattr(coverage_analysis, "_contour_generator", None)

    points, lines = build_coverage_isoline(
        np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32),
        grid_origin=np.zeros(3, dtype=np.float32),
        grid_spacing=np.ones(3, dtype=np.float32),
        z_level=0.0,
        level=0.25,
    )

    assert _canonical_xy_segments(points, lines) == {
        frozenset({(0.625, 0.5), (0.5, 0.75)}),
        frozenset({(1.5, 1.375), (1.25, 1.5)}),
    }


def test_isoline_practical_grid_performance():
    if coverage_analysis._contour_generator is None:
        pytest.skip("contourpy is unavailable")

    axis = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(axis, axis)
    values = np.hypot(x_grid, y_grid)
    kwargs = {
        "grid_origin": np.zeros(3, dtype=np.float32),
        "grid_spacing": np.ones(3, dtype=np.float32),
        "z_level": 0.0,
    }

    build_coverage_isoline(values[:16, :16], level=0.5, **kwargs)
    started = time.perf_counter()
    payloads = [
        build_coverage_isoline(values, level=level, **kwargs) for level in np.linspace(0.2, 1.2, 6)
    ]
    elapsed = time.perf_counter() - started

    assert all(lines.size for _, lines in payloads)
    assert elapsed < 1.5, f"six isolines on a 256x256 grid took {elapsed:.3f}s"
