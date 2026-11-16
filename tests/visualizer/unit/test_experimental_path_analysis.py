"""Analytic, renderer-free checks of frame-local path experiment kernels."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from tests.visualizer.unit.mpc_explorer_test_data import make_test_catalog
from visualizer.src.experiments.path_analysis import (
    SPEED_OF_LIGHT_M_PER_NS,
    PathWeightMode,
    arrival_gate,
    delay_angle_histogram,
    geometric_progress,
    select_paths,
    weighted_delay_percentiles,
)
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog, MpcPathScope


def _catalog(
    delays: list[float],
    *,
    losses: list[float] | None = None,
    angles: list[float] | None = None,
    estimated_loss: list[bool] | None = None,
    known_loss: bool = True,
    known_delay: bool = True,
    points: list[list[tuple[float, float, float]]] | None = None,
) -> MpcPathCatalog:
    """Make canonical synthetic data without importing a renderer or RF runtime."""
    count = len(delays)
    if points is None:
        points = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)] for _ in delays]
    counts = np.array([len(group) for group in points], dtype=np.int32)
    starts = (
        np.cumsum(np.r_[0, counts[:-1]], dtype=np.int32) if count else np.array([], dtype=np.int32)
    )
    point_ids = np.repeat(np.arange(count, dtype=np.int32), counts)
    lines = [
        (int(start + offset), int(start + offset + 1))
        for start, size in zip(starts, counts)
        for offset in range(size - 1)
    ]
    path_delays = np.asarray(delays, dtype=np.float64)
    path_losses = np.asarray(losses if losses is not None else np.zeros(count), dtype=np.float64)
    canonical = CanonicalStepData(
        points=np.asarray([point for group in points for point in group], dtype=np.float64).reshape(
            -1, 3
        ),
        lines=np.asarray(lines, dtype=np.int32).reshape(-1, 2),
        order=np.zeros(point_ids.size, dtype=np.uint8),
        itype=np.zeros(point_ids.size, dtype=np.uint8),
        delay=np.repeat(path_delays, counts),
        loss=np.repeat(path_losses, counts),
        path_id=point_ids,
        path_start_indices=starts,
        path_delays=path_delays,
        path_losses=path_losses,
        path_orders=np.maximum(counts - 2, 0).astype(np.uint8),
        path_tx=np.zeros(count, dtype=np.int16),
        path_rx=np.zeros(count, dtype=np.int16),
        path_delay_is_estimated=np.zeros(count, dtype=bool) if known_delay else None,
        path_loss_is_estimated=(
            np.asarray(estimated_loss, dtype=bool)
            if estimated_loss is not None
            else np.zeros(count, dtype=bool) if known_loss else None
        ),
        path_aoa_az=np.asarray(angles, dtype=np.float32) if angles is not None else None,
    )
    return MpcPathCatalog(canonical, validate=True)


def _assert_immutable_arrays(snapshot: object) -> None:
    for field in fields(snapshot):
        array = getattr(snapshot, field.name)
        if not isinstance(array, np.ndarray):
            continue
        assert not array.flags.writeable, field.name
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [(MpcPathScope.ALL, [0, 1, 2]), ("filtered", [0, 1]), ("rendered", [0, 2])],
)
def test_select_exact_pair_and_scope_preserves_canonical_ids(scope, expected) -> None:
    catalog = make_test_catalog()
    result = select_paths(catalog, tx_id=0, rx_id=0, scope=scope)
    np.testing.assert_array_equal(result.path_ids, expected)
    assert result.tx_id == result.rx_id == 0
    assert result.scope is MpcPathScope(scope)
    np.testing.assert_array_equal(result.weights, np.ones(len(expected)))
    np.testing.assert_array_equal(result.delays_ns, catalog.delays_ns[expected])
    assert result.path_count == result.weighted_path_count == len(expected)
    assert result.weight_label == "Path count"


@pytest.mark.parametrize("bad_id", [-1, True, 0.0, "0", None])
@pytest.mark.parametrize("axis", ["tx_id", "rx_id"])
def test_selection_rejects_missing_or_noninteger_pair_ids(bad_id, axis) -> None:
    kwargs = {"tx_id": 0, "rx_id": 0, axis: bad_id}
    with pytest.raises(ValueError):
        select_paths(make_test_catalog(), **kwargs)


def test_selection_detaches_arrays_without_freezing_or_mutating_catalog() -> None:
    catalog = make_test_catalog()
    result = select_paths(catalog, tx_id=0, rx_id=0)
    _assert_immutable_arrays(result)
    with pytest.raises(FrozenInstanceError):
        result.tx_id = 2
    assert catalog.delays_ns.flags.writeable
    assert not np.shares_memory(result.delays_ns, catalog.delays_ns)
    catalog.delays_ns[0] = 999.0
    catalog.canonical_data.path_aoa_az[0] = 42.0
    assert result.delays_ns[0] == 10.0
    assert result.angles_deg("aoa_azimuth")[0] == 0.0
    with pytest.raises(ValueError, match="angle must"):
        result.angles_deg("bearing")


def test_power_is_relative_exported_loss_only_and_excludes_estimates() -> None:
    catalog = _catalog(
        [1, 2, 3, 4, 5, 6],
        losses=[80, 90, 100, -100, np.nan, np.inf],
        estimated_loss=[False, False, False, True, False, False],
    )
    data = select_paths(catalog, tx_id=0, rx_id=0, weight_mode="power")
    np.testing.assert_allclose(data.weights, [1, 0.1, 0.01, 0, 0, 0])
    assert data.path_count == 6 and data.weighted_path_count == 3
    assert data.weight_mode is PathWeightMode.POWER
    assert "strongest path = 1" in data.weight_label
    assert "1 path(s): loss estimated." in data.disclosures
    assert "2 path(s): loss missing/nonfinite." in data.disclosures
    assert any("3 path(s) excluded from power" in text for text in data.disclosures)
    np.testing.assert_array_equal(data.path_loss_provenance_codes, [0, 0, 0, 1, -1, -1])


def test_missing_provenance_never_silently_becomes_exported_power() -> None:
    catalog = _catalog([1, 2, np.nan], losses=[0, 10, 20], known_loss=False, known_delay=False)
    power = select_paths(catalog, tx_id=0, rx_id=0, weight_mode="power")
    count = select_paths(catalog, tx_id=0, rx_id=0, weight_mode="count")
    np.testing.assert_array_equal(power.weights, [0, 0, 0])
    np.testing.assert_array_equal(count.weights, [1, 1, 1])
    assert "3 path(s): loss unknown provenance." in power.disclosures
    assert "2 path(s): delay unknown provenance." in power.disclosures
    assert "1 path(s): delay missing/nonfinite." in power.disclosures
    assert "3 path(s): aoa azimuth missing/nonfinite." in power.disclosures
    assert np.isnan(weighted_delay_percentiles(power)).all()
    np.testing.assert_array_equal(weighted_delay_percentiles(count), [1, 1, 2])


def test_power_extreme_loss_is_stable_and_underflow_is_disclosed() -> None:
    data = select_paths(
        _catalog([0, 1, 2], losses=[-1e308, -1e308, 1e308]),
        tx_id=0,
        rx_id=0,
        weight_mode="power",
    )
    np.testing.assert_array_equal(data.weights, [1, 1, 0])
    assert any("1 exported-loss proxy weight(s) underflowed" in text for text in data.disclosures)


def test_gate_is_inclusive_and_share_uses_whole_scope_weight() -> None:
    data = select_paths(_catalog([10, 20, 30, np.nan]), tx_id=0, rx_id=0)
    gate = arrival_gate(data, 10, 20)
    np.testing.assert_array_equal(gate.path_ids, [0, 1])
    np.testing.assert_array_equal(gate.row_indices, [0, 1])
    assert gate.total_weight == 2
    assert gate.weight_share == 0.5
    assert gate.path_count == gate.weighted_path_count == 2
    assert (gate.p10_ns, gate.p50_ns, gate.p90_ns) == (10, 10, 20)
    _assert_immutable_arrays(gate)
    np.testing.assert_array_equal(arrival_gate(data, 20, 20).path_ids, [1])


def test_gate_retains_zero_weight_rows_and_reports_no_weighted_summary() -> None:
    data = select_paths(
        _catalog([10, 20, np.nan], losses=[0, 10, 0], estimated_loss=[False, True, False]),
        tx_id=0,
        rx_id=0,
        weight_mode="power",
    )
    gate = arrival_gate(data, 10, 20)
    assert gate.path_count == 2 and gate.weighted_path_count == 1
    assert gate.weight_share == 0.5
    np.testing.assert_array_equal(gate.percentiles_ns, [10, 10, 10])
    empty_weight = arrival_gate(data, 20, 20)
    assert empty_weight.path_count == 1 and empty_weight.total_weight == 0
    assert empty_weight.weight_share == 0
    assert np.isnan(empty_weight.percentiles_ns).all()


def test_inverse_cdf_quantiles_are_observed_delays_and_support_row_subset() -> None:
    data = select_paths(_catalog([30, 10, 20, 20]), tx_id=0, rx_id=0)
    result = weighted_delay_percentiles(data, (0, 25, 25.01, 50, 75, 100))
    np.testing.assert_array_equal(result, [10, 10, 20, 20, 20, 30])
    np.testing.assert_array_equal(
        weighted_delay_percentiles(data, row_indices=np.array([0, 2])), [20, 20, 30]
    )
    assert not result.flags.writeable
    assert weighted_delay_percentiles(data, ()).size == 0


def test_weighted_quantiles_use_linear_power_not_db_or_amplitude() -> None:
    data = select_paths(
        _catalog([1, 2, 3], losses=[20, 10, 0]), tx_id=0, rx_id=0, weight_mode="power"
    )
    np.testing.assert_array_equal(weighted_delay_percentiles(data), [3, 3, 3])


@pytest.mark.parametrize("bounds", [(20, 10), (np.nan, 10), (0, np.inf)])
def test_gate_rejects_invalid_bounds(bounds) -> None:
    data = select_paths(_catalog([0]), tx_id=0, rx_id=0)
    with pytest.raises(ValueError, match="gate bounds"):
        arrival_gate(data, *bounds)
    with pytest.raises(ValueError, match="gate bounds"):
        delay_angle_histogram(data, gate_ns=bounds)


@pytest.mark.parametrize("levels", [(-1,), (101,), (np.nan,), (np.inf,), ((10, 20),)])
def test_percentiles_validate_levels(levels) -> None:
    with pytest.raises(ValueError, match="percentiles"):
        weighted_delay_percentiles(select_paths(_catalog([0]), tx_id=0, rx_id=0), levels)


def test_joint_bins_wrap_angles_and_preserve_boundary_membership() -> None:
    data = select_paths(
        _catalog([0, 10, 20, 20, 20, 30, 40], angles=[-180, 180, 540, -540, 0, 90, -90]),
        tx_id=0,
        rx_id=0,
    )
    histogram = delay_angle_histogram(data, angular_bin_width_deg=90, delay_bin_width_ns=10)
    np.testing.assert_array_equal(histogram.delay_edges_ns, [0, 10, 20, 30, 40, 50])
    np.testing.assert_array_equal(histogram.angle_edges_deg, [-180, -90, 0, 90, 180])
    np.testing.assert_array_equal(histogram.delay_bin_indices, [0, 1, 2, 2, 2, 3, 4])
    np.testing.assert_array_equal(histogram.angle_bin_indices, [0, 0, 0, 0, 2, 3, 1])
    np.testing.assert_array_equal(histogram.path_ids_for_bin(2, 0), [2, 3])
    np.testing.assert_array_equal(histogram.row_indices_for_bin(2, 0), [2, 3])
    assert histogram.delay_interval_ns(2) == (20, 30)
    assert histogram.angle_interval_deg(0) == (-180, -90)
    assert histogram.total_weight == 7
    assert histogram.dominant_bin_share == pytest.approx(2 / 7)
    assert histogram.occupied_bins == histogram.min_bins_90 == 6
    _assert_immutable_arrays(histogram)


def test_angle_wrap_does_not_round_near_positive_boundary_into_negative_bin() -> None:
    data = select_paths(_catalog([0, 0], angles=[0, 0]), tx_id=0, rx_id=0)
    data = replace(data, aoa_azimuth_deg=np.array([np.nextafter(180.0, -np.inf), 180.0]))
    histogram = delay_angle_histogram(data, angular_bin_width_deg=90)
    np.testing.assert_array_equal(histogram.angle_bin_indices, [3, 0])


def test_terminal_partial_angle_bin_and_negative_delay_edges() -> None:
    data = select_paths(_catalog([-11, -10, 0], angles=[170, -80, 179]), tx_id=0, rx_id=0)
    histogram = delay_angle_histogram(data, angular_bin_width_deg=100, delay_bin_width_ns=10)
    np.testing.assert_array_equal(histogram.angle_edges_deg, [-180, -80, 20, 120, 180])
    np.testing.assert_array_equal(histogram.delay_edges_ns, [-20, -10, 0, 10])
    np.testing.assert_array_equal(histogram.angle_bin_indices, [3, 1, 3])
    np.testing.assert_array_equal(histogram.delay_bin_indices, [0, 1, 2])
    one_bin = delay_angle_histogram(data, angular_bin_width_deg=1000)
    np.testing.assert_array_equal(one_bin.angle_edges_deg, [-180, 180])


def test_bins_disclose_exclusions_and_gate_denominator_is_explicit() -> None:
    data = select_paths(
        _catalog(
            [0, 1, 2, 3, 4, np.nan], angles=[0, 0, np.nan, 0, 0, 0], losses=[0, 10, 0, np.inf, 0, 0]
        ),
        tx_id=0,
        rx_id=0,
        weight_mode="power",
    )
    histogram = delay_angle_histogram(data, gate_ns=(0, 3))
    np.testing.assert_array_equal(histogram.path_ids, [0, 1])
    assert histogram.source_path_count == 6
    assert histogram.gated_path_count == 4
    assert histogram.excluded_path_count == 2
    assert histogram.total_weight == pytest.approx(1.1)
    assert histogram.dominant_bin_share == 1
    assert histogram.occupied_bins == histogram.min_bins_90 == 1
    assert any("2 in-scope/gate path(s) excluded" in text for text in histogram.disclosures)
    assert any("not physical clusters" in text for text in histogram.disclosures)


def test_bin_row_mapping_is_distinct_from_canonical_id_after_pair_selection() -> None:
    data = select_paths(make_test_catalog(), tx_id=1, rx_id=1)
    histogram = delay_angle_histogram(data, angular_bin_width_deg=360)
    np.testing.assert_array_equal(histogram.path_ids_for_bin(0, 0), [6, 7])
    np.testing.assert_array_equal(histogram.row_indices_for_bin(0, 0), [0, 1])


def test_empty_selection_and_empty_bins_have_explicit_zero_summaries() -> None:
    data = select_paths(_catalog([]), tx_id=0, rx_id=0)
    assert data.path_count == data.weighted_path_count == 0
    gate = arrival_gate(data, 0, 0)
    assert gate.path_count == 0 and gate.weight_share == 0
    histogram = delay_angle_histogram(data, angular_bin_width_deg=360, delay_bin_width_ns=5)
    np.testing.assert_array_equal(histogram.values, [[0]])
    np.testing.assert_array_equal(histogram.delay_edges_ns, [0, 5])
    assert histogram.total_weight == histogram.dominant_bin_share == 0
    assert histogram.occupied_bins == histogram.min_bins_90 == 0
    assert np.isnan(gate.percentiles_ns).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angular_bin_width_deg": 0},
        {"angular_bin_width_deg": np.nan},
        {"delay_bin_width_ns": -1},
        {"delay_bin_width_ns": np.inf},
        {"max_cells": 0},
        {"max_cells": 1_048_577},
        {"max_cells": True},
        {"angular_bin_width_deg": 1e-300},
        {"delay_bin_width_ns": 1e-300},
        {"angular_bin_width_deg": 90, "delay_bin_width_ns": 10, "max_cells": 7},
    ],
)
def test_histogram_rejects_invalid_or_overbudget_grids(kwargs) -> None:
    data = select_paths(_catalog([0, 10], angles=[0, 0]), tx_id=0, rx_id=0)
    with pytest.raises(ValueError):
        delay_angle_histogram(data, **kwargs)


def test_histogram_accepts_exact_cell_budget_without_changing_resolution() -> None:
    data = select_paths(_catalog([0, 10], angles=[0, 0]), tx_id=0, rx_id=0)
    result = delay_angle_histogram(
        data, angular_bin_width_deg=90, delay_bin_width_ns=10, max_cells=8
    )
    assert result.values.shape == (2, 4)
    np.testing.assert_array_equal(np.diff(result.delay_edges_ns), [10, 10])


def test_decimal_delay_edges_contain_every_reported_bin_member() -> None:
    width = 0.21951496601488152
    delays = np.array([371, -446, -971, -834, 945, 791, -394, -141, -519, -705]) * width
    data = select_paths(_catalog(delays.tolist(), angles=[0] * len(delays)), tx_id=0, rx_id=0)
    histogram = delay_angle_histogram(data, angular_bin_width_deg=360, delay_bin_width_ns=width)
    lower = histogram.delay_edges_ns[histogram.delay_bin_indices]
    upper = histogram.delay_edges_ns[histogram.delay_bin_indices + 1]
    assert np.all(lower <= delays)
    assert np.all(delays < upper)
    assert histogram.total_weight == len(delays)


@pytest.mark.parametrize("indices", [(-1, 0), (0, -1), (1, 0), (0, 36)])
def test_bin_mapping_rejects_out_of_range_indices(indices) -> None:
    histogram = delay_angle_histogram(select_paths(_catalog([0], angles=[0]), tx_id=0, rx_id=0))
    with pytest.raises(IndexError):
        histogram.path_ids_for_bin(*indices)


def test_geometry_uses_arc_length_over_c_not_exported_delay_or_endpoint_distance() -> None:
    catalog = _catalog([9999], points=[[(0, 0, 0), (3, 0, 0), (3, 4, 0)]])
    result = geometric_progress(catalog, np.array([0]), 5 / SPEED_OF_LIGHT_M_PER_NS)
    np.testing.assert_allclose(result.positions, [[3, 2, 0]])
    np.testing.assert_allclose(result.geometric_arrival_ns, [7 / SPEED_OF_LIGHT_M_PER_NS])
    np.testing.assert_array_equal(result.valid, [True])
    np.testing.assert_array_equal(result.active, [True])
    np.testing.assert_array_equal(result.arrived, [False])
    _assert_immutable_arrays(result)


@pytest.mark.parametrize("time", [-1, 0, 1, 100])
def test_geometry_clamps_endpoints_and_handles_repeated_zero_length_segments(time) -> None:
    catalog = _catalog(
        [9, 9],
        points=[[(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)], [(4, 5, 6), (4, 5, 6)]],
    )
    result = geometric_progress(catalog, np.array([0, 1]), time)
    np.testing.assert_allclose(
        result.positions[0], [np.clip(time * SPEED_OF_LIGHT_M_PER_NS, 0, 1), 0, 0]
    )
    np.testing.assert_array_equal(result.positions[1], [4, 5, 6])
    assert bool(result.arrived[1]) is (time >= 0)
    assert not result.active[1]
    assert bool(result.active[0]) is (0 <= time < 1 / SPEED_OF_LIGHT_M_PER_NS)
    endpoint = geometric_progress(catalog, np.array([0]), 1 / SPEED_OF_LIGHT_M_PER_NS)
    np.testing.assert_array_equal(endpoint.positions, [[1, 0, 0]])
    assert endpoint.arrived[0] and not endpoint.active[0]


def test_geometry_invalid_path_does_not_contaminate_other_paths() -> None:
    catalog = _catalog(
        [1, 2, 3],
        points=[[(0, 0, 0), (np.nan, 0, 0)], [(0, 0, 0), (np.inf, 0, 0)], [(0, 0, 0), (2, 0, 0)]],
    )
    result = geometric_progress(catalog, np.array([0, 1, 2]), 1 / SPEED_OF_LIGHT_M_PER_NS)
    np.testing.assert_array_equal(result.valid, [False, False, True])
    assert np.isnan(result.positions[:2]).all()
    assert np.isnan(result.geometric_arrival_ns[:2]).all()
    np.testing.assert_allclose(result.positions[2], [1, 0, 0])
    assert not result.arrived[:2].any() and not result.active[:2].any()
    assert "2 path(s) have nonfinite/invalid geometry." in result.disclosures


def test_geometry_path_lengths_are_independent_of_other_paths_scale_and_order() -> None:
    catalog = _catalog([1, 2], points=[[(0, 0, 0), (1e20, 0, 0)], [(0, 0, 0), (1, 0, 0)]])
    result = geometric_progress(catalog, np.array([0, 1]), 0.5 / SPEED_OF_LIGHT_M_PER_NS)
    np.testing.assert_allclose(result.positions, [[0.5, 0, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(
        result.geometric_arrival_ns, np.array([1e20, 1]) / SPEED_OF_LIGHT_M_PER_NS
    )
    reversed_result = geometric_progress(catalog, np.array([1, 0]), result.time_ns)
    np.testing.assert_allclose(result.positions, reversed_result.positions[::-1])


def test_geometry_caps_are_deterministic_prefixes_and_do_not_skip_large_paths() -> None:
    catalog = _catalog([0, 0, 0], points=[[(0, 0, 0)] * 2, [(0, 0, 0)] * 4, [(0, 0, 0)] * 2])
    ids = np.array([2, 1, 0, 2])
    point_cap = geometric_progress(catalog, ids, 0, max_points=5)
    np.testing.assert_array_equal(point_cap.path_ids, [2])
    assert point_cap.omitted_count == 3
    path_cap = geometric_progress(catalog, ids, 0, max_paths=2)
    np.testing.assert_array_equal(path_cap.path_ids, [2, 1])
    assert path_cap.omitted_count == 2
    none = geometric_progress(catalog, np.array([1, 0]), 0, max_points=2)
    assert none.path_ids.size == 0 and none.positions.shape == (0, 3)
    assert none.omitted_count == 2
    assert any("omitted" in text for text in none.disclosures)


@pytest.mark.parametrize("bad_ids", [[-1], [2], [0.0], [True], [[0]]])
def test_geometry_and_quantiles_reject_bad_indices(bad_ids) -> None:
    catalog = _catalog([0, 0])
    data = select_paths(catalog, tx_id=0, rx_id=0)
    with pytest.raises(ValueError):
        geometric_progress(catalog, np.asarray(bad_ids), 0)
    with pytest.raises(ValueError):
        weighted_delay_percentiles(data, row_indices=np.asarray(bad_ids))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_ns": np.nan},
        {"time_ns": np.inf},
        {"max_paths": 0},
        {"max_paths": 100_001},
        {"max_points": 1},
        {"max_points": 2_000_001},
        {"max_points": 3.0},
    ],
)
def test_geometry_rejects_invalid_time_and_budgets(kwargs) -> None:
    with pytest.raises(ValueError):
        geometric_progress(_catalog([0]), np.array([0]), **{"time_ns": 0, **kwargs})


def test_geometry_empty_selection_has_no_invented_markers() -> None:
    result = geometric_progress(_catalog([]), np.array([], dtype=np.int32), 0)
    assert result.positions.shape == (0, 3)
    assert result.path_ids.size == result.omitted_count == 0


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("moment", [0.0, 1.0, np.finfo(np.float64).max])
def test_geometry_normal_and_extreme_values_do_not_emit_runtime_warnings(moment) -> None:
    catalog = _catalog(
        [0, 0, 0, 0],
        points=[
            [(0, 0, 0), (1, 0, 0)],
            [(0, 0, 0), (1e200, 0, 0)],
            [(-1e308, 0, 0), (1e308, 0, 0)],
            [(0, 0, 0), (1e308, 0, 0)],
        ],
    )
    result = geometric_progress(catalog, np.arange(4), moment)
    np.testing.assert_array_equal(result.valid, [True, True, False, False])
    assert np.isfinite(result.positions[:2]).all()
    assert np.isnan(result.positions[2:]).all()
    assert np.isnan(result.geometric_arrival_ns[2:]).all()


def test_geometry_vectorized_result_matches_independent_polyline_oracle() -> None:
    rng = np.random.default_rng(701)
    groups = [np.cumsum(rng.normal(size=(depth, 3)), axis=0).tolist() for depth in (2, 3, 5, 8, 11)]
    catalog = _catalog([1000] * len(groups), points=groups)
    ids = np.array([4, 0, 2, 1, 4, 3])
    for distance in (0.01, 0.5, 2, 5, 20):
        result = geometric_progress(catalog, ids, distance / SPEED_OF_LIGHT_M_PER_NS)
        for output_row, path_id in enumerate(ids):
            points = np.array(groups[path_id])
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.r_[0, np.cumsum(lengths)]
            target = min(distance, cumulative[-1])
            segment = min(
                int(np.searchsorted(cumulative, target, side="right")) - 1, len(lengths) - 1
            )
            fraction = (target - cumulative[segment]) / lengths[segment]
            expected = points[segment] + fraction * (points[segment + 1] - points[segment])
            np.testing.assert_allclose(result.positions[output_row], expected, atol=1e-13)
            assert result.geometric_arrival_ns[output_row] == pytest.approx(
                cumulative[-1] / SPEED_OF_LIGHT_M_PER_NS
            )
