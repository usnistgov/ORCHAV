"""Focused tests for the renderer-neutral MPC path catalog."""

from __future__ import annotations

import numpy as np
import pytest

from tests.visualizer.unit.mpc_explorer_test_data import (
    make_large_los_catalog,
    make_test_catalog,
)
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.metrics.mpc_path_catalog import (
    MpcPathCatalog,
    MpcPathCatalogError,
    MpcPathScope,
)


def test_catalog_borrows_canonical_arrays_and_defers_derived_work() -> None:
    catalog = make_test_catalog(validate=False)
    canonical = catalog.canonical_data

    assert catalog.cached_derived_columns == frozenset()
    assert catalog.path_start_indices is canonical.path_start_indices
    assert catalog.path_losses_db is canonical.path_losses
    assert catalog.delays_ns is canonical.path_delays
    assert catalog.cached_derived_columns == frozenset()

    path_ids = catalog.path_ids
    assert path_ids.dtype == np.int32
    assert "path_ids" not in catalog.cached_derived_columns
    assert catalog.path_ids is not path_ids


def test_whole_path_and_interior_slices_preserve_canonical_identity() -> None:
    catalog = make_test_catalog()

    np.testing.assert_allclose(
        catalog.path_points(2),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [7.0, -1.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(catalog.interior_bounces(0), np.empty((0, 3)))
    np.testing.assert_array_equal(catalog.interaction_sequence(2), np.array([1, 2]))
    np.testing.assert_array_equal(catalog.material_sequence(2), np.array([1, 2]))
    assert np.shares_memory(catalog.path_points(2), catalog.canonical_data.points)
    assert np.shares_memory(
        catalog.interaction_sequence(2),
        catalog.canonical_data.itype,
    )


def test_interaction_classifications_cover_los_pure_mixed_exact_and_unknown() -> None:
    catalog = make_test_catalog()

    np.testing.assert_array_equal(
        np.flatnonzero(catalog.contains_interaction(1)),
        np.array([1, 2, 5]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(catalog.pure_interaction(1)),
        np.array([1]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(catalog.pure_interaction(3)),
        np.array([7]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(catalog.mixed_interaction_mask),
        np.array([2, 5]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(catalog.exact_interaction_sequence((1, 2))),
        np.array([2]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(catalog.exact_interaction_sequence(())),
        np.array([0]),
    )


def test_material_lookup_is_constant_time_per_path_and_full_column_is_lazy() -> None:
    catalog = make_test_catalog(validate=False)

    assert catalog.first_material_id(0) == 0
    assert catalog.first_material_id(2) == 1
    assert "first_material_ids" not in catalog.cached_derived_columns
    np.testing.assert_array_equal(
        catalog.first_material_ids,
        np.array([0, 1, 1, 2, 3, 4, 0, 5], dtype=np.int16),
    )
    assert catalog.material_name(2) == "Glass"


def test_all_filtered_and_rendered_scopes_use_path_level_semantics() -> None:
    catalog = make_test_catalog()

    np.testing.assert_array_equal(catalog.scope_path_ids(MpcPathScope.ALL), np.arange(8))
    np.testing.assert_array_equal(
        catalog.scope_path_ids(MpcPathScope.FILTERED),
        np.array([0, 1, 3, 4, 5, 7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        catalog.scope_path_ids(MpcPathScope.RENDERED),
        np.array([0, 2, 3], dtype=np.int32),
    )
    assert catalog.is_rendered(2)
    assert not catalog.is_rendered(1)
    assert catalog.is_filtered(1)
    assert not catalog.is_filtered(2)


def test_disabled_bulk_path_visibility_makes_rendered_scope_empty() -> None:
    source = make_test_catalog(validate=False)
    catalog = MpcPathCatalog(
        source.canonical_data,
        filtered_path_mask=source.scope_mask(MpcPathScope.FILTERED),
        rendered_segment_mask=np.ones(source.segment_count, dtype=bool),
        rendered_paths_enabled=False,
    )

    assert not np.any(catalog.scope_mask(MpcPathScope.RENDERED))
    assert not catalog.is_rendered(0)


def test_geometric_length_and_stretch_match_path_slices() -> None:
    catalog = make_test_catalog()
    expected_lengths = np.array(
        [
            np.sum(np.linalg.norm(np.diff(catalog.path_points(path_id), axis=0), axis=1))
            for path_id in range(catalog.path_count)
        ]
    )

    np.testing.assert_allclose(catalog.geometric_lengths_m, expected_lengths)
    expected_direct = np.array(
        [
            np.linalg.norm(catalog.path_points(path_id)[-1] - catalog.path_points(path_id)[0])
            for path_id in range(catalog.path_count)
        ]
    )
    np.testing.assert_allclose(catalog.direct_distances_m, expected_direct)
    np.testing.assert_allclose(
        catalog.stretch_ratios,
        expected_lengths / expected_direct,
        rtol=2e-7,
    )
    assert catalog.stretch_ratios[0] == pytest.approx(1.0)


def test_pair_relative_metrics_are_independent_and_deterministic_with_ties() -> None:
    catalog = make_test_catalog()
    relative = catalog.pair_relative

    # Paths 0, 1, and 2 share TX/RX. Equal-loss paths 1 and 2 use path ID.
    assert relative.strength_rank[1] == 1
    assert relative.strength_rank[2] == 2
    assert relative.strength_rank[0] == 3
    assert relative.path_loss_delta_db[1] == pytest.approx(0.0)
    assert relative.path_loss_delta_db[2] == pytest.approx(0.0)
    assert relative.path_loss_delta_db[0] == pytest.approx(10.0)
    assert relative.excess_delay_ns[0] == pytest.approx(0.0)
    assert relative.excess_delay_ns[1] == pytest.approx(10.0)
    assert np.isnan(relative.excess_delay_ns[2])

    assert relative.relative_power_proxy[1] == pytest.approx(10.0 ** (-50.0 / 10.0))
    assert relative.relative_power_proxy[0] == pytest.approx(10.0 ** (-60.0 / 10.0))
    assert relative.contribution_tier[1] == 1
    assert relative.contribution_tier[2] == 1
    assert relative.contribution_tier[0] == 3
    np.testing.assert_array_equal(
        np.flatnonzero(relative.contribution_mask(50)),
        np.array([1, 2, 3, 4, 6]),
    )

    # Paths 6 and 7 also tie, independently within a second pair.
    assert relative.strength_rank[6] == 1
    assert relative.strength_rank[7] == 2


def test_pair_relative_many_single_path_groups_avoids_pair_python_loop() -> None:
    catalog = make_large_los_catalog(20_000, unique_pairs=True)
    relative = catalog.pair_relative

    # Every path has its own TX/RX pair. A Python loop over groups would make
    # this shape disproportionately expensive even at unit-test scale.
    assert relative.strength_rank.shape == (20_000,)
    assert np.all(relative.strength_rank == 1)
    assert np.all(relative.excess_delay_ns == 0.0)


def test_path_angles_preserve_zero_and_can_be_derived_from_point_data() -> None:
    catalog = make_test_catalog()
    assert catalog.aoa_azimuth_deg[0] == pytest.approx(0.0)
    assert np.isnan(catalog.aoa_azimuth_deg[1])

    canonical = catalog.canonical_data
    point_counts = np.diff(np.append(canonical.path_start_indices, len(canonical.points)))
    point_only = CanonicalStepData(
        points=canonical.points,
        lines=canonical.lines,
        order=canonical.order,
        itype=canonical.itype,
        delay=canonical.delay,
        loss=canonical.loss,
        path_id=canonical.path_id,
        path_start_indices=canonical.path_start_indices,
        path_orders=canonical.path_orders,
        path_delays=canonical.path_delays,
        path_losses=canonical.path_losses,
        path_tx=canonical.path_tx,
        path_rx=canonical.path_rx,
        aoa_az=np.repeat(canonical.path_aoa_az, point_counts),
    )
    point_only_catalog = MpcPathCatalog(point_only)
    np.testing.assert_allclose(
        point_only_catalog.aoa_azimuth_deg,
        catalog.aoa_azimuth_deg,
        equal_nan=True,
    )


def test_metric_provenance_distinguishes_exported_estimated_and_unavailable() -> None:
    catalog = make_test_catalog()

    assert catalog.delay_provenance(0) == "exported"
    assert catalog.delay_provenance(3) == "estimated"
    assert catalog.delay_provenance(2) == "unavailable"
    assert catalog.path_loss_provenance(2) == "estimated"
    assert catalog.path_loss_provenance(5) == "unavailable"
    np.testing.assert_array_equal(
        catalog.column("delay_provenance"),
        np.array([0, 0, -1, 1, 0, 1, 0, 0], dtype=np.int8),
    )
    assert catalog.column("delay") is catalog.delays_ns
    assert catalog.column("path_loss") is catalog.path_losses_db
    assert catalog.column("geometric_length") is catalog.geometric_lengths_m
    assert catalog.column("relative_path_loss") is catalog.pair_relative.path_loss_delta_db


def test_missing_metric_provenance_is_unknown_not_exported() -> None:
    canonical = make_test_catalog().canonical_data
    canonical.path_delay_is_estimated = None
    canonical.path_loss_is_estimated = None
    catalog = MpcPathCatalog(canonical)

    assert catalog.delay_provenance(0) == "unknown"
    assert catalog.path_loss_provenance(0) == "unknown"
    assert catalog.delay_provenance(2) == "unavailable"
    assert catalog.path_loss_provenance(5) == "unavailable"


def test_empty_frame_and_absent_optional_arrays_are_safe() -> None:
    canonical = CanonicalStepData(
        points=np.empty((0, 3), dtype=np.float32),
        lines=np.empty((0, 2), dtype=np.int32),
        order=np.empty((0,), dtype=np.uint8),
        itype=np.empty((0,), dtype=np.uint8),
        delay=np.empty((0,), dtype=np.float32),
        loss=np.empty((0,), dtype=np.float32),
        path_start_indices=np.empty((0,), dtype=np.int32),
    )
    catalog = MpcPathCatalog(canonical, validate=True)

    assert catalog.path_count == 0
    assert catalog.first_material_ids.size == 0
    assert catalog.aoa_azimuth_deg.size == 0
    assert catalog.scope_path_ids(MpcPathScope.RENDERED).dtype == np.int32
    assert catalog.scope_path_ids(MpcPathScope.RENDERED).size == 0
    assert catalog.geometric_lengths_m.size == 0
    assert catalog.pair_relative.strength_rank.size == 0


def test_validation_rejects_non_monotonic_path_starts() -> None:
    catalog = make_test_catalog(validate=False)
    canonical = catalog.canonical_data
    canonical.path_start_indices = np.array([0, 2, 2, 9, 12, 15, 19, 22], dtype=np.int32)

    with pytest.raises(MpcPathCatalogError, match="strictly increasing"):
        MpcPathCatalog(canonical, validate=True)
