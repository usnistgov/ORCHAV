"""Focused tests for vectorized MPC filtering and compound sorting."""

from __future__ import annotations

import numpy as np
import pytest

from tests.visualizer.unit.mpc_explorer_test_data import make_test_catalog
from visualizer.src.metrics.mpc_path_catalog import MpcPathScope
from visualizer.src.metrics.mpc_path_query import (
    DEFAULT_SORT_SPEC,
    MpcGrouping,
    MpcPathQueryEngine,
    MpcQuerySpec,
    MpcSortClause,
    MpcSortField,
    MpcSortPreset,
    MpcSortSpec,
    SortDirection,
    grouping_for_preset,
    query_spec_for_preset,
    replace_sort_primary,
    sort_spec_for_preset,
)


@pytest.mark.parametrize(
    ("preset", "expected"),
    (
        (MpcSortPreset.TX_RX_STRONGEST, [1, 2, 0, 3, 4, 5, 6, 7]),
        (MpcSortPreset.TX_RX_EARLIEST, [0, 1, 2, 3, 5, 4, 6, 7]),
        (MpcSortPreset.STRONGEST_OVERALL, [4, 1, 2, 0, 3, 6, 7, 5]),
        (MpcSortPreset.EARLIEST_OVERALL, [3, 0, 5, 4, 1, 6, 7, 2]),
        (MpcSortPreset.INTERACTIONS_STRONGEST, [0, 4, 1, 3, 6, 7, 2, 5]),
        (MpcSortPreset.INTERACTION_MIX_STRONGEST, [0, 1, 3, 2, 4, 5, 6, 7]),
        (MpcSortPreset.FIRST_MATERIAL_STRONGEST, [0, 6, 7, 1, 2, 3, 4, 5]),
        (MpcSortPreset.DELAY_BAND_STRONGEST, [3, 4, 0, 5, 1, 6, 7, 2]),
        (MpcSortPreset.LOSS_BAND_EARLIEST, [4, 1, 2, 0, 3, 6, 7, 5]),
    ),
)
def test_every_builtin_preset_has_deterministic_expected_order(
    preset: MpcSortPreset,
    expected: list[int],
) -> None:
    engine = MpcPathQueryEngine(make_test_catalog())

    result = engine.execute(query_spec_for_preset(preset), generation=12)

    assert result.generation == 12
    assert result.path_ids.dtype == np.int32
    np.testing.assert_array_equal(result.path_ids, np.asarray(expected, dtype=np.int32))


def test_preset_helpers_cover_every_declared_preset() -> None:
    for preset in MpcSortPreset:
        assert isinstance(sort_spec_for_preset(preset), MpcSortSpec)
        assert isinstance(grouping_for_preset(preset), MpcGrouping)
        assert query_spec_for_preset(preset).sort == sort_spec_for_preset(preset)


def test_nan_values_sort_last_in_both_directions() -> None:
    engine = MpcPathQueryEngine(make_test_catalog())
    descending_delay = MpcSortSpec(
        (
            MpcSortClause(MpcSortField.DELAY, SortDirection.DESCENDING),
            MpcSortClause(MpcSortField.PATH_LOSS, SortDirection.DESCENDING),
        )
    )
    descending_loss = MpcSortSpec(
        (
            MpcSortClause(MpcSortField.PATH_LOSS, SortDirection.DESCENDING),
            MpcSortClause(MpcSortField.DELAY, SortDirection.DESCENDING),
        )
    )

    delay_result = engine.execute(MpcQuerySpec(grouping=MpcGrouping.NONE, sort=descending_delay))
    loss_result = engine.execute(MpcQuerySpec(grouping=MpcGrouping.NONE, sort=descending_loss))

    assert delay_result.path_ids[-1] == 2
    assert loss_result.path_ids[-1] == 5
    # Equal finite delay/loss paths 6 and 7 use canonical path ID.
    assert list(delay_result.path_ids).index(6) < list(delay_result.path_ids).index(7)


@pytest.mark.parametrize(
    ("field", "ascending", "descending"),
    (
        (
            MpcSortField.AOD_AZIMUTH,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ),
        (
            MpcSortField.AOD_ELEVATION,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ),
        (
            MpcSortField.AOA_AZIMUTH,
            [0, 2, 3, 4, 5, 6, 7, 1],
            [7, 6, 5, 4, 3, 2, 0, 1],
        ),
        (
            MpcSortField.AOA_ELEVATION,
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ),
    ),
)
def test_angle_columns_sort_globally_in_degrees_with_unavailable_last(
    field: MpcSortField,
    ascending: list[int],
    descending: list[int],
) -> None:
    engine = MpcPathQueryEngine(make_test_catalog())

    def sorted_paths(direction: SortDirection) -> np.ndarray:
        return engine.execute(
            MpcQuerySpec(
                grouping=MpcGrouping.NONE,
                sort=MpcSortSpec(
                    (
                        MpcSortClause(field, direction),
                        MpcSortClause(MpcSortField.PATH_ID),
                    )
                ),
                include_status=False,
            )
        ).path_ids

    np.testing.assert_array_equal(
        sorted_paths(SortDirection.ASCENDING),
        np.asarray(ascending, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        sorted_paths(SortDirection.DESCENDING),
        np.asarray(descending, dtype=np.int32),
    )


def test_grouped_angle_sort_is_monotonic_within_each_pair_only() -> None:
    catalog = make_test_catalog()
    result = MpcPathQueryEngine(catalog).execute(
        MpcQuerySpec(
            grouping=MpcGrouping.TX_RX,
            sort=MpcSortSpec(
                (
                    MpcSortClause(MpcSortField.AOA_AZIMUTH),
                    MpcSortClause(MpcSortField.DELAY),
                )
            ),
            include_status=False,
        )
    )

    pair_to_angles: dict[tuple[int, int], list[float]] = {}
    for path_id in result.path_ids:
        pair = (int(catalog.tx_ids[path_id]), int(catalog.rx_ids[path_id]))
        pair_to_angles.setdefault(pair, []).append(float(catalog.aoa_azimuth_deg[path_id]))

    for values in pair_to_angles.values():
        finite = [value for value in values if np.isfinite(value)]
        assert finite == sorted(finite)
        if len(finite) != len(values):
            assert not np.isfinite(values[-1])


def test_azimuth_sort_uses_numeric_zero_to_360_seam_and_nonfinite_last() -> None:
    catalog = make_test_catalog()
    catalog.canonical_data.path_aoa_az[:] = np.array(
        [359.0, 0.0, 1.0, 180.0, np.nan, np.inf, -np.inf, 45.0],
        dtype=np.float32,
    )
    engine = MpcPathQueryEngine(catalog)

    def sorted_paths(direction: SortDirection) -> np.ndarray:
        return engine.execute(
            MpcQuerySpec(
                grouping=MpcGrouping.NONE,
                sort=MpcSortSpec(
                    (
                        MpcSortClause(MpcSortField.AOA_AZIMUTH, direction),
                        MpcSortClause(MpcSortField.PATH_ID),
                    )
                ),
                include_status=False,
            )
        ).path_ids

    np.testing.assert_array_equal(
        sorted_paths(SortDirection.ASCENDING),
        np.array([1, 2, 7, 3, 0, 4, 5, 6], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        sorted_paths(SortDirection.DESCENDING),
        np.array([0, 3, 7, 2, 1, 4, 5, 6], dtype=np.int32),
    )


def test_compound_sort_requires_two_to_four_unique_clauses() -> None:
    with pytest.raises(ValueError, match="between 2 and 4"):
        MpcSortSpec((MpcSortClause(MpcSortField.TX),))
    with pytest.raises(ValueError, match="between 2 and 4"):
        MpcSortSpec(
            (
                MpcSortClause(MpcSortField.TX),
                MpcSortClause(MpcSortField.RX),
                MpcSortClause(MpcSortField.PATH_LOSS),
                MpcSortClause(MpcSortField.DELAY),
                MpcSortClause(MpcSortField.INTERACTIONS),
            )
        )
    with pytest.raises(ValueError, match="unique"):
        MpcSortSpec(
            (
                MpcSortClause(MpcSortField.TX),
                MpcSortClause(MpcSortField.TX),
            )
        )


def test_custom_compound_sort_and_primary_replacement() -> None:
    custom = MpcSortSpec(
        (
            MpcSortClause(MpcSortField.TX),
            MpcSortClause(MpcSortField.RX),
            MpcSortClause(MpcSortField.INTERACTIONS),
            MpcSortClause(MpcSortField.DELAY),
        )
    )
    replaced = replace_sort_primary(
        custom,
        MpcSortField.DELAY,
        SortDirection.DESCENDING,
    )

    assert replaced.clauses[0] == MpcSortClause(
        MpcSortField.DELAY,
        SortDirection.DESCENDING,
    )
    assert len(replaced.clauses) == 4
    assert len({clause.field for clause in replaced.clauses}) == 4


def test_scope_and_numeric_filters_are_vectorized_together() -> None:
    engine = MpcPathQueryEngine(make_test_catalog())
    result = engine.execute(
        MpcQuerySpec(
            scope=MpcPathScope.FILTERED,
            grouping=MpcGrouping.NONE,
            sort=sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
            path_loss_min_db=45.0,
            path_loss_max_db=75.0,
            delay_min_ns=5.0,
            interaction_count_max=1,
            include_status=False,
        )
    )

    np.testing.assert_array_equal(result.path_ids, np.array([1, 0, 3], dtype=np.int32))
    assert result.scope_path_count == 6
    assert result.total_path_count == 8


def test_mechanism_filters_support_contains_pure_mixed_and_exact_sequence() -> None:
    engine = MpcPathQueryEngine(make_test_catalog())
    base = {
        "grouping": MpcGrouping.NONE,
        "sort": sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        "include_status": False,
    }

    contains = engine.execute(MpcQuerySpec(**base, contains_interactions=(1,)))
    pure = engine.execute(MpcQuerySpec(**base, pure_interaction=1))
    mixed = engine.execute(MpcQuerySpec(**base, mixed_only=True))
    exact = engine.execute(MpcQuerySpec(**base, exact_interaction_sequence=(8, 1)))

    np.testing.assert_array_equal(contains.path_ids, np.array([1, 2, 5], dtype=np.int32))
    np.testing.assert_array_equal(pure.path_ids, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(mixed.path_ids, np.array([2, 5], dtype=np.int32))
    np.testing.assert_array_equal(exact.path_ids, np.array([5], dtype=np.int32))


def test_first_material_filter_uses_numeric_ids_without_path_strings() -> None:
    catalog = make_test_catalog(validate=False)
    engine = MpcPathQueryEngine(catalog)

    result = engine.execute(
        MpcQuerySpec(
            grouping=MpcGrouping.NONE,
            sort=DEFAULT_SORT_SPEC,
            first_material_ids=(1,),
            include_status=False,
        )
    )

    np.testing.assert_array_equal(result.path_ids, np.array([1, 2], dtype=np.int32))
    assert "first_material_ids" in catalog.cached_derived_columns


def test_grouping_keys_are_primary_to_custom_sort_clauses() -> None:
    engine = MpcPathQueryEngine(make_test_catalog())
    result = engine.execute(
        MpcQuerySpec(
            grouping=MpcGrouping.TX_RX,
            sort=sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        )
    )
    catalog = engine.catalog
    pairs = [
        (int(catalog.tx_ids[path_id]), int(catalog.rx_ids[path_id])) for path_id in result.path_ids
    ]

    assert pairs == sorted(pairs)


def test_grouping_preserves_explicit_sort_directions() -> None:
    catalog = make_test_catalog()
    descending_groups = MpcSortSpec(
        (
            MpcSortClause(MpcSortField.PATH_LOSS),
            MpcSortClause(MpcSortField.TX, SortDirection.DESCENDING),
            MpcSortClause(MpcSortField.RX, SortDirection.DESCENDING),
        )
    )

    result = MpcPathQueryEngine(catalog).execute(
        MpcQuerySpec(
            grouping=MpcGrouping.TX_RX,
            sort=descending_groups,
            include_status=False,
        )
    )
    pairs = [
        (int(catalog.tx_ids[path_id]), int(catalog.rx_ids[path_id])) for path_id in result.path_ids
    ]

    assert pairs == sorted(pairs, reverse=True)


def test_status_warmup_is_worker_controlled() -> None:
    cold_catalog = make_test_catalog(validate=False)
    cold_engine = MpcPathQueryEngine(cold_catalog)
    cold_engine.execute(
        MpcQuerySpec(
            grouping=MpcGrouping.NONE,
            sort=sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
            include_status=False,
        )
    )
    assert "scope_mask:rendered" not in cold_catalog.cached_derived_columns

    warm_catalog = make_test_catalog(validate=False)
    MpcPathQueryEngine(warm_catalog).execute(
        MpcQuerySpec(
            grouping=MpcGrouping.NONE,
            sort=sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
            include_status=True,
        )
    )
    assert "scope_mask:rendered" in warm_catalog.cached_derived_columns


def test_worker_prewarms_requested_columns_and_pick_mapping() -> None:
    catalog = make_test_catalog(validate=False)
    spec = MpcQuerySpec(
        grouping=MpcGrouping.NONE,
        sort=sort_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        include_status=False,
        prewarm_columns=(
            " Geometric_Length ",
            "geometric_length",
            "relative_power",
        ),
        include_pick_mapping=True,
    )

    assert spec.prewarm_columns == ("geometric_length", "relative_power")
    assert not catalog.derived_column_is_ready("geometric_length")
    assert not catalog.derived_column_is_ready("relative_power")
    assert "rendered_segment_indices" not in catalog.cached_derived_columns

    MpcPathQueryEngine(catalog).execute(spec)

    assert catalog.derived_column_is_ready("geometric_length")
    assert catalog.derived_column_is_ready("relative_power")
    assert "rendered_segment_indices" in catalog.cached_derived_columns
    mapping = catalog.rendered_segment_indices
    assert mapping is not None
    assert mapping.dtype == np.int32


def test_query_spec_rejects_conflicting_or_invalid_filters() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MpcQuerySpec(pure_interaction=1, mixed_only=True)
    with pytest.raises(ValueError, match="minimum"):
        MpcQuerySpec(delay_min_ns=2.0, delay_max_ns=1.0)
    with pytest.raises(ValueError, match="must be finite"):
        MpcQuerySpec(path_loss_min_db=np.nan)
    with pytest.raises(ValueError, match="must be finite"):
        MpcQuerySpec(delay_max_ns=np.inf)
    with pytest.raises(ValueError, match="positive"):
        MpcQuerySpec(path_loss_band_width_db=0.0)
