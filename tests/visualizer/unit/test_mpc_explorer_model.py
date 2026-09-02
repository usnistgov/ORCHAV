"""Qt model tests for MPC Explorer batching and stale-result rejection."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QAbstractItemModelTester

from tests.visualizer.unit.mpc_explorer_test_data import (
    make_large_los_catalog,
    make_test_catalog,
)
from visualizer.src.metrics.mpc_path_query import (
    MpcGrouping,
    MpcPathQueryEngine,
    MpcQueryResult,
    MpcQuerySpec,
    MpcSortField,
    MpcSortPreset,
    SortDirection,
    query_spec_for_preset,
)
from visualizer.src.model.mpc_explorer_model import (
    DEFAULT_COLUMNS,
    MpcExplorerColumn,
    MpcExplorerTableModel,
)


def _cache_row(model: MpcExplorerTableModel, path_id: int) -> int | None:
    """Emulate the session's worker result for focused model tests."""
    revision, path_ids = model.path_lookup_snapshot()
    matches = np.flatnonzero(path_ids == int(path_id))
    row = int(matches[0]) if matches.size else None
    assert model.cache_path_row(
        path_id,
        row,
        permutation_revision=revision,
    )
    return row


def test_model_passes_qt_tester_and_formats_default_columns(qapp) -> None:
    catalog = make_test_catalog()
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=4)
    result = MpcPathQueryEngine(catalog).execute(
        query_spec_for_preset(MpcSortPreset.TX_RX_STRONGEST),
        generation=4,
    )
    assert model.apply_query_result(result)
    tester = QAbstractItemModelTester(
        model,
        QAbstractItemModelTester.FailureReportingMode.Warning,
    )

    assert tester.model() is model
    assert model.rowCount() == 8
    assert model.columnCount() == len(DEFAULT_COLUMNS)
    first_path_id = model.path_id_for_row(0)
    assert first_path_id == 1
    assert model.data(model.index(0, 0), model.PathIdRole) == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "1"
    assert model.headerData(3, Qt.Orientation.Horizontal) == "Path Loss (dB)"
    assert model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole) == "50.00"
    assert model.data(model.index(0, 8), Qt.ItemDataRole.DisplayRole) == (
        "Filtered - outside rendered set"
    )


def test_angle_headers_make_world_coordinate_convention_explicit(qapp) -> None:
    model = MpcExplorerTableModel(
        columns=(
            MpcExplorerColumn.AOD_AZIMUTH,
            MpcExplorerColumn.AOA_ELEVATION,
        ),
        fetch_batch_size=25_000,
    )

    assert model.headerData(0, Qt.Orientation.Horizontal) == ("AoD Azimuth (world deg)")
    tooltip = model.headerData(
        0,
        Qt.Orientation.Horizontal,
        Qt.ItemDataRole.ToolTipRole,
    )
    assert "[0, 360)" in tooltip
    assert "+X toward +Y" in tooltip


def test_model_fetches_large_results_in_fixed_batches(qapp) -> None:
    catalog = make_large_los_catalog(60_001)
    spec = MpcQuerySpec(grouping=MpcGrouping.NONE, include_status=False)
    result = MpcQueryResult(
        generation=10,
        path_ids=np.arange(60_001, dtype=np.int32),
        total_path_count=60_001,
        scope_path_count=60_001,
        elapsed_ms=0.0,
        spec=spec,
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=10)
    assert model.apply_query_result(result)

    assert model.rowCount() == 25_000
    assert model.total_row_count == 60_001
    assert model.canFetchMore()
    model.fetchMore()
    assert model.rowCount() == 50_000
    model.fetchMore()
    assert model.rowCount() == 60_001
    assert not model.canFetchMore()


def test_model_rejects_stale_worker_result_without_resetting_current_rows(qapp) -> None:
    catalog = make_test_catalog()
    engine = MpcPathQueryEngine(catalog)
    current = engine.execute(
        query_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        generation=8,
    )
    stale = engine.execute(
        query_spec_for_preset(MpcSortPreset.EARLIEST_OVERALL),
        generation=7,
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=8)
    assert model.apply_query_result(current)
    current_ids = [model.path_id_for_row(row) for row in range(model.total_row_count)]

    assert not model.apply_query_result(stale)
    assert [model.path_id_for_row(row) for row in range(model.total_row_count)] == current_ids


def test_model_restores_path_identity_after_sort_and_filter(qapp) -> None:
    catalog = make_test_catalog()
    engine = MpcPathQueryEngine(catalog)
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=3)
    strongest = engine.execute(
        query_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        generation=3,
    )
    assert model.apply_query_result(strongest)
    original_row = _cache_row(model, 0)
    assert original_row is not None

    earliest = engine.execute(
        query_spec_for_preset(MpcSortPreset.EARLIEST_OVERALL),
        generation=3,
    )
    assert model.apply_query_result(earliest)
    restored_row = _cache_row(model, 0)

    assert restored_row is not None
    assert restored_row != original_row
    assert model.path_id_for_row(restored_row) == 0


def test_ensure_row_loaded_advances_only_one_batch_per_call(qapp) -> None:
    catalog = make_large_los_catalog(60_001)
    result = MpcQueryResult(
        generation=2,
        path_ids=np.arange(60_001, dtype=np.int32),
        total_path_count=60_001,
        scope_path_count=60_001,
        elapsed_ms=0.0,
        spec=MpcQuerySpec(grouping=MpcGrouping.NONE, include_status=False),
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=2)
    model.apply_query_result(result)
    insertions = []
    model.rowsInserted.connect(
        lambda _parent, first, last: insertions.append((int(first), int(last)))
    )

    assert _cache_row(model, 55_000) == 55_000
    assert not model.ensure_row_loaded(55_000)
    assert model.loaded_row_count == 50_000
    assert model.ensure_row_loaded(55_000)
    assert model.loaded_row_count == 60_001
    assert insertions == [(25_000, 49_999), (50_000, 60_000)]
    assert all(last - first + 1 <= model.fetch_batch_size for first, last in insertions)


def test_cold_path_lookup_never_scans_on_gui_thread_and_worker_rows_are_cached(
    qapp,
    monkeypatch,
) -> None:
    catalog = make_large_los_catalog(60_001)
    result = MpcQueryResult(
        generation=4,
        path_ids=np.arange(60_001, dtype=np.int32),
        total_path_count=60_001,
        scope_path_count=60_001,
        elapsed_ms=0.0,
        spec=MpcQuerySpec(grouping=MpcGrouping.NONE, include_status=False),
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=4)
    model.apply_query_result(result)

    monkeypatch.setattr(
        np,
        "flatnonzero",
        lambda _values: (_ for _ in ()).throw(
            AssertionError("GUI-thread row lookup scanned the permutation")
        ),
    )

    assert model.row_for_path_id(60_000) is None
    revision, path_ids = model.path_lookup_snapshot()
    assert path_ids is result.path_ids
    assert model.cache_path_row(60_000, 60_000, permutation_revision=revision)
    assert model.row_for_path_id(60_000) == 60_000

    assert model.row_for_path_id(70_000) is None
    assert model.cache_path_row(70_000, None, permutation_revision=revision)
    assert model.row_for_path_id(70_000) is None

    for path_id in range(model.PATH_ROW_CACHE_SIZE + 5):
        assert model.cache_path_row(path_id, path_id, permutation_revision=revision)
        assert model.row_for_path_id(path_id) == path_id
    assert len(model._path_row_cache) == model.PATH_ROW_CACHE_SIZE


def test_model_rejects_stale_or_mismatched_worker_row_results(qapp) -> None:
    catalog = make_large_los_catalog(60_001)
    first = MpcQueryResult(
        generation=1,
        path_ids=np.arange(60_001, dtype=np.int32),
        total_path_count=60_001,
        scope_path_count=60_001,
        elapsed_ms=0.0,
        spec=MpcQuerySpec(grouping=MpcGrouping.NONE, include_status=False),
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=1)
    assert model.apply_query_result(first)
    old_revision, _old_ids = model.path_lookup_snapshot()

    model.begin_generation(catalog, generation=2, clear_rows=False)
    reversed_result = MpcQueryResult(
        generation=2,
        path_ids=np.arange(60_000, -1, -1, dtype=np.int32),
        total_path_count=60_001,
        scope_path_count=60_001,
        elapsed_ms=0.0,
        spec=MpcQuerySpec(grouping=MpcGrouping.NONE, include_status=False),
    )
    assert model.apply_query_result(reversed_result)
    new_revision, _new_ids = model.path_lookup_snapshot()

    assert new_revision != old_revision
    assert not model.cache_path_row(5, 5, permutation_revision=old_revision)
    assert not model.cache_path_row(5, 5, permutation_revision=new_revision)
    assert model.cache_path_row(5, 59_995, permutation_revision=new_revision)
    assert model.row_for_path_id(5) == 59_995


def test_begin_generation_can_retain_last_good_rows(qapp) -> None:
    catalog = make_test_catalog()
    engine = MpcPathQueryEngine(catalog)
    strongest = engine.execute(
        query_spec_for_preset(MpcSortPreset.STRONGEST_OVERALL),
        generation=1,
    )
    earliest = engine.execute(
        query_spec_for_preset(MpcSortPreset.EARLIEST_OVERALL),
        generation=2,
    )
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=1)
    assert model.apply_query_result(strongest)
    retained_ids = [model.path_id_for_row(row) for row in range(model.total_row_count)]

    model.begin_generation(catalog, generation=2, clear_rows=False)

    assert model.generation == 2
    assert model.rowCount() == len(retained_ids)
    assert [model.path_id_for_row(row) for row in range(model.total_row_count)] == retained_ids
    assert not model.apply_query_result(strongest)
    assert model.apply_query_result(earliest)
    assert [model.path_id_for_row(row) for row in range(model.total_row_count)] != retained_ids


def test_sort_emits_compound_request_without_reordering_on_gui_thread(qapp) -> None:
    catalog = make_test_catalog()
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=1)
    result = MpcPathQueryEngine(catalog).execute(
        query_spec_for_preset(MpcSortPreset.TX_RX_STRONGEST),
        generation=1,
    )
    model.apply_query_result(result)
    before = result.path_ids.copy()
    requests = []
    model.sortRequested.connect(requests.append)

    loss_column = model.columns.index(MpcExplorerColumn.PATH_LOSS)
    model.sort(loss_column, Qt.SortOrder.DescendingOrder)

    assert len(requests) == 1
    assert requests[0].clauses[0].field is MpcSortField.PATH_LOSS
    assert requests[0].clauses[0].direction is SortDirection.DESCENDING
    np.testing.assert_array_equal(result.path_ids, before)


def test_optional_values_and_lazy_strings_use_selected_path_only(qapp) -> None:
    catalog = make_test_catalog(validate=False)
    columns = (
        MpcExplorerColumn.PATH_ID,
        MpcExplorerColumn.AOA_AZIMUTH,
        MpcExplorerColumn.INTERACTION_SEQUENCE,
        MpcExplorerColumn.MATERIAL_SEQUENCE,
        MpcExplorerColumn.DELAY_PROVENANCE,
        MpcExplorerColumn.PATH_LOSS_PROVENANCE,
    )
    model = MpcExplorerTableModel(
        columns=columns,
        fetch_batch_size=25_000,
    )
    model.begin_generation(catalog, generation=6)
    result = MpcPathQueryEngine(catalog).execute(
        query_spec_for_preset(
            MpcSortPreset.STRONGEST_OVERALL,
            include_status=False,
        ),
        generation=6,
    )
    model.apply_query_result(result)
    row = _cache_row(model, 2)
    assert row is not None

    assert model.data(model.index(row, 1), Qt.ItemDataRole.DisplayRole) == "20.00"
    assert model.data(model.index(row, 2), Qt.ItemDataRole.DisplayRole) == ("Spec -> Diffuse")
    assert model.data(model.index(row, 3), Qt.ItemDataRole.DisplayRole) == ("Concrete -> Glass")
    assert model.data(model.index(row, 4), Qt.ItemDataRole.DisplayRole) == "unavailable"
    assert model.data(model.index(row, 5), Qt.ItemDataRole.DisplayRole) == "estimated"
    assert "first_material_ids" not in catalog.cached_derived_columns


def test_deferred_expensive_columns_never_build_on_gui_data_access(qapp) -> None:
    catalog = make_test_catalog(validate=False)
    model = MpcExplorerTableModel(
        columns=(
            MpcExplorerColumn.GEOMETRIC_LENGTH,
            MpcExplorerColumn.RELATIVE_POWER,
        ),
        fetch_batch_size=25_000,
        defer_expensive_columns=True,
    )
    model.begin_generation(catalog, generation=9)
    cold_result = MpcPathQueryEngine(catalog).execute(
        query_spec_for_preset(
            MpcSortPreset.STRONGEST_OVERALL,
            include_status=False,
        ),
        generation=9,
    )
    assert model.apply_query_result(cold_result)
    length_index = model.index(0, 0)
    power_index = model.index(0, 1)

    assert model.data(length_index, Qt.ItemDataRole.TextAlignmentRole) is not None
    assert model.data(power_index, Qt.ItemDataRole.ToolTipRole).startswith("Dimensionless")
    assert model.data(length_index, Qt.ItemDataRole.DisplayRole) == "--"
    assert model.data(length_index, Qt.ItemDataRole.EditRole) is None
    assert model.data(length_index, model.RawValueRole) is None
    assert not catalog.derived_column_is_ready("geometric_length")
    assert not catalog.derived_column_is_ready("relative_power")

    warm_result = MpcPathQueryEngine(catalog).execute(
        query_spec_for_preset(
            MpcSortPreset.STRONGEST_OVERALL,
            include_status=False,
            prewarm_columns=("geometric_length", "relative_power"),
        ),
        generation=9,
    )
    assert model.apply_query_result(warm_result)

    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) != "--"
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) != "--"


def test_group_boundary_role_marks_only_changes_in_flat_order(qapp) -> None:
    catalog = make_test_catalog()
    model = MpcExplorerTableModel(fetch_batch_size=25_000)
    model.begin_generation(catalog, generation=5)
    model.apply_query_result(
        MpcPathQueryEngine(catalog).execute(
            query_spec_for_preset(MpcSortPreset.TX_RX_STRONGEST),
            generation=5,
        )
    )

    boundaries = [
        row
        for row in range(model.rowCount())
        if model.data(model.index(row, 0), model.GroupBoundaryRole)
    ]
    assert boundaries == [3, 4, 6]
