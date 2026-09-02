"""Focused Qt and lifecycle tests for the lazy MPC Explorer session."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QToolButton

from visualizer.src.panels.mpc_explorer_window import MpcExplorerWindow
from visualizer.src.services.mpc_explorer_service import (
    MpcExplorerSession,
    MpcPresentedFrameToken,
    toggle_mpc_explorer,
)

_APP = QApplication.instance() or QApplication([])


class _Scope(str, Enum):
    ALL = "all"
    FILTERED = "filtered"
    RENDERED = "rendered"


class _Grouping(str, Enum):
    NONE = "none"
    TX_RX = "tx_rx"
    RX_TX = "rx_tx"
    INTERACTIONS = "interactions"
    INTERACTION_MIX = "interaction_mix"
    FIRST_MATERIAL = "first_material"
    DELAY_BAND = "delay_band"
    PATH_LOSS_BAND = "path_loss_band"


class _Preset(str, Enum):
    TX_RX_STRONGEST = "tx_rx_strongest"
    TX_RX_EARLIEST = "tx_rx_earliest"
    STRONGEST_OVERALL = "strongest_overall"
    EARLIEST_OVERALL = "earliest_overall"
    INTERACTIONS_STRONGEST = "interactions_strongest"
    INTERACTION_MIX_STRONGEST = "interaction_mix_strongest"
    FIRST_MATERIAL_STRONGEST = "first_material_strongest"
    DELAY_BAND_STRONGEST = "delay_band_strongest"
    LOSS_BAND_EARLIEST = "loss_band_earliest"


class _SortField(str, Enum):
    PATH_ID = "path_id"
    TX = "tx"
    RX = "rx"
    PATH_LOSS = "path_loss"
    DELAY = "delay"
    INTERACTIONS = "interactions"
    INTERACTION_MIX = "interaction_mix"
    FIRST_MATERIAL = "first_material"
    DELAY_BAND = "delay_band"
    PATH_LOSS_BAND = "path_loss_band"
    GEOMETRIC_LENGTH = "geometric_length"
    STRETCH_RATIO = "stretch_ratio"
    EXCESS_DELAY = "excess_delay"
    STRENGTH_RANK = "strength_rank"
    RELATIVE_PATH_LOSS = "relative_path_loss"
    RELATIVE_POWER = "relative_power"
    AOD_AZIMUTH = "aod_azimuth"
    AOD_ELEVATION = "aod_elevation"
    AOA_AZIMUTH = "aoa_azimuth"
    AOA_ELEVATION = "aoa_elevation"


class _Direction(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class _SortClause:
    field: _SortField
    direction: _Direction


@dataclass(frozen=True)
class _SortSpec:
    clauses: tuple[_SortClause, ...]


@dataclass(frozen=True)
class _QuerySpec:
    scope: _Scope
    grouping: _Grouping
    sort: _SortSpec
    tx_ids: tuple[int, ...] = ()
    rx_ids: tuple[int, ...] = ()
    path_loss_min_db: float | None = None
    path_loss_max_db: float | None = None
    delay_min_ns: float | None = None
    delay_max_ns: float | None = None
    interaction_count_min: int | None = None
    interaction_count_max: int | None = None
    contains_interactions: tuple[int, ...] = ()
    pure_interaction: int | None = None
    mixed_only: bool = False
    exact_interaction_sequence: tuple[int, ...] | None = None
    first_material_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.path_loss_min_db,
            self.path_loss_max_db,
            self.delay_min_ns,
            self.delay_max_ns,
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError("query bounds must be finite")


def _sort_spec_for_preset(_preset: _Preset) -> _SortSpec:
    return _SortSpec(
        (
            _SortClause(_SortField.TX, _Direction.ASCENDING),
            _SortClause(_SortField.RX, _Direction.ASCENDING),
            _SortClause(_SortField.PATH_LOSS, _Direction.ASCENDING),
            _SortClause(_SortField.PATH_ID, _Direction.ASCENDING),
        )
    )


def _grouping_for_preset(preset: _Preset) -> _Grouping:
    if preset in {_Preset.STRONGEST_OVERALL, _Preset.EARLIEST_OVERALL}:
        return _Grouping.NONE
    return _Grouping.TX_RX


QUERY_TYPES = {
    "scope": _Scope,
    "grouping": _Grouping,
    "query_spec": _QuerySpec,
    "sort_clause": _SortClause,
    "sort_field": _SortField,
    "sort_preset": _Preset,
    "sort_spec": _SortSpec,
    "sort_direction": _Direction,
    "grouping_for_preset": _grouping_for_preset,
    "sort_spec_for_preset": _sort_spec_for_preset,
}


class _Catalog:
    created = 0

    def __init__(
        self,
        canonical_data: Any,
        *,
        filtered_path_mask: Any = None,
        rendered_segment_mask: Any = None,
        validate: bool = False,
    ) -> None:
        type(self).created += 1
        self.canonical_data = canonical_data
        self.path_count = int(canonical_data.path_count)
        self.filtered_path_mask = filtered_path_mask
        self.rendered_segment_mask = rendered_segment_mask
        self.validate_requested = validate

    def column(self, name: str) -> np.ndarray:
        columns = {
            "path_id": np.arange(self.path_count, dtype=np.int32),
            "tx": np.zeros(self.path_count, dtype=np.int32),
            "rx": np.ones(self.path_count, dtype=np.int32),
            "path_loss_db": np.linspace(60.0, 80.0, self.path_count),
            "delay_ns": np.linspace(10.0, 30.0, self.path_count),
            "interactions": np.arange(self.path_count, dtype=np.int32),
            "geometric_length_m": np.linspace(3.0, 9.0, self.path_count),
        }
        return columns[name]

    def scope_mask(self, scope: _Scope) -> np.ndarray:
        if scope is _Scope.FILTERED:
            return np.asarray(self.filtered_path_mask, dtype=bool)
        if scope is _Scope.RENDERED:
            return np.ones(self.path_count, dtype=bool)
        return np.ones(self.path_count, dtype=bool)

    def interaction_sequence(self, path_id: int) -> tuple[int, ...]:
        return (1,) if path_id else ()

    def material_sequence(self, path_id: int) -> tuple[str, ...]:
        return ("concrete",) if path_id else ()


class _QueryEngine:
    executed_specs: list[_QuerySpec] = []

    def __init__(self, catalog: _Catalog) -> None:
        self.catalog = catalog

    def execute(self, spec: _QuerySpec, *, generation: int) -> Any:
        type(self).executed_specs.append(spec)
        mask = self.catalog.scope_mask(spec.scope)
        return SimpleNamespace(
            generation=generation,
            path_ids=np.flatnonzero(mask).astype(np.int32, copy=False),
        )


class _BlockingQueryEngine(_QueryEngine):
    started = Event()
    release = Event()
    finished = Event()

    def execute(self, spec: _QuerySpec, *, generation: int) -> Any:
        type(self).started.set()
        type(self).release.wait(timeout=2.0)
        result = super().execute(spec, generation=generation)
        type(self).finished.set()
        return result


class _Model(QAbstractTableModel):
    sortRequested = Signal(object)
    GroupBoundaryRole = int(Qt.UserRole) + 3
    created = 0

    def __init__(
        self,
        *,
        fetch_batch_size: int = 50_000,
        columns: tuple[Any, ...] = (),
    ) -> None:
        super().__init__()
        type(self).created += 1
        self.fetch_batch_size = fetch_batch_size
        self.columns = tuple(columns)
        self.generation = 0
        self.path_ids = np.empty((0,), dtype=np.int32)
        self.total_row_count = 0
        self.shutdown_calls = 0

    def begin_generation(
        self,
        _catalog: Any,
        *,
        generation: int,
        clear_rows: bool = True,
    ) -> None:
        self.generation = generation
        if not clear_rows:
            return
        self.beginResetModel()
        self.path_ids = np.empty((0,), dtype=np.int32)
        self.total_row_count = 0
        self.endResetModel()

    def apply_query_result(self, result: Any) -> bool:
        if result.generation != self.generation:
            return False
        self.beginResetModel()
        self.path_ids = np.asarray(result.path_ids, dtype=np.int32)
        self.total_row_count = int(self.path_ids.size)
        self.endResetModel()
        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(self.path_ids.size)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else 2

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.UserRole):
            return None
        path_id = int(self.path_ids[index.row()])
        return path_id if index.column() == 0 else f"Path {path_id}"

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ("Path ID", "Description")[section]
        return None

    def path_id_for_row(self, row: int) -> int:
        return int(self.path_ids[row])

    def row_for_path_id(self, path_id: int) -> int | None:
        matches = np.flatnonzero(self.path_ids == int(path_id))
        return int(matches[0]) if matches.size else None

    def ensure_row_loaded(self, row: int) -> bool:
        return 0 <= int(row) < self.total_row_count

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _Pipeline:
    def __init__(self) -> None:
        self.callback = None
        self.update_calls = 0

    def set_mpc_explorer_presented_callback(self, callback) -> None:
        assert self.callback is None
        self.callback = callback

    def clear_mpc_explorer_presented_callback(self, callback=None) -> None:
        if callback is None or self.callback is callback:
            self.callback = None

    def update(self, _step: int) -> bool:
        self.update_calls += 1
        return True


class _Renderer:
    def __init__(self) -> None:
        self.selection_callback = None
        self.clear_calls = 0

    def set_mpc_path_selection_callback(self, callback) -> None:
        self.selection_callback = callback

    def clear_mpc_path_inspection(self) -> bool:
        self.clear_calls += 1
        return True


class _SelectionPort(QObject):
    selectionChanged = Signal(object)
    detailsChanged = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[tuple[MpcPresentedFrameToken, bool]] = []
        self.selections: list[tuple[MpcPresentedFrameToken, int, str]] = []
        self.packet_identities: list[int | None] = []
        self.clear_reasons: list[str] = []

    def set_presented_frame(
        self,
        token: MpcPresentedFrameToken,
        _catalog: Any,
        _packet: Any,
        *,
        frame_changed: bool,
    ) -> None:
        self.frames.append((token, frame_changed))

    def select_path(
        self,
        token: MpcPresentedFrameToken,
        path_id: int,
        *,
        origin: str,
        packet_identity: int | None = None,
    ) -> None:
        self.selections.append((token, path_id, origin))
        self.packet_identities.append(packet_identity)
        self.selectionChanged.emit(
            SimpleNamespace(
                frame_token=token,
                canonical_path_id=path_id,
            )
        )

    def clear_presented_frame(self, *, reason: str) -> None:
        self.clear_reasons.append(reason)


class _Visualizer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = _Pipeline()
        self.renderer = _Renderer()
        self.animation_step = 4
        self.force_update_next_frame = False
        self._shutdown_started = False
        self._mpc_explorer_session = None


def _wait_for(predicate, timeout: float = 2.0, state=None) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        _APP.processEvents()
        time.sleep(0.005)
    assert predicate(), state() if state is not None else "condition did not become true"


def _make_session(
    viz: _Visualizer,
    port: _SelectionPort | None = None,
    *,
    query_engine_type: type[_QueryEngine] = _QueryEngine,
) -> MpcExplorerSession:
    return MpcExplorerSession(
        viz,
        catalog_factory=_Catalog,
        model_factory=_Model,
        query_engine_type=query_engine_type,
        query_types=dict(QUERY_TYPES),
        selection_port=port,
    )


def _present(
    viz: _Visualizer,
    canonical: Any,
    *,
    epoch: int = 2,
    step: int = 4,
) -> tuple[Any, Any]:
    assert viz.pipeline.callback is not None
    view_model = SimpleNamespace(
        canonical_data=canonical,
        path_mask=np.array([True, False, True], dtype=bool),
    )
    packet = SimpleNamespace(
        canonical_data=canonical,
        segment_mask=np.array([True, True, False], dtype=bool),
    )
    viz.pipeline.callback(epoch, step, view_model, packet)
    return view_model, packet


def test_window_shell_has_no_model_until_a_session_feeds_it() -> None:
    window = MpcExplorerWindow()
    try:
        assert window.table.model() is None
        assert window.current_scope() == "all"
        assert window.current_grouping() == "tx_rx"
        assert not window.apply_custom_sort_button.isEnabled()
        assert window.count_label.text() == "0 / 0 paths"
        chip_texts = [
            widget.text()
            for index in range(window.sort_chip_layout.count())
            if isinstance(
                (widget := window.sort_chip_layout.itemAt(index).widget()),
                QToolButton,
            )
        ]
        assert chip_texts.count("Path ID ASC (tie)") == 1
    finally:
        window.close()
        _APP.processEvents()


def test_window_emits_unbounded_compound_filters() -> None:
    window = MpcExplorerWindow()
    emitted: list[dict[str, Any]] = []
    window.filtersChanged.connect(emitted.append)
    try:
        window.filter_edits["tx_ids"].setText("0, 2")
        window.filter_edits["path_loss_min_db"].setText("-125.5")
        window.filter_edits["delay_max_ns"].setText("1000000000")
        window.filter_edits["contains_interactions"].setText("1, 3")
        window.filter_edits["exact_interaction_sequence"].setText("1, 2, 1")
        window.filter_edits["first_material_ids"].setText("4, 9")
        window.mixed_only_checkbox.setChecked(True)

        window._apply_filters()

        assert emitted[-1] == {
            "tx_ids": (0, 2),
            "rx_ids": (),
            "path_loss_min_db": -125.5,
            "path_loss_max_db": None,
            "delay_min_ns": None,
            "delay_max_ns": 1_000_000_000.0,
            "interaction_count_min": None,
            "interaction_count_max": None,
            "contains_interactions": (1, 3),
            "pure_interaction": None,
            "mixed_only": True,
            "exact_interaction_sequence": (1, 2, 1),
            "first_material_ids": (4, 9),
        }
        assert window.filter_button.text() == "Filters (7)"
    finally:
        window.close()
        _APP.processEvents()


def test_window_reports_optional_column_visibility_for_worker_prewarm() -> None:
    window = MpcExplorerWindow()
    model = _Model(columns=("path_id", "geometric_length"))
    emitted: list[tuple[str, bool]] = []
    window.columnVisibilityChanged.connect(
        lambda column, visible: emitted.append((column, visible))
    )
    try:
        window.set_model(model, optional_start_column=1)
        assert window.table.isColumnHidden(1)
        actions = window._column_menu.actions()
        actions[1].setChecked(True)

        assert not window.table.isColumnHidden(1)
        assert emitted[-1] == ("geometric_length", True)
    finally:
        window.close()
        _APP.processEvents()


def test_window_separates_complete_presets_from_custom_ordering() -> None:
    window = MpcExplorerWindow()
    preset_events: list[str] = []
    grouping_events: list[str] = []
    clause_events: list[tuple[tuple[str, bool], ...]] = []
    window.sortPresetChanged.connect(preset_events.append)
    window.groupingChanged.connect(grouping_events.append)
    window.sortClausesChanged.connect(clause_events.append)
    try:
        assert window.ordering_tabs.tabText(0) == "Preset recipes"
        assert window.ordering_tabs.tabText(1) == "Custom order"
        assert window.current_ordering_mode() == "preset"
        assert "TX/RX pair" in window.preset_description_label.text()

        delay_band_index = window.preset_combo.findData("delay_band_strongest")
        window.preset_combo.setCurrentIndex(delay_band_index)
        assert preset_events[-1] == "delay_band_strongest"
        assert "10 ns delay bands" in window.preset_description_label.text()

        window.ordering_tabs.setCurrentIndex(1)
        assert window.current_ordering_mode() == "custom"
        assert grouping_events[-1] == "delay_band"
        assert clause_events[-1] == (("path_loss", True), ("delay", True))
        assert window.sort_chip_caption.text() == "Custom effective order:"

        emitted_group_count = len(grouping_events)
        none_index = window.grouping_combo.findData("none")
        window.grouping_combo.setCurrentIndex(none_index)
        assert len(grouping_events) == emitted_group_count
        assert window.sort_chip_caption.text() == "Custom draft — Apply to use:"

        aod_index = window.sort_column_combo.findData("aod_azimuth")
        window.sort_column_combo.setCurrentIndex(aod_index)
        window._set_custom_sort_primary()
        assert window.apply_custom_sort_button.isEnabled()
        window._apply_custom_sort()

        assert grouping_events[-1] == "none"
        assert clause_events[-1][0] == ("aod_azimuth", True)
        assert window.sort_chip_caption.text() == "Custom effective order:"
        global_texts = [
            widget.text()
            for index in range(window.sort_chip_layout.count())
            if isinstance(
                (widget := window.sort_chip_layout.itemAt(index).widget()),
                QToolButton,
            )
        ]
        assert global_texts[0] == "AoD azimuth (world deg) ASC  x"

        window.ordering_tabs.setCurrentIndex(0)
        assert preset_events[-1] == "delay_band_strongest"
        assert window.sort_chip_caption.text() == "Preset effective order:"
    finally:
        window.close()
        _APP.processEvents()


def test_table_header_sort_becomes_an_explicit_global_custom_order() -> None:
    window = MpcExplorerWindow()
    try:
        sort_spec = SimpleNamespace(
            clauses=(
                SimpleNamespace(
                    field=SimpleNamespace(value="aod_azimuth"),
                    direction=SimpleNamespace(value="ascending"),
                ),
                SimpleNamespace(
                    field=SimpleNamespace(value="path_loss"),
                    direction=SimpleNamespace(value="ascending"),
                ),
            )
        )

        window.set_sort_spec(sort_spec, global_order=True)

        assert window.current_ordering_mode() == "custom"
        assert window.current_grouping() == "none"
        assert window.sort_chip_caption.text() == "Custom effective order:"
        chip_texts = [
            widget.text()
            for index in range(window.sort_chip_layout.count())
            if isinstance(
                (widget := window.sort_chip_layout.itemAt(index).widget()),
                QToolButton,
            )
        ]
        assert chip_texts[:2] == [
            "AoD azimuth (world deg) ASC",
            "Path loss ASC",
        ]
    finally:
        window.close()
        _APP.processEvents()


def test_toggle_lazily_constructs_an_owned_selection_service() -> None:
    viz = _Visualizer()
    session = toggle_mpc_explorer(viz)
    _APP.processEvents()
    try:
        assert session is viz._mpc_explorer_session
        assert session._selection_port.__class__.__name__ == "MpcSelectionService"
        assert session._owns_selection_port is True
        assert viz.pipeline.callback is not None
        assert viz.renderer.selection_callback is None
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_open_session_defers_catalog_model_and_worker_until_presented() -> None:
    _Catalog.created = 0
    _Model.created = 0
    viz = _Visualizer()
    session = _make_session(viz)
    viz._mpc_explorer_session = session
    session.open()
    _APP.processEvents()

    try:
        assert viz.pipeline.callback is not None
        assert _Catalog.created == 0
        assert _Model.created == 0
        assert session._executor is None

        canonical = SimpleNamespace(path_count=3)
        _present(viz, canonical)
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)

        assert _Catalog.created == 1
        assert _Model.created == 1
        assert session.frame_token == MpcPresentedFrameToken(2, 4, id(canonical))
        assert session.window.count_label.text() == "3 / 3 paths"
        assert viz.renderer.selection_callback is not None

        session.window.hide()
        _APP.processEvents()
        assert viz.pipeline.callback is None
        assert viz.renderer.selection_callback is None
        assert session._catalog is None
        assert session._model is None
        assert session._executor is None
        assert session.window.table.model() is None
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_filter_values_reach_the_shared_query_spec() -> None:
    _QueryEngine.executed_specs.clear()
    viz = _Visualizer()
    session = _make_session(viz)
    session.open()
    _APP.processEvents()

    try:
        canonical = SimpleNamespace(path_count=3)
        _present(viz, canonical)
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)
        baseline = len(_QueryEngine.executed_specs)

        session._on_filters_changed(
            {
                "tx_ids": (2,),
                "path_loss_min_db": -140.0,
                "delay_max_ns": 1_000_000_000.0,
                "contains_interactions": (1, 3),
                "mixed_only": True,
            }
        )
        _wait_for(lambda: len(_QueryEngine.executed_specs) > baseline)

        spec = _QueryEngine.executed_specs[-1]
        assert spec.tx_ids == (2,)
        assert spec.path_loss_min_db == -140.0
        assert spec.delay_max_ns == 1_000_000_000.0
        assert spec.contains_interactions == (1, 3)
        assert spec.mixed_only is True
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_identical_presented_inputs_skip_catalog_and_query_rebuild() -> None:
    _Catalog.created = 0
    _QueryEngine.executed_specs.clear()
    viz = _Visualizer()
    port = _SelectionPort()
    session = _make_session(viz, port)
    session.open()
    _APP.processEvents()

    try:
        canonical = SimpleNamespace(path_count=3)
        view_model, packet = _present(viz, canonical)
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)
        catalog_count = _Catalog.created
        query_count = len(_QueryEngine.executed_specs)

        assert viz.pipeline.callback is not None
        viz.pipeline.callback(2, 4, view_model, packet)
        _APP.processEvents()

        assert _Catalog.created == catalog_count
        assert len(_QueryEngine.executed_specs) == query_count
        assert len(port.frames) == 2
        assert port.frames[-1][1] is False
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_invalid_query_keeps_last_valid_rows_and_starts_no_worker() -> None:
    _QueryEngine.executed_specs.clear()
    viz = _Visualizer()
    session = _make_session(viz)
    session.open()
    _APP.processEvents()

    try:
        _present(viz, SimpleNamespace(path_count=3))
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)
        query_count = len(_QueryEngine.executed_specs)
        original_ids = session._model.path_ids.copy()

        session._on_filters_changed({"path_loss_min_db": float("nan")})
        _APP.processEvents()

        np.testing.assert_array_equal(session._model.path_ids, original_ids)
        assert session._model.rowCount() == 3
        assert len(_QueryEngine.executed_specs) == query_count
        assert not session._query_debounce_timer.isActive()
        assert session.window.status_label.text().startswith("Invalid MPC query:")
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_hiding_waits_for_inflight_query_then_has_no_worker_or_catalog() -> None:
    for event in (
        _BlockingQueryEngine.started,
        _BlockingQueryEngine.release,
        _BlockingQueryEngine.finished,
    ):
        event.clear()
    viz = _Visualizer()
    session = _make_session(viz, query_engine_type=_BlockingQueryEngine)
    session.open()
    _APP.processEvents()

    try:
        _present(viz, SimpleNamespace(path_count=3))
        assert _BlockingQueryEngine.started.wait(timeout=1.0)

        def release_worker() -> None:
            time.sleep(0.05)
            _BlockingQueryEngine.release.set()

        releaser = Thread(target=release_worker, daemon=True)
        releaser.start()
        started_at = time.monotonic()
        session.window.hide()
        elapsed = time.monotonic() - started_at
        releaser.join(timeout=1.0)
        _APP.processEvents()

        assert elapsed >= 0.03
        assert _BlockingQueryEngine.finished.is_set()
        assert session._executor is None
        assert session._active_future is None
        assert session._catalog is None
        assert session._model is None
    finally:
        _BlockingQueryEngine.release.set()
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_same_frame_refresh_preserves_selection_and_new_frame_clears_it() -> None:
    viz = _Visualizer()
    port = _SelectionPort()
    session = _make_session(viz, port)
    viz._mpc_explorer_session = session
    session.open()
    _APP.processEvents()

    try:
        canonical = SimpleNamespace(path_count=3)
        _present(viz, canonical)
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)
        session._select_path(2, origin="table")
        assert session._selected_path_id == 2
        assert port.selections[-1][1:] == (2, "table")

        _present(viz, canonical)
        _wait_for(lambda: len(port.frames) >= 2)
        assert port.frames[-1][1] is False
        assert session._selected_path_id == 2

        replacement = SimpleNamespace(path_count=3)
        _present(viz, replacement, step=5)
        _wait_for(lambda: len(port.frames) >= 3)
        assert port.frames[-1][1] is True
        assert session._selected_path_id is None
        assert "presented frame changed" in port.clear_reasons
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()


def test_selection_port_exclusively_owns_renderer_callback_and_overlay() -> None:
    viz = _Visualizer()
    port = _SelectionPort()
    session = _make_session(viz, port)
    viz._mpc_explorer_session = session
    session.open()
    _APP.processEvents()

    try:
        canonical = SimpleNamespace(path_count=3)
        _present(viz, canonical)
        _wait_for(lambda: session._model is not None and session._model.rowCount() == 3)
        assert viz.renderer.selection_callback is None
        accepted_identity = id(session._render_packet)

        session._select_path(1, origin="viewport", packet_identity=id(object()))
        assert session._selected_path_id is None
        assert port.selections == []

        session._select_path(1, origin="viewport", packet_identity=accepted_identity)
        assert session._selected_path_id == 1
        assert len(port.selections) == 1
        assert port.selections[-1][1:] == (1, "viewport")
        assert port.packet_identities[-1] == accepted_identity

        # Keep lightweight callback adapters usable when they omit the optional
        # identity; the Explorer substitutes its accepted packet identity.
        session._select_path(2, origin="viewport")
        assert session._selected_path_id == 2
        assert port.packet_identities[-1] == accepted_identity

        session._clear_selection(reason="test")
        assert port.clear_reasons[-1] == "test"
        assert viz.renderer.clear_calls == 0
    finally:
        session.shutdown()
        viz.close()
        _APP.processEvents()
