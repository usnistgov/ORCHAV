"""Scalable read-only Qt model for MPC Explorer query results."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Any, Iterable, Optional

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal

from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog
from visualizer.src.metrics.mpc_path_query import (
    DEFAULT_SORT_SPEC,
    MpcGrouping,
    MpcQueryResult,
    MpcQuerySpec,
    MpcSortField,
    MpcSortSpec,
    SortDirection,
    group_key_for_path,
    replace_sort_primary,
)
from visualizer.src.services.mpc_interaction_style_service import mpc_interaction_label


class MpcExplorerColumn(str, Enum):
    """Columns supported by the flat Explorer table."""

    PATH_ID = "path_id"
    TX = "tx"
    RX = "rx"
    PATH_LOSS = "path_loss"
    DELAY = "delay"
    INTERACTIONS = "interactions"
    INTERACTION_MIX = "interaction_mix"
    FIRST_MATERIAL = "first_material"
    STATUS = "status"
    AOD_AZIMUTH = "aod_azimuth"
    AOD_ELEVATION = "aod_elevation"
    AOA_AZIMUTH = "aoa_azimuth"
    AOA_ELEVATION = "aoa_elevation"
    GEOMETRIC_LENGTH = "geometric_length"
    STRETCH_RATIO = "stretch_ratio"
    EXCESS_DELAY = "excess_delay"
    STRENGTH_RANK = "strength_rank"
    RELATIVE_PATH_LOSS = "relative_path_loss"
    RELATIVE_POWER = "relative_power"
    INTERACTION_SEQUENCE = "interaction_sequence"
    MATERIAL_SEQUENCE = "material_sequence"
    DELAY_PROVENANCE = "delay_provenance"
    PATH_LOSS_PROVENANCE = "path_loss_provenance"


DEFAULT_COLUMNS: tuple[MpcExplorerColumn, ...] = (
    MpcExplorerColumn.PATH_ID,
    MpcExplorerColumn.TX,
    MpcExplorerColumn.RX,
    MpcExplorerColumn.PATH_LOSS,
    MpcExplorerColumn.DELAY,
    MpcExplorerColumn.INTERACTIONS,
    MpcExplorerColumn.INTERACTION_MIX,
    MpcExplorerColumn.FIRST_MATERIAL,
    MpcExplorerColumn.STATUS,
)

OPTIONAL_COLUMNS: tuple[MpcExplorerColumn, ...] = tuple(
    column for column in MpcExplorerColumn if column not in DEFAULT_COLUMNS
)

_COLUMN_HEADERS = {
    MpcExplorerColumn.PATH_ID: "Path ID",
    MpcExplorerColumn.TX: "TX",
    MpcExplorerColumn.RX: "RX",
    MpcExplorerColumn.PATH_LOSS: "Path Loss (dB)",
    MpcExplorerColumn.DELAY: "Delay (ns)",
    MpcExplorerColumn.INTERACTIONS: "Interactions",
    MpcExplorerColumn.INTERACTION_MIX: "Interaction Mix",
    MpcExplorerColumn.FIRST_MATERIAL: "First Material",
    MpcExplorerColumn.STATUS: "Rendered / Filtered",
    MpcExplorerColumn.AOD_AZIMUTH: "AoD Azimuth (world deg)",
    MpcExplorerColumn.AOD_ELEVATION: "AoD Elevation (world deg)",
    MpcExplorerColumn.AOA_AZIMUTH: "AoA Azimuth (world deg)",
    MpcExplorerColumn.AOA_ELEVATION: "AoA Elevation (world deg)",
    MpcExplorerColumn.GEOMETRIC_LENGTH: "Length (m)",
    MpcExplorerColumn.STRETCH_RATIO: "Stretch Ratio",
    MpcExplorerColumn.EXCESS_DELAY: "Excess Delay (ns)",
    MpcExplorerColumn.STRENGTH_RANK: "Pair Strength Rank",
    MpcExplorerColumn.RELATIVE_PATH_LOSS: "Delta Loss from Strongest (dB)",
    MpcExplorerColumn.RELATIVE_POWER: "Relative Power Proxy",
    MpcExplorerColumn.INTERACTION_SEQUENCE: "Interaction Sequence",
    MpcExplorerColumn.MATERIAL_SEQUENCE: "Material Sequence",
    MpcExplorerColumn.DELAY_PROVENANCE: "Delay Provenance",
    MpcExplorerColumn.PATH_LOSS_PROVENANCE: "Loss Provenance",
}

_SORT_FIELD_BY_COLUMN = {
    MpcExplorerColumn.PATH_ID: MpcSortField.PATH_ID,
    MpcExplorerColumn.TX: MpcSortField.TX,
    MpcExplorerColumn.RX: MpcSortField.RX,
    MpcExplorerColumn.PATH_LOSS: MpcSortField.PATH_LOSS,
    MpcExplorerColumn.DELAY: MpcSortField.DELAY,
    MpcExplorerColumn.INTERACTIONS: MpcSortField.INTERACTIONS,
    MpcExplorerColumn.INTERACTION_MIX: MpcSortField.INTERACTION_MIX,
    MpcExplorerColumn.FIRST_MATERIAL: MpcSortField.FIRST_MATERIAL,
    MpcExplorerColumn.AOD_AZIMUTH: MpcSortField.AOD_AZIMUTH,
    MpcExplorerColumn.AOD_ELEVATION: MpcSortField.AOD_ELEVATION,
    MpcExplorerColumn.AOA_AZIMUTH: MpcSortField.AOA_AZIMUTH,
    MpcExplorerColumn.AOA_ELEVATION: MpcSortField.AOA_ELEVATION,
    MpcExplorerColumn.GEOMETRIC_LENGTH: MpcSortField.GEOMETRIC_LENGTH,
    MpcExplorerColumn.STRETCH_RATIO: MpcSortField.STRETCH_RATIO,
    MpcExplorerColumn.EXCESS_DELAY: MpcSortField.EXCESS_DELAY,
    MpcExplorerColumn.STRENGTH_RANK: MpcSortField.STRENGTH_RANK,
    MpcExplorerColumn.RELATIVE_PATH_LOSS: MpcSortField.RELATIVE_PATH_LOSS,
    MpcExplorerColumn.RELATIVE_POWER: MpcSortField.RELATIVE_POWER,
}

_NUMERIC_COLUMNS = frozenset(
    {
        MpcExplorerColumn.PATH_ID,
        MpcExplorerColumn.TX,
        MpcExplorerColumn.RX,
        MpcExplorerColumn.PATH_LOSS,
        MpcExplorerColumn.DELAY,
        MpcExplorerColumn.INTERACTIONS,
        MpcExplorerColumn.AOD_AZIMUTH,
        MpcExplorerColumn.AOD_ELEVATION,
        MpcExplorerColumn.AOA_AZIMUTH,
        MpcExplorerColumn.AOA_ELEVATION,
        MpcExplorerColumn.GEOMETRIC_LENGTH,
        MpcExplorerColumn.STRETCH_RATIO,
        MpcExplorerColumn.EXCESS_DELAY,
        MpcExplorerColumn.STRENGTH_RANK,
        MpcExplorerColumn.RELATIVE_PATH_LOSS,
        MpcExplorerColumn.RELATIVE_POWER,
    }
)

_ANGLE_COLUMNS = frozenset(
    {
        MpcExplorerColumn.AOD_AZIMUTH,
        MpcExplorerColumn.AOD_ELEVATION,
        MpcExplorerColumn.AOA_AZIMUTH,
        MpcExplorerColumn.AOA_ELEVATION,
    }
)


class MpcExplorerTableModel(QAbstractTableModel):
    """Flat, incrementally exposed view over one int32 path permutation.

    Sorting is intentionally not performed in ``sort()``.  The method emits
    :attr:`sortRequested`; the Explorer session runs
    :class:`~visualizer.src.metrics.mpc_path_query.MpcPathQueryEngine` on its
    latest-only worker, then calls :meth:`apply_query_result` on the GUI thread.
    """

    sortRequested = Signal(object)

    ROW_HEIGHT = 22
    DEFAULT_FETCH_BATCH_SIZE = 50_000
    PATH_ROW_CACHE_SIZE = 64

    PathIdRole = int(Qt.ItemDataRole.UserRole) + 1
    RawValueRole = int(Qt.ItemDataRole.UserRole) + 2
    GroupBoundaryRole = int(Qt.ItemDataRole.UserRole) + 3
    GroupKeyRole = int(Qt.ItemDataRole.UserRole) + 4

    def __init__(
        self,
        catalog: Optional[MpcPathCatalog] = None,
        *,
        generation: int = 0,
        columns: Iterable[MpcExplorerColumn | str] = DEFAULT_COLUMNS,
        fetch_batch_size: int = DEFAULT_FETCH_BATCH_SIZE,
        defer_expensive_columns: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        batch_size = int(fetch_batch_size)
        if not 25_000 <= batch_size <= 50_000:
            raise ValueError("fetch_batch_size must be between 25,000 and 50,000")
        self._fetch_batch_size = batch_size
        self._columns = self._normalize_columns(columns)
        self._catalog = catalog
        self._pending_catalog = catalog
        self._generation = int(generation)
        self._path_ids = np.empty((0,), dtype=np.int32)
        self._loaded_rows = 0
        self._query_spec = MpcQuerySpec(sort=DEFAULT_SORT_SPEC)
        self._sort_spec = DEFAULT_SORT_SPEC
        self._defer_expensive_columns = bool(defer_expensive_columns)
        self._path_row_cache: OrderedDict[int, Optional[int]] = OrderedDict()
        self._permutation_revision = 0

    @property
    def catalog(self) -> Optional[MpcPathCatalog]:
        """Catalog backing the currently displayed last-good permutation."""
        return self._catalog

    @property
    def generation(self) -> int:
        """Generation accepted by :meth:`apply_query_result`."""
        return self._generation

    @property
    def query_spec(self) -> MpcQuerySpec:
        """Query spec associated with the current permutation."""
        return self._query_spec

    @property
    def sort_spec(self) -> MpcSortSpec:
        """Current user-visible compound sort."""
        return self._sort_spec

    @property
    def columns(self) -> tuple[MpcExplorerColumn, ...]:
        """Current visible columns."""
        return self._columns

    @property
    def total_row_count(self) -> int:
        """Total matching rows, including rows not fetched into Qt yet."""
        return int(self._path_ids.shape[0])

    @property
    def loaded_row_count(self) -> int:
        """Rows currently exposed through ``rowCount()``."""
        return self._loaded_rows

    @property
    def fetch_batch_size(self) -> int:
        """Configured incremental insertion size."""
        return self._fetch_batch_size

    @property
    def permutation_revision(self) -> int:
        """Monotonic identity of the currently displayed path permutation."""
        return self._permutation_revision

    def begin_generation(
        self,
        catalog: MpcPathCatalog,
        *,
        generation: int,
        clear_rows: bool = True,
    ) -> None:
        """Start a query generation, optionally retaining last-good rows.

        ``clear_rows=False`` is intended for a local sort/filter query. The
        current display catalog and permutation remain coherent until a result
        for ``catalog`` is accepted.
        """
        self._generation = int(generation)
        self._pending_catalog = catalog
        if not clear_rows:
            if self._catalog is None:
                self._catalog = catalog
            return
        self.beginResetModel()
        self._catalog = catalog
        self._path_ids = np.empty((0,), dtype=np.int32)
        self._loaded_rows = 0
        self._path_row_cache.clear()
        self._permutation_revision += 1
        self.endResetModel()

    def apply_query_result(self, result: MpcQueryResult) -> bool:
        """Apply a worker result, rejecting obsolete generations before reset."""
        if not isinstance(result, MpcQueryResult):
            raise TypeError("result must be an MpcQueryResult")
        if result.generation != self._generation:
            return False
        catalog = self._pending_catalog
        if catalog is None:
            return False
        if result.total_path_count != catalog.path_count:
            return False

        self.beginResetModel()
        # MpcQueryResult validates dtype and contiguity. Retain the worker's
        # int32 permutation directly instead of making a second full copy.
        self._catalog = catalog
        self._path_ids = result.path_ids
        self._loaded_rows = min(self._fetch_batch_size, self.total_row_count)
        self._query_spec = result.spec
        self._sort_spec = result.spec.sort
        self._path_row_cache.clear()
        self._permutation_revision += 1
        self.endResetModel()
        return True

    def clear(self, *, generation: Optional[int] = None) -> None:
        """Release all frame/query references."""
        self.beginResetModel()
        self._catalog = None
        self._pending_catalog = None
        if generation is not None:
            self._generation = int(generation)
        self._path_ids = np.empty((0,), dtype=np.int32)
        self._loaded_rows = 0
        self._path_row_cache.clear()
        self._permutation_revision += 1
        self.endResetModel()

    def set_columns(self, columns: Iterable[MpcExplorerColumn | str]) -> None:
        """Replace the visible column list without rebuilding path order."""
        normalized = self._normalize_columns(columns)
        if normalized == self._columns:
            return
        self.beginResetModel()
        self._columns = normalized
        self.endResetModel()

    def set_sort_spec(self, sort_spec: MpcSortSpec) -> None:
        """Update visible compound-sort chips without sorting in the GUI thread."""
        if not isinstance(sort_spec, MpcSortSpec):
            raise TypeError("sort_spec must be an MpcSortSpec")
        self._sort_spec = sort_spec

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Return only the fetched prefix so Qt never instantiates 2M rows at once."""
        return 0 if parent.isValid() else self._loaded_rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Return the visible flat-table column count."""
        return 0 if parent.isValid() else len(self._columns)

    def canFetchMore(self, parent: QModelIndex = QModelIndex()) -> bool:  # noqa: N802
        """Whether another fixed-size result batch can be inserted."""
        return not parent.isValid() and self._loaded_rows < self.total_row_count

    def fetchMore(self, parent: QModelIndex = QModelIndex()) -> None:  # noqa: N802
        """Expose the next 25k-50k rows using proper Qt insert notifications."""
        if parent.isValid() or not self.canFetchMore(parent):
            return
        first = self._loaded_rows
        last_exclusive = min(first + self._fetch_batch_size, self.total_row_count)
        self.beginInsertRows(QModelIndex(), first, last_exclusive - 1)
        self._loaded_rows = last_exclusive
        self.endInsertRows()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return stable headers without measuring cell contents."""
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._columns):
            if role == Qt.ItemDataRole.ToolTipRole and self._columns[section] in _ANGLE_COLUMNS:
                return (
                    "World-space degrees. Azimuth is [0, 360) from +X toward +Y; "
                    "elevation is [-90, 90] toward +Z."
                )
            if role != Qt.ItemDataRole.DisplayRole:
                return None
            return _COLUMN_HEADERS[self._columns[section]]
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical and 0 <= section < self._loaded_rows:
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Format only the visible cell requested by Qt."""
        if (
            not index.isValid()
            or self._catalog is None
            or not 0 <= index.row() < self._loaded_rows
            or not 0 <= index.column() < len(self._columns)
        ):
            return None
        path_id = int(self._path_ids[index.row()])
        column = self._columns[index.column()]

        if role == self.PathIdRole:
            return path_id
        if role == self.GroupBoundaryRole:
            return self._is_group_boundary(index.row())
        if role == self.GroupKeyRole:
            return group_key_for_path(self._catalog, path_id, self._query_spec)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            horizontal = (
                Qt.AlignmentFlag.AlignRight
                if column in _NUMERIC_COLUMNS
                else Qt.AlignmentFlag.AlignLeft
            )
            return int(horizontal | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if column is MpcExplorerColumn.RELATIVE_POWER:
                return (
                    "Dimensionless 10^(-path_loss_db/10) proxy within this TX/RX pair; "
                    "not received dBm or absolute power."
                )
            if column is MpcExplorerColumn.STATUS:
                return "Filtered is pre-Top-K; rendered is the final segment population."
            return None
        if role == Qt.ItemDataRole.EditRole:
            return None
        if role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.UserRole,
            self.RawValueRole,
        ):
            return None
        if self._defer_expensive_columns and not self._catalog.derived_column_is_ready(
            column.value
        ):
            return "--" if role == Qt.ItemDataRole.DisplayRole else None

        raw_value = self._raw_value(path_id, column)
        if role == self.RawValueRole or role == Qt.ItemDataRole.UserRole:
            return self._qt_scalar(raw_value)
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(column, raw_value)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Keep table cells read-only and row-selectable."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def sort(
        self,
        column: int,
        order: Qt.SortOrder = Qt.SortOrder.AscendingOrder,
    ) -> None:
        """Request a worker-side vectorized sort for a clicked column."""
        if not 0 <= int(column) < len(self._columns):
            return
        sort_field = _SORT_FIELD_BY_COLUMN.get(self._columns[int(column)])
        if sort_field is None:
            return
        direction = (
            SortDirection.ASCENDING
            if order == Qt.SortOrder.AscendingOrder
            else SortDirection.DESCENDING
        )
        self._sort_spec = replace_sort_primary(self._sort_spec, sort_field, direction)
        self.sortRequested.emit(self._sort_spec)

    def path_id_for_row(self, row: int) -> Optional[int]:
        """Return canonical identity for a result row, fetched or not."""
        index = int(row)
        if index < 0 or index >= self.total_row_count:
            return None
        return int(self._path_ids[index])

    def row_for_path_id(self, path_id: int, *, fetch: bool = False) -> Optional[int]:
        """Return a worker-resolved row without scanning on the GUI thread.

        A small LRU is populated by :meth:`cache_path_row`.  Cold lookups
        intentionally return ``None``; the Explorer session resolves them on
        its scalar-lookup worker against :meth:`path_lookup_snapshot`.
        """
        canonical_path_id = int(path_id)
        if canonical_path_id in self._path_row_cache:
            row = self._path_row_cache.pop(canonical_path_id)
            self._path_row_cache[canonical_path_id] = row
        else:
            return None
        if row is None:
            return None
        if fetch:
            self.ensure_row_loaded(row)
        return row

    def cached_path_row(self, path_id: int) -> tuple[bool, Optional[int]]:
        """Return ``(resolved, row)`` for one cached canonical path identity."""
        canonical_path_id = int(path_id)
        if canonical_path_id not in self._path_row_cache:
            return False, None
        row = self._path_row_cache.pop(canonical_path_id)
        self._path_row_cache[canonical_path_id] = row
        return True, row

    def path_lookup_snapshot(self) -> tuple[int, np.ndarray]:
        """Borrow the immutable displayed permutation for a scalar worker lookup.

        The model never mutates an accepted query-result array.  The returned
        revision lets the GUI thread reject a result if a newer query replaces
        the permutation while the worker is searching it.
        """
        return self._permutation_revision, self._path_ids

    def cache_path_row(
        self,
        path_id: int,
        row: Optional[int],
        *,
        permutation_revision: int,
    ) -> bool:
        """Accept one worker lookup only for the still-current permutation."""
        if int(permutation_revision) != self._permutation_revision:
            return False
        canonical_path_id = int(path_id)
        normalized_row = None if row is None else int(row)
        if normalized_row is not None:
            if normalized_row < 0 or normalized_row >= self.total_row_count:
                return False
            if int(self._path_ids[normalized_row]) != canonical_path_id:
                return False
        self._remember_path_row(canonical_path_id, normalized_row)
        return True

    def ensure_row_loaded(self, row: int) -> bool:
        """Advance at most one fetch batch toward a target result row.

        Repeated event-loop calls are used for distant viewport selections so
        the GUI never emits dozens of insert operations in one blocking call.
        """
        target = int(row)
        if target < 0 or target >= self.total_row_count:
            return False
        if target < self._loaded_rows:
            return True
        if self.canFetchMore():
            self.fetchMore()
        return target < self._loaded_rows

    def _remember_path_row(self, path_id: int, row: Optional[int]) -> None:
        """Retain one bounded canonical-ID lookup, including misses."""
        self._path_row_cache[path_id] = row
        self._path_row_cache.move_to_end(path_id)
        while len(self._path_row_cache) > self.PATH_ROW_CACHE_SIZE:
            self._path_row_cache.popitem(last=False)

    def _raw_value(self, path_id: int, column: MpcExplorerColumn) -> Any:
        catalog = self._catalog
        assert catalog is not None
        if column is MpcExplorerColumn.PATH_ID:
            return path_id
        if column is MpcExplorerColumn.TX:
            return catalog.tx_ids[path_id]
        if column is MpcExplorerColumn.RX:
            return catalog.rx_ids[path_id]
        if column is MpcExplorerColumn.PATH_LOSS:
            return catalog.path_losses_db[path_id]
        if column is MpcExplorerColumn.DELAY:
            return catalog.delays_ns[path_id]
        if column is MpcExplorerColumn.INTERACTIONS:
            return catalog.interaction_counts[path_id]
        if column is MpcExplorerColumn.INTERACTION_MIX:
            return self._interaction_mix_text(path_id)
        if column is MpcExplorerColumn.FIRST_MATERIAL:
            return self._material_text(catalog.first_material_id(path_id))
        if column is MpcExplorerColumn.STATUS:
            return self._status_text(path_id)
        if column is MpcExplorerColumn.AOD_AZIMUTH:
            return catalog.aod_azimuth_deg[path_id]
        if column is MpcExplorerColumn.AOD_ELEVATION:
            return catalog.aod_elevation_deg[path_id]
        if column is MpcExplorerColumn.AOA_AZIMUTH:
            return catalog.aoa_azimuth_deg[path_id]
        if column is MpcExplorerColumn.AOA_ELEVATION:
            return catalog.aoa_elevation_deg[path_id]
        if column is MpcExplorerColumn.GEOMETRIC_LENGTH:
            return catalog.geometric_lengths_m[path_id]
        if column is MpcExplorerColumn.STRETCH_RATIO:
            return catalog.stretch_ratios[path_id]
        if column is MpcExplorerColumn.EXCESS_DELAY:
            return catalog.pair_relative.excess_delay_ns[path_id]
        if column is MpcExplorerColumn.STRENGTH_RANK:
            rank = int(catalog.pair_relative.strength_rank[path_id])
            return rank if rank > 0 else None
        if column is MpcExplorerColumn.RELATIVE_PATH_LOSS:
            return catalog.pair_relative.path_loss_delta_db[path_id]
        if column is MpcExplorerColumn.RELATIVE_POWER:
            return catalog.pair_relative.relative_power_proxy[path_id]
        if column is MpcExplorerColumn.INTERACTION_SEQUENCE:
            return self._interaction_sequence_text(path_id)
        if column is MpcExplorerColumn.MATERIAL_SEQUENCE:
            return self._material_sequence_text(path_id)
        if column is MpcExplorerColumn.DELAY_PROVENANCE:
            return catalog.delay_provenance(path_id)
        if column is MpcExplorerColumn.PATH_LOSS_PROVENANCE:
            return catalog.path_loss_provenance(path_id)
        raise AssertionError(f"unhandled MPC Explorer column: {column}")

    @staticmethod
    def _display_value(column: MpcExplorerColumn, value: Any) -> str:
        if value is None:
            return "--"
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return "--"
            decimals = {
                MpcExplorerColumn.PATH_LOSS: 2,
                MpcExplorerColumn.DELAY: 3,
                MpcExplorerColumn.AOD_AZIMUTH: 2,
                MpcExplorerColumn.AOD_ELEVATION: 2,
                MpcExplorerColumn.AOA_AZIMUTH: 2,
                MpcExplorerColumn.AOA_ELEVATION: 2,
                MpcExplorerColumn.GEOMETRIC_LENGTH: 3,
                MpcExplorerColumn.STRETCH_RATIO: 3,
                MpcExplorerColumn.EXCESS_DELAY: 3,
                MpcExplorerColumn.RELATIVE_PATH_LOSS: 2,
                MpcExplorerColumn.RELATIVE_POWER: 5,
            }.get(column, 3)
            return f"{numeric:.{decimals}f}"
        if isinstance(value, (int, np.integer)):
            numeric = int(value)
            if column in (MpcExplorerColumn.TX, MpcExplorerColumn.RX) and numeric < 0:
                return "--"
            return str(numeric)
        return str(value)

    def _interaction_sequence_text(self, path_id: int) -> str:
        assert self._catalog is not None
        sequence = self._catalog.interaction_sequence(path_id)
        if sequence.size == 0:
            return "LoS"
        return " -> ".join(
            mpc_interaction_label(int(value), compact=True, explicit_unknown=True)
            for value in sequence
        )

    def _interaction_mix_text(self, path_id: int) -> str:
        assert self._catalog is not None
        sequence = self._catalog.interaction_sequence(path_id)
        if sequence.size == 0:
            return "LoS"
        unique = tuple(dict.fromkeys(int(value) for value in sequence))
        labels = [
            mpc_interaction_label(value, compact=True, explicit_unknown=True) for value in unique
        ]
        return labels[0] if len(labels) == 1 else "Mixed: " + " + ".join(labels)

    def _material_text(self, material_id: int) -> str:
        assert self._catalog is not None
        if material_id < 0:
            return "Unavailable"
        if material_id == 0:
            return "None"
        return self._catalog.material_name(material_id) or f"Material {material_id}"

    def _material_sequence_text(self, path_id: int) -> str:
        assert self._catalog is not None
        sequence = self._catalog.material_sequence(path_id)
        if sequence is None:
            return "Unavailable"
        if sequence.size == 0:
            return "None"
        return " -> ".join(self._material_text(int(value)) for value in sequence)

    def _status_text(self, path_id: int) -> str:
        assert self._catalog is not None
        if self._catalog.is_rendered(path_id):
            return "Rendered"
        if self._catalog.is_filtered(path_id):
            return "Filtered - outside rendered set"
        return "Outside current filtered set"

    def _is_group_boundary(self, row: int) -> bool:
        if row <= 0 or self._catalog is None or self._query_spec.grouping is MpcGrouping.NONE:
            return False
        current_id = int(self._path_ids[row])
        previous_id = int(self._path_ids[row - 1])
        return group_key_for_path(
            self._catalog,
            current_id,
            self._query_spec,
        ) != group_key_for_path(
            self._catalog,
            previous_id,
            self._query_spec,
        )

    @staticmethod
    def _qt_scalar(value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    @staticmethod
    def _normalize_columns(
        columns: Iterable[MpcExplorerColumn | str],
    ) -> tuple[MpcExplorerColumn, ...]:
        normalized = tuple(MpcExplorerColumn(column) for column in columns)
        if not normalized:
            raise ValueError("at least one MPC Explorer column is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("MPC Explorer columns must be unique")
        return normalized


__all__ = [
    "DEFAULT_COLUMNS",
    "OPTIONAL_COLUMNS",
    "MpcExplorerColumn",
    "MpcExplorerTableModel",
]
