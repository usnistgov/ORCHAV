"""Shared selection tables for saved detections and actual tracker states."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Any

from PySide6.QtCore import QModelIndex, QSignalBlocker, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .display_style import DETECTION_COLOR, detection_color


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, Real) and isfinite(value) else None


def _format(value: Any, decimals: int = 2) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{decimals}f}"


def _cell(
    text: str, sort_value: Any, *, color: QColor | None = None, tooltip: str | None = None
) -> dict[int, Any]:
    """Prepare every supported role once, including clearing roles from an older schema."""
    return {
        Qt.ItemDataRole.DisplayRole: text,
        Qt.ItemDataRole.UserRole: sort_value,
        Qt.ItemDataRole.ForegroundRole: color,
        Qt.ItemDataRole.ToolTipRole: tooltip,
    }


class _NativeResultModel(QStandardItemModel):
    """Keep painting and sorting in Qt without Python virtual data callbacks."""

    def __init__(self, headers: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self._headers = tuple(headers)
        self.setHorizontalHeaderLabels(headers)
        prototype = QStandardItem()
        prototype.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.setItemPrototype(prototype)

    def _set_rows(self, rows: Sequence[Sequence[dict[int, Any]]], headers: Sequence[str]) -> None:
        # Native structural notifications keep proxy indexes valid. Role writes
        # then produce one repaint/sort notification for the complete result.
        self.setColumnCount(len(headers))
        self.setRowCount(len(rows))
        headers_changed = tuple(headers) != self._headers
        with QSignalBlocker(self):
            if headers_changed:
                self.setHorizontalHeaderLabels(headers)
                self._headers = tuple(headers)
            for row, cells in enumerate(rows):
                for column, roles in enumerate(cells):
                    self.setItemData(self.index(row, column), roles)
        if headers_changed:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(headers) - 1)
        if rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(rows) - 1, len(headers) - 1))


class DetectionTableModel(_NativeResultModel):
    """Detection observations; a linked track ID never changes the row's type."""

    HEADERS = ("Detection", "Range (m)", "Radial velocity (m/s)", "SNR (dB)")
    ANGLE_COLUMNS = (("aoa_az_deg", "Azimuth (deg)"), ("aoa_el_deg", "Elevation (deg)"))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.HEADERS, parent)
        self._detections: list[dict] = []
        self._extra_columns: list[tuple[str, str]] = []

    def set_detections(
        self, detections: Sequence[dict], *, elevation_available: bool = True
    ) -> bool:
        columns = []
        if any(d.get("track_id") is not None for d in detections):
            columns.append(("track_id", "Track"))
        columns.extend(
            (key, label)
            for key, label in self.ANGLE_COLUMNS
            if (key != "aoa_el_deg" or elevation_available)
            and any(_number(d.get(key)) is not None for d in detections)
        )
        schema_changed = columns != self._extra_columns
        keys = ("range_m", "velocity_m_s", "snr_db", *(key for key, _ in columns))
        rows = []
        for index, detection in enumerate(detections):
            try:
                color = QColor(detection_color(detection))
            except (TypeError, ValueError):
                color = QColor(DETECTION_COLOR)
            cells = [
                _cell(
                    f"Detection {index + 1}",
                    index,
                    color=color,
                    tooltip=str(detection.get("detection_id", "")),
                )
            ]
            for key in keys:
                value = detection.get(key)
                if key == "velocity_m_s" and value is None:
                    value = detection.get("radial_velocity_m_s")
                text = (
                    str(value)
                    if key == "track_id" and value is not None
                    else _format(value, decimals=1 if key.startswith("aoa_") else 2)
                )
                number = _number(value)
                cells.append(
                    _cell(
                        text,
                        number if number is not None else float("-inf"),
                        tooltip="Receiver-local direction" if key.startswith("aoa_") else None,
                    )
                )
            rows.append(cells)
        self._detections = list(detections)
        self._extra_columns = columns
        self._set_rows(rows, (*self.HEADERS, *(label for _, label in columns)))
        return schema_changed


class TrackTableModel(_NativeResultModel):
    """Actual saved states supplied by the tracking adapter, including coasts."""

    HEADERS = (
        "Track",
        "State",
        "Estimated range (m)",
        "Estimated radial velocity (m/s)",
        "Position (m)",
        "Coordinates",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.HEADERS, parent)
        self.states: list[dict] = []

    def set_states(self, states: Sequence[dict]) -> None:
        rows = []
        for state in states:
            confirmation = state.get("confirmed")
            status = (
                "Confirmed"
                if confirmation is True
                else "Tentative" if confirmation is False else "—"
            )
            if state.get("predicted_only") is True:
                status = "Coasting" if confirmation is None else status + " · coasting"
            position = state.get("position_m")
            values = (
                state.get("track_id"),
                status,
                state.get("estimated_range_m"),
                state.get("radial_velocity_m_s"),
                "—" if position is None else ", ".join(_format(value) for value in position),
                state.get("position_frame", "—"),
            )
            try:
                color = QColor(detection_color(state))
            except (TypeError, ValueError):
                color = None
            kind = {"native": "Native track", "saved_world_track": "Saved world track"}.get(
                state.get("source_kind"), "Saved track"
            )
            source = state.get("source_key")
            tooltip = kind + ("\nSource: " + " / ".join(map(str, source)) if source else "")
            cells = []
            for column, value in enumerate(values):
                text = (
                    _format(value) if column in {2, 3} else str(value if value is not None else "—")
                )
                number = _number(value)
                cells.append(
                    _cell(
                        text,
                        number if number is not None else str(value or ""),
                        color=color if column == 0 else None,
                        tooltip=tooltip if column == 0 else None,
                    )
                )
            rows.append(cells)
        self.states = list(states)
        self._set_rows(rows, self.HEADERS)


class _SelectionTable(QTableView):
    """A second click on a selected row explicitly clears the selection."""

    row_cleared = Signal()
    keyboard_selected = Signal(QModelIndex)

    def request_column_resize(self) -> None:
        """Fit a new schema once; playback keeps widths and user adjustments."""
        self._resize_columns_pending = True
        if self.isVisible():
            self._resize_columns()

    def _resize_columns(self) -> None:
        if not getattr(self, "_resize_columns_pending", False):
            return
        self.resizeColumnsToContents()
        for column, text in getattr(self, "_minimum_column_text", {}).items():
            self.setColumnWidth(
                column,
                max(self.columnWidth(column), self.fontMetrics().horizontalAdvance(text) + 20),
            )
        self._resize_columns_pending = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._resize_columns()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        before = tuple(index.row() for index in self.selectionModel().selectedRows())
        super().keyPressEvent(event)
        after = tuple(index.row() for index in self.selectionModel().selectedRows())
        current = self.currentIndex()
        if before != after and current.isValid() and current.row() in after:
            self.keyboard_selected.emit(current)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self.indexAt(event.position().toPoint())
        self._cleared_on_press = (
            event.button() == Qt.MouseButton.LeftButton
            and index.isValid()
            and self.selectionModel().isRowSelected(index.row(), QModelIndex())
        )
        if self._cleared_on_press:
            self.clearSelection()
            self.setCurrentIndex(QModelIndex())
            self.row_cleared.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if getattr(self, "_cleared_on_press", False):
            self._cleared_on_press = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class SensingResultBrowser(QWidget):
    """One source-scoped selection browser shared by all sensing workspaces."""

    detection_selected = Signal(int)
    track_selected = Signal(object)
    selection_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context_key: tuple = ()
        self._acquisition_id = ""
        self._collection = "detections"
        self._selection_key: tuple | None = None
        self._updating = False
        self._detections_sized = False
        self._tracks_sized = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.context = QLabel()
        header.addWidget(self.context, 1)
        self.clear_button = QPushButton("Clear selection")
        self.clear_button.clicked.connect(lambda: self.clear_selection(notify=True))
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.track_status = QLabel()
        layout.addWidget(self.track_status)
        self.model = DetectionTableModel(self)
        self.track_model = TrackTableModel(self)
        self.table, self.proxy = self._make_table(self.model)
        self.track_table, self.track_proxy = self._make_table(self.track_model)
        self.tabs.addTab(self.table, "Detections")
        self.tabs.addTab(self.track_table, "Tracks")
        self.table.clicked.connect(self._choose_detection)
        self.track_table.clicked.connect(self._choose_track)
        self.table.keyboard_selected.connect(self._choose_detection)
        self.track_table.keyboard_selected.connect(self._choose_track)
        self.table.row_cleared.connect(lambda: self.clear_selection(notify=True))
        self.track_table.row_cleared.connect(lambda: self.clear_selection(notify=True))
        self.tabs.currentChanged.connect(self._restore_selection)
        self.tabs.currentChanged.connect(self._show_track_status)
        self.track_status.hide()
        self.setMinimumHeight(100)

    def _make_table(self, model):
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        proxy.setSortRole(Qt.ItemDataRole.UserRole)
        table = _SelectionTable(self)
        table.setModel(proxy)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSortingEnabled(True)
        table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        if isinstance(model, TrackTableModel):
            table._minimum_column_text = {1: "Confirmed · coasting"}
        table.request_column_resize()
        return table, proxy

    def update_result(
        self,
        detections: Sequence[dict],
        tracks: Sequence[dict] = (),
        *,
        context_key: tuple = (),
        acquisition_id: str = "",
        detection_collection: str = "detections",
        result_label: str = "",
        elevation_available: bool = True,
        tracks_note: str = "No saved track states",
        complete_track_states: bool = False,
    ) -> None:
        """Accept compact rows only; updates never request arrays or tracking decisions."""
        self._updating = True
        try:
            self._context_key = tuple(context_key)
            self._acquisition_id = acquisition_id
            self._collection = detection_collection
            self.context.setText(result_label)
            first_detections = not self._detections_sized and bool(detections)
            schema_changed = self.model.set_detections(
                detections, elevation_available=elevation_available
            )
            first_tracks = not self._tracks_sized and bool(tracks)
            self.track_model.set_states(tracks)
            if schema_changed or first_detections:
                self.table.request_column_resize()
                self._detections_sized = bool(detections)
            if first_tracks:
                self.track_table.request_column_resize()
                self._tracks_sized = True
            self.tabs.setTabText(0, f"Detections ({len(detections)})")
            self.tabs.setTabText(
                1, f"Tracks ({len(tracks)})" if tracks or complete_track_states else "Tracks"
            )
            self.track_status.setText(tracks_note if not tracks else "")
            self._show_track_status()
            self._restore_selection()
        finally:
            self._updating = False

    def update_from_request(
        self,
        request,
        *,
        tracks: Sequence[dict] = (),
        result_label: str = "",
        tracks_note: str = "No saved track states",
        complete_track_states: bool = False,
    ) -> None:
        """Use the accepted request's metadata; never access its dense products."""
        sensing = request.display_frame.get("sensing") or {}
        temporal = sensing.get("temporal") or {}
        result = temporal.get("result") or {}
        acquisition = temporal.get("acquisition") or {}
        context = (
            request.source_identity,
            (temporal.get("clock") or {}).get("clock_id"),
            result.get("processing_run_id"),
            acquisition.get("pair_id"),
        )
        self.update_result(
            sensing.get("detections") or (),
            tracks,
            context_key=context,
            acquisition_id=acquisition.get("acquisition_id", ""),
            detection_collection=sensing.get("_display_detection_collection", "detections"),
            result_label=result_label,
            elevation_available=(sensing.get("config_resolved") or {}).get("antenna_rows") != 1,
            tracks_note=tracks_note,
            complete_track_states=complete_track_states,
        )

    def _detection_key(self, row: int) -> tuple:
        detection = self.model._detections[row]
        identity = (
            ("id", detection["detection_id"])
            if detection.get("detection_id") is not None
            else ("row", row)
        )
        return ("detection", self._context_key, self._acquisition_id, self._collection, identity)

    def _show_track_status(self, *_args) -> None:
        self.track_status.setVisible(
            self.tabs.currentIndex() == 1 and bool(self.track_status.text())
        )

    def _track_key(self, row: int) -> tuple:
        state = self.track_model.states[row]
        return (
            "track",
            self._context_key,
            state.get("source_key", ()),
            state.get("source_kind"),
            state.get("track_id"),
        )

    def _choose_detection(self, index: QModelIndex) -> None:
        if self._updating or not index.isValid():
            return
        row = self.proxy.mapToSource(index).row()
        self._selection_key = self._detection_key(row)
        self.track_table.clearSelection()
        self.detection_selected.emit(row)

    def _choose_track(self, index: QModelIndex) -> None:
        if self._updating or not index.isValid():
            return
        row = self.track_proxy.mapToSource(index).row()
        self._selection_key = self._track_key(row)
        self.table.clearSelection()
        self.track_selected.emit(self.track_model.states[row]["track_id"])

    def select_detection(self, row: int) -> None:
        """Synchronize a plot selection without emitting a new user action."""
        self._selection_key = self._detection_key(row) if 0 <= row < self.model.rowCount() else None
        self._restore_selection()

    def select_track(self, track_id: Any) -> None:
        self._selection_key = next(
            (
                self._track_key(row)
                for row, state in enumerate(self.track_model.states)
                if state.get("track_id") == track_id
            ),
            None,
        )
        self._restore_selection()

    def _restore_selection(self, *_args) -> None:
        if self._selection_key is None:
            for table in (self.table, self.track_table):
                if table.selectionModel().hasSelection():
                    table.clearSelection()
                if table.currentIndex().isValid():
                    table.setCurrentIndex(QModelIndex())
            return
        for model, proxy, table, key in (
            (self.model, self.proxy, self.table, self._detection_key),
            (self.track_model, self.track_proxy, self.track_table, self._track_key),
        ):
            for row in range(model.rowCount()):
                if key(row) == self._selection_key:
                    other = self.track_table if table is self.table else self.table
                    if other.selectionModel().hasSelection():
                        other.clearSelection()
                    target = proxy.mapFromSource(model.index(row, 0)).row()
                    if (
                        not table.selectionModel().isRowSelected(target, QModelIndex())
                        or table.currentIndex().row() != target
                    ):
                        table.selectRow(target)
                    return
        self._selection_key = None
        self._restore_selection()

    def clear_selection(self, *, notify: bool = False) -> None:
        self._selection_key = None
        self.table.clearSelection()
        self.track_table.clearSelection()
        self.table.setCurrentIndex(QModelIndex())
        self.track_table.setCurrentIndex(QModelIndex())
        if notify:
            self.selection_cleared.emit()
