"""Lazy MPC Explorer session, query worker, and presented-frame lifecycle.

Nothing in this module is imported during normal visualizer startup. The
session exists only while the user has opened the MPC Explorer, and canonical
arrays are not wrapped or queried until the renderer accepts a frame.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Signal

from shared.logging import get_logger

if TYPE_CHECKING:
    from ..pipeline.core import FrameRenderPacket, ViewModel

logger = get_logger("orchav.mpc_explorer")

QUERY_DEBOUNCE_MS = 180
QUERY_FETCH_BATCH_SIZE = 50_000
ROW_LOOKUP_CHUNK_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class MpcPresentedFrameToken:
    """Frame-local selection identity after successful renderer acceptance."""

    source_epoch: int
    step: int
    canonical_identity: int


class MpcExplorerSelectionPort(Protocol):
    """Optional visual-selection collaborator bound to one Explorer session."""

    def set_presented_frame(
        self,
        token: MpcPresentedFrameToken,
        catalog: Any,
        render_packet: FrameRenderPacket,
        *,
        frame_changed: bool,
    ) -> None:
        """Adopt one accepted frame and preserve selection only when allowed."""

    def select_path(
        self,
        token: MpcPresentedFrameToken,
        canonical_path_id: int,
        *,
        origin: str,
        packet_identity: Optional[int] = None,
    ) -> None:
        """Select a canonical path from a table, viewport, or later plot."""

    def clear_presented_frame(self, *, reason: str) -> None:
        """Clear transient selection, overlay objects, and animation."""

    def prepare_viewport_mapping(
        self,
        token: MpcPresentedFrameToken,
        catalog: Any,
        render_packet: FrameRenderPacket,
    ) -> bool:
        """Publish one worker-prepared filtered-segment pick mapping."""

    def refresh_selected_details(self) -> None:
        """Refresh mask-derived details without rebuilding selected geometry."""


@dataclass(frozen=True, slots=True)
class _QueryRequest:
    generation: int
    catalog: Any
    spec: Any
    query_engine_type: Any


@dataclass(frozen=True, slots=True)
class _RowLookupRequest:
    """One latest-only scalar lookup against a borrowed path permutation."""

    serial: int
    lifecycle_generation: int
    permutation_revision: int
    path_id: int
    path_ids: np.ndarray


class _QueryResultBridge(QObject):
    """Marshal one background NumPy query result onto the Qt GUI thread."""

    finished = Signal(object, object, object)


class _RowLookupResultBridge(QObject):
    """Marshal one scalar permutation lookup onto the Qt GUI thread."""

    finished = Signal(object, object, object)


def _execute_query_request(request: _QueryRequest) -> Any:
    """Execute one immutable catalog query in a worker thread."""
    engine = request.query_engine_type(request.catalog)
    return engine.execute(request.spec, generation=request.generation)


def _execute_row_lookup_request(request: _RowLookupRequest) -> Optional[int]:
    """Find one canonical path without copying or scanning on the GUI thread."""
    path_ids = np.asarray(request.path_ids)
    if path_ids.ndim != 1 or path_ids.dtype != np.int32 or not path_ids.flags.c_contiguous:
        raise ValueError("row lookup requires a C-contiguous int32 path permutation")
    target = int(request.path_id)
    for start in range(0, int(path_ids.size), ROW_LOOKUP_CHUNK_SIZE):
        stop = min(start + ROW_LOOKUP_CHUNK_SIZE, int(path_ids.size))
        matches = np.flatnonzero(path_ids[start:stop] == target)
        if matches.size:
            return start + int(matches[0])
    return None


class MpcExplorerSession(QObject):
    """Own the Explorer only while its separate window is visibly active."""

    closed = Signal()
    pathSelectionRequested = Signal(object, int, str)

    def __init__(
        self,
        visualizer: Any,
        *,
        window_factory: Optional[Callable[..., Any]] = None,
        catalog_factory: Optional[Callable[..., Any]] = None,
        model_factory: Optional[Callable[..., Any]] = None,
        query_engine_type: Any = None,
        query_types: Optional[dict[str, Any]] = None,
        selection_port: Optional[MpcExplorerSelectionPort] = None,
        selection_port_owned: bool = False,
    ) -> None:
        """Store lightweight factories; defer every data object until presentation."""
        parent = visualizer if isinstance(visualizer, QObject) else None
        super().__init__(parent)
        self.visualizer = visualizer
        self._window_factory = window_factory
        self._catalog_factory = catalog_factory
        self._model_factory = model_factory
        self._query_engine_type = query_engine_type
        self._query_types = query_types
        self._selection_port = selection_port
        self._owns_selection_port = bool(selection_port is not None and selection_port_owned)
        self._default_columns: tuple[Any, ...] = ()
        self._optional_columns: tuple[Any, ...] = ()

        self.window: Any = None
        self._catalog: Any = None
        self._model: Any = None
        self._render_packet: Any = None
        self._frame_token: Optional[MpcPresentedFrameToken] = None
        self._catalog_input_signature: Any = None
        self._selected_path_id: Optional[int] = None
        self._presentation_revision = 0
        self._lifecycle_generation = 0
        self._query_generation = 0
        self._active_query_generation: Optional[int] = None
        self._active_future: Optional[Future[Any]] = None
        self._pending_request: Optional[_QueryRequest] = None
        self._debounced_request: Optional[_QueryRequest] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._row_lookup_serial = 0
        self._active_row_lookup_serial: Optional[int] = None
        self._active_row_lookup_future: Optional[Future[Optional[int]]] = None
        self._pending_row_lookup_request: Optional[_RowLookupRequest] = None
        self._row_lookup_executor: Optional[ThreadPoolExecutor] = None
        self._selection_signal_origin: Optional[str] = None
        self._active = False
        self._finished = False

        self._scope_key = "all"
        self._grouping_key = "tx_rx"
        self._preset_key = "tx_rx_strongest"
        self._sort_spec: Any = None
        self._filters: dict[str, Any] = {}
        self._visible_optional_columns: set[str] = set()

        self._presented_callback = self._on_presented_frame
        self._viewport_callback: Optional[Callable[..., None]] = None

        self._query_bridge = _QueryResultBridge(self)
        self._query_bridge.finished.connect(self._on_query_finished)
        self._row_lookup_bridge = _RowLookupResultBridge(self)
        self._row_lookup_bridge.finished.connect(self._on_row_lookup_finished)
        self._query_debounce_timer = QTimer(self)
        self._query_debounce_timer.setSingleShot(True)
        self._query_debounce_timer.timeout.connect(self._submit_debounced_query)
        self._connect_selection_port_signals(self._selection_port)

    @property
    def frame_token(self) -> Optional[MpcPresentedFrameToken]:
        """Return the accepted frame currently represented by the table."""
        return self._frame_token

    @property
    def is_active(self) -> bool:
        """Return whether the Explorer is visible and subscribed."""
        return self._active and not self._finished

    def bind_selection_port(
        self,
        port: Optional[MpcExplorerSelectionPort],
        *,
        owned: bool = False,
    ) -> None:
        """Bind or replace the renderer-neutral transient selection owner."""
        if port is self._selection_port:
            return
        old_port = self._selection_port
        old_owned = self._owns_selection_port
        if old_port is None and port is not None:
            self._remove_viewport_callback()
        self._disconnect_selection_port_signals(old_port)
        self._selection_port = port
        self._owns_selection_port = bool(port is not None and owned)
        self._connect_selection_port_signals(port)
        if old_port is not None:
            self._call_selection_port(
                old_port,
                "clear_presented_frame",
                reason="selection port replaced",
            )
            if old_owned:
                shutdown = getattr(old_port, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                delete_later = getattr(old_port, "deleteLater", None)
                if callable(delete_later):
                    delete_later()
        if (
            port is not None
            and self._frame_token is not None
            and self._catalog is not None
            and self._render_packet is not None
        ):
            self._call_selection_port(
                port,
                "set_presented_frame",
                self._frame_token,
                self._catalog,
                self._render_packet,
                frame_changed=True,
            )
        elif port is None and self.is_active and self._frame_token is not None:
            self._install_viewport_callback()

    def _connect_selection_port_signals(self, port: Any) -> None:
        if port is None:
            return
        selection_signal = getattr(port, "selectionChanged", None)
        if selection_signal is not None:
            selection_signal.connect(self._on_selection_port_changed)
        details_signal = getattr(port, "detailsChanged", None)
        if details_signal is not None:
            details_signal.connect(self._on_selection_details_changed)

    def _disconnect_selection_port_signals(self, port: Any) -> None:
        if port is None:
            return
        for signal_name, callback in (
            ("selectionChanged", self._on_selection_port_changed),
            ("detailsChanged", self._on_selection_details_changed),
        ):
            signal = getattr(port, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass

    def _on_selection_port_changed(self, selection: Any) -> None:
        """Mirror table focus from viewport-originated canonical selection."""
        if selection is None:
            self._selected_path_id = None
            self._invalidate_row_lookup()
            if self.window is not None:
                self.window.table.clearSelection()
            return
        token = getattr(selection, "frame_token", None)
        path_id = getattr(selection, "canonical_path_id", None)
        if token != self._frame_token or path_id is None:
            return
        self._selected_path_id = int(path_id)
        # A table-originated signal is a synchronous echo of the row the user
        # already selected. Viewport/plot origins need a worker lookup because
        # the displayed permutation can contain millions of rows.
        if self._selection_signal_origin != "table":
            self._request_path_row_reveal(self._selected_path_id)

    def _on_selection_details_changed(self, details: Any) -> None:
        """Render authoritative selected-path details from the selection owner."""
        if self.window is None:
            return
        if details is None:
            self.window.clear_details()
            return
        token = getattr(details, "frame_token", None)
        if token != self._frame_token:
            return
        self.window.set_details(self._format_selection_details(details))

    @staticmethod
    def _format_selection_details(details: Any) -> list[tuple[str, Any]]:
        """Format selected-path status and value provenance without RF invention."""
        if not is_dataclass(details):
            return [("Details", str(details))]

        delay_value = getattr(details, "delay_ns", None)
        loss_value = getattr(details, "path_loss_db", None)
        delay_provenance = MpcExplorerSession._provenance_label(
            getattr(details, "delay_is_estimated", None)
        )
        loss_provenance = MpcExplorerSession._provenance_label(
            getattr(details, "path_loss_is_estimated", None)
        )
        values: list[tuple[str, Any]] = [
            ("Path ID", getattr(details, "canonical_path_id", "Unavailable")),
            ("Selection origin", getattr(details, "origin", "Unavailable")),
            ("TX", MpcExplorerSession._available_value(getattr(details, "tx_id", None))),
            ("RX", MpcExplorerSession._available_value(getattr(details, "rx_id", None))),
            (
                "Path loss",
                (
                    "Unavailable"
                    if loss_value is None
                    else f"{float(loss_value):.3f} dB ({loss_provenance})"
                ),
            ),
            (
                "Delay",
                (
                    "Unavailable"
                    if delay_value is None
                    else f"{float(delay_value):.3f} ns ({delay_provenance})"
                ),
            ),
            ("Interactions", getattr(details, "interaction_count", 0)),
            (
                "Interaction sequence",
                " -> ".join(getattr(details, "interaction_labels", ())) or "None",
            ),
            (
                "Material sequence",
                MpcExplorerSession._format_material_sequence(details),
            ),
            (
                "Geometric length",
                f"{float(getattr(details, 'geometric_length_m', 0.0)):.3f} m",
            ),
            (
                "Filtered",
                "Yes" if bool(getattr(details, "is_filtered", False)) else "No",
            ),
            (
                "Rendered",
                "Yes" if bool(getattr(details, "is_rendered", False)) else "No",
            ),
            ("Status", getattr(details, "render_status", "Unavailable")),
        ]
        known = {
            "identity",
            "origin",
            "tx_id",
            "rx_id",
            "path_loss_db",
            "delay_ns",
            "interaction_count",
            "interaction_types",
            "interaction_labels",
            "material_ids",
            "material_names",
            "geometric_length_m",
            "is_filtered",
            "is_rendered",
            "render_status",
            "delay_is_estimated",
            "path_loss_is_estimated",
        }
        for field in fields(details):
            if field.name not in known:
                values.append((field.name.replace("_", " ").title(), getattr(details, field.name)))
        return values

    @staticmethod
    def _available_value(value: Any) -> Any:
        return "Unavailable" if value is None else value

    @staticmethod
    def _format_material_sequence(details: Any) -> str:
        """Pair material IDs with names so numeric filters are discoverable."""
        material_ids = getattr(details, "material_ids", None)
        material_names = tuple(getattr(details, "material_names", ()) or ())
        if material_ids is None:
            return "Unavailable"
        labels = []
        for index, material_id in enumerate(material_ids):
            name = material_names[index] if index < len(material_names) else ""
            labels.append(f"{int(material_id)} {name}".strip())
        return " -> ".join(labels) or "None"

    @staticmethod
    def _provenance_label(estimated: Optional[bool]) -> str:
        if estimated is True:
            return "estimated"
        if estimated is False:
            return "authoritative"
        return "provenance unavailable"

    def open(self) -> None:
        """Create and show the separate window on explicit user request."""
        if self._finished:
            return
        window = self._ensure_window()
        window.show()
        window.raise_()
        window.activateWindow()

    def toggle(self) -> None:
        """Close a visible Explorer or restore a hidden one."""
        if self._finished:
            return
        window = self._ensure_window()
        if window.isVisible() and not window.isMinimized():
            window.close()
            return
        window.showNormal()
        window.raise_()
        window.activateWindow()

    def on_scenario_teardown(self) -> None:
        """Retire frame-local data before scenario-owned resources disappear."""
        if self._finished:
            return
        self._clear_presented_data(reason="scenario changed")
        if self.window is not None:
            self.window.set_status("Waiting for the new scenario's first presented MPC frame...")

    def shutdown(self) -> None:
        """Synchronously detach callbacks and release frame-heavy references."""
        if self._finished:
            return
        self._deactivate(reason="Explorer closed")
        self._shutdown_selection_port()
        window = self.window
        self.window = None
        self._finished = True
        if window is not None:
            try:
                window.close()
            except RuntimeError:
                pass
        self._clear_root_reference()
        self.closed.emit()
        self.deleteLater()

    def _ensure_window(self) -> Any:
        if self.window is not None:
            return self.window
        factory = self._window_factory
        if factory is None:
            from ..panels.mpc_explorer_window import MpcExplorerWindow

            factory = MpcExplorerWindow
        parent = self.visualizer if isinstance(self.visualizer, QObject) else None
        try:
            window = factory(parent)
        except TypeError:
            window = factory()
        self.window = window
        window.activityChanged.connect(self._on_window_activity_changed)
        window.closing.connect(self._on_window_closing)
        window.destroyed.connect(self._on_window_destroyed)
        window.scopeChanged.connect(self._on_scope_changed)
        window.groupingChanged.connect(self._on_grouping_changed)
        window.sortPresetChanged.connect(self._on_sort_preset_changed)
        window.sortClausesChanged.connect(self._on_sort_clauses_changed)
        window.filtersChanged.connect(self._on_filters_changed)
        column_signal = getattr(window, "columnVisibilityChanged", None)
        if column_signal is not None:
            column_signal.connect(self._on_column_visibility_changed)
        window.pathSelectionRequested.connect(
            lambda path_id: self._select_path(path_id, origin="table")
        )
        return window

    def _on_window_activity_changed(self, active: bool) -> None:
        if self._finished:
            return
        if active:
            self._activate()
        else:
            self._deactivate(reason="Explorer hidden")

    def _on_window_closing(self) -> None:
        if not self._finished:
            self._deactivate(reason="Explorer closed")

    def _on_window_destroyed(self, _object: Any = None) -> None:
        if self._finished:
            return
        self.window = None
        self._deactivate(reason="Explorer destroyed")
        self._shutdown_selection_port()
        self._finished = True
        self._clear_root_reference()
        self.closed.emit()
        self.deleteLater()

    def _activate(self) -> None:
        if self._active or self._finished:
            return
        self._active = True
        self._lifecycle_generation += 1
        pipeline = getattr(self.visualizer, "pipeline", None)
        subscribe = getattr(pipeline, "set_mpc_explorer_presented_callback", None)
        if callable(subscribe):
            subscribe(self._presented_callback)
        if self._selection_port is None:
            self._install_viewport_callback()
        if self.window is not None:
            self.window.set_status("Waiting for a successfully presented MPC frame...")
        generation = self._lifecycle_generation
        QTimer.singleShot(0, lambda: self._force_current_frame(generation))

    def _deactivate(self, *, reason: str) -> None:
        if not self._active and self._catalog is None and self._model is None:
            return
        self._active = False
        self._lifecycle_generation += 1
        pipeline = getattr(self.visualizer, "pipeline", None)
        unsubscribe = getattr(pipeline, "clear_mpc_explorer_presented_callback", None)
        if callable(unsubscribe):
            unsubscribe(self._presented_callback)
        if self._selection_port is None:
            self._remove_viewport_callback()
        self._clear_presented_data(reason=reason)
        if self.window is not None and not self._finished:
            self.window.set_status("Explorer paused; no MPC indexing is active.")

    def _force_current_frame(self, activation_generation: int) -> None:
        if (
            not self.is_active
            or activation_generation != self._lifecycle_generation
            or bool(getattr(self.visualizer, "_shutdown_started", False))
        ):
            return
        pipeline = getattr(self.visualizer, "pipeline", None)
        update = getattr(pipeline, "update", None)
        if not callable(update):
            if self.window is not None:
                self.window.set_status("Frame pipeline is not available.")
            return
        try:
            self.visualizer.force_update_next_frame = True
            step = int(getattr(self.visualizer, "animation_step", 0))
            update(step)
        except (RuntimeError, AttributeError, ValueError):
            logger.exception("Unable to refresh the current frame for MPC Explorer")
            if self.window is not None:
                self.window.set_status("Could not refresh the current MPC frame.")

    def _on_presented_frame(
        self,
        source_epoch: int,
        step: int,
        view_model: ViewModel,
        render_packet: FrameRenderPacket,
    ) -> None:
        """Adopt one frame only after the renderer accepted its transaction."""
        if not self.is_active:
            return
        canonical_data = getattr(view_model, "canonical_data", None)
        if canonical_data is None:
            canonical_data = getattr(render_packet, "canonical_data", None)
        if canonical_data is None:
            self._clear_presented_data(reason="presented frame has no MPC paths")
            if self.window is not None:
                self.window.set_status("The presented frame contains no MPC paths.")
            return

        token = MpcPresentedFrameToken(
            source_epoch=int(source_epoch),
            step=int(step),
            canonical_identity=id(canonical_data),
        )
        frame_changed = token != self._frame_token
        input_signature = self._frame_catalog_input_signature(view_model, render_packet)
        if (
            not frame_changed
            and self._catalog is not None
            and input_signature == self._catalog_input_signature
        ):
            self._render_packet = render_packet
            self._presentation_revision += 1
            port = self._selection_port
            if port is not None:
                self._call_selection_port(
                    port,
                    "set_presented_frame",
                    token,
                    self._catalog,
                    render_packet,
                    frame_changed=False,
                )
                if bool(
                    getattr(
                        self._catalog,
                        "rendered_segment_mapping_is_ready",
                        False,
                    )
                ):
                    self._call_selection_port(
                        port,
                        "prepare_viewport_mapping",
                        token,
                        self._catalog,
                        render_packet,
                    )
            if self.window is not None:
                self.window.set_status(
                    f"Frame {token.step} - revision {self._presentation_revision}"
                )
            return

        had_catalog = self._catalog is not None
        if frame_changed:
            self._clear_selection(reason="presented frame changed")
        self._frame_token = token
        self._render_packet = render_packet
        self._presentation_revision += 1

        try:
            catalog = self._create_catalog(canonical_data, view_model, render_packet)
            self._install_catalog(
                catalog,
                reset_counts=frame_changed or not had_catalog,
            )
        except (RuntimeError, ValueError, TypeError, ImportError) as exc:
            logger.exception("Unable to initialize MPC Explorer frame: %s", exc)
            self._clear_presented_data(reason="Explorer frame initialization failed")
            if self.window is not None:
                self.window.set_status(f"MPC Explorer could not index this frame: {exc}")
            return
        self._catalog_input_signature = input_signature

        port = self._selection_port
        if port is not None:
            self._call_selection_port(
                port,
                "set_presented_frame",
                token,
                catalog,
                render_packet,
                frame_changed=frame_changed,
            )
        if port is None:
            self._install_viewport_callback()
        self._queue_query(
            debounce=False,
            clear_model=frame_changed or not had_catalog,
        )

    @staticmethod
    def _frame_catalog_input_signature(view_model: Any, render_packet: Any) -> tuple[Any, ...]:
        """Identify borrowed mask inputs without inspecting their contents."""

        def borrowed_array_identity(values: Any) -> tuple[Any, ...]:
            if values is None:
                return (None,)
            return (
                id(values),
                tuple(getattr(values, "shape", ()) or ()),
                str(getattr(values, "dtype", "")),
            )

        visibility = getattr(view_model, "mpc_visibility", None)
        return (
            borrowed_array_identity(getattr(view_model, "path_mask", None)),
            borrowed_array_identity(getattr(render_packet, "segment_mask", None)),
            bool(getattr(visibility, "effective_paths", True)),
        )

    def _load_shared_types(self) -> None:
        if (
            self._catalog_factory is not None
            and self._model_factory is not None
            and self._query_engine_type is not None
            and self._query_types is not None
        ):
            return
        from ..metrics.mpc_path_catalog import MpcPathCatalog, MpcPathScope
        from ..metrics.mpc_path_query import (
            MpcGrouping,
            MpcPathQueryEngine,
            MpcQuerySpec,
            MpcSortClause,
            MpcSortField,
            MpcSortPreset,
            MpcSortSpec,
            SortDirection,
            grouping_for_preset,
            sort_spec_for_preset,
        )
        from ..model.mpc_explorer_model import (
            DEFAULT_COLUMNS,
            OPTIONAL_COLUMNS,
            MpcExplorerTableModel,
        )

        self._catalog_factory = self._catalog_factory or MpcPathCatalog
        self._model_factory = self._model_factory or MpcExplorerTableModel
        self._query_engine_type = self._query_engine_type or MpcPathQueryEngine
        self._default_columns = tuple(DEFAULT_COLUMNS)
        self._optional_columns = tuple(OPTIONAL_COLUMNS)
        self._query_types = self._query_types or {
            "scope": MpcPathScope,
            "grouping": MpcGrouping,
            "query_spec": MpcQuerySpec,
            "sort_clause": MpcSortClause,
            "sort_field": MpcSortField,
            "sort_preset": MpcSortPreset,
            "sort_spec": MpcSortSpec,
            "sort_direction": SortDirection,
            "grouping_for_preset": grouping_for_preset,
            "sort_spec_for_preset": sort_spec_for_preset,
        }

    def _shutdown_selection_port(self) -> None:
        port = self._selection_port
        if port is None:
            return
        self._disconnect_selection_port_signals(port)
        self._selection_port = None
        owned = self._owns_selection_port
        self._owns_selection_port = False
        shutdown = getattr(port, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except (RuntimeError, ValueError, TypeError):
                logger.exception("Unable to shut down MPC selection service")
        if owned:
            delete_later = getattr(port, "deleteLater", None)
            if callable(delete_later):
                delete_later()

    def _create_catalog(
        self,
        canonical_data: Any,
        view_model: ViewModel,
        render_packet: FrameRenderPacket,
    ) -> Any:
        self._load_shared_types()
        assert self._catalog_factory is not None
        kwargs: dict[str, Any] = {
            "filtered_path_mask": getattr(view_model, "path_mask", None),
            "rendered_segment_mask": getattr(render_packet, "segment_mask", None),
            "validate": False,
        }
        parameters = signature(self._catalog_factory).parameters
        if "rendered_paths_enabled" in parameters:
            visibility = getattr(view_model, "mpc_visibility", None)
            kwargs["rendered_paths_enabled"] = bool(getattr(visibility, "effective_paths", True))
        return self._catalog_factory(canonical_data, **kwargs)

    def _install_catalog(self, catalog: Any, *, reset_counts: bool) -> None:
        self._catalog = catalog
        if self._model is None:
            assert self._model_factory is not None
            columns = (*self._default_columns, *self._optional_columns)
            model_kwargs: dict[str, Any] = {
                "fetch_batch_size": QUERY_FETCH_BATCH_SIZE,
            }
            model_parameters = signature(self._model_factory).parameters
            if "defer_expensive_columns" in model_parameters:
                model_kwargs["defer_expensive_columns"] = True
            if columns:
                model_kwargs["columns"] = columns
            try:
                self._model = self._model_factory(**model_kwargs)
            except TypeError:
                self._model = self._model_factory()
            sort_signal = getattr(self._model, "sortRequested", None)
            if sort_signal is not None:
                sort_signal.connect(self._on_model_sort_requested)
            if self.window is not None:
                optional_start = len(self._default_columns) if self._optional_columns else None
                self.window.set_model(
                    self._model,
                    optional_start_column=optional_start,
                )
        if self.window is not None:
            if reset_counts:
                self.window.set_counts(0, int(getattr(catalog, "path_count", 0)))
            material_mapping = getattr(
                getattr(catalog, "canonical_data", None),
                "material_id_to_name",
                None,
            )
            set_material_options = getattr(self.window, "set_material_filter_options", None)
            if callable(set_material_options):
                set_material_options(self._normalized_material_mapping(material_mapping))
            self.window.set_status(
                f"Indexing presented frame {self._frame_token.step if self._frame_token else '?'}..."
            )

    @staticmethod
    def _normalized_material_mapping(values: Any) -> dict[int, str]:
        """Normalize frame-local names for filter help without touching path arrays."""
        if not hasattr(values, "items"):
            return {}
        result: dict[int, str] = {}
        for raw_id, raw_name in values.items():
            try:
                material_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if material_id > 0:
                result[material_id] = str(raw_name or "")
        return result

    def _queue_query(self, *, debounce: bool, clear_model: bool = False) -> None:
        if not self.is_active or self._catalog is None or self._model is None:
            return
        generation = self._query_generation + 1
        try:
            request = self._make_query_request(generation)
        except ValueError as exc:
            # Invalidate older work while preserving the last valid rows.
            self._query_generation = generation
            self._pending_request = None
            self._debounced_request = None
            self._query_debounce_timer.stop()
            self._begin_model_generation(generation, clear_rows=False)
            self._show_invalid_query(exc)
            return

        self._query_generation = generation
        self._pending_request = None
        self._begin_model_generation(generation, clear_rows=clear_model)
        if debounce:
            self._debounced_request = request
            self._query_debounce_timer.start(QUERY_DEBOUNCE_MS)
        else:
            self._debounced_request = None
            self._query_debounce_timer.stop()
            self._submit_query(request)

    def _submit_debounced_query(self) -> None:
        request = self._debounced_request
        self._debounced_request = None
        if (
            not self.is_active
            or self._catalog is None
            or request is None
            or request.generation != self._query_generation
        ):
            return
        self._submit_query(request)

    def _begin_model_generation(self, generation: int, *, clear_rows: bool) -> None:
        """Advance stale-result rejection while optionally retaining valid rows."""
        begin_generation = getattr(self._model, "begin_generation", None)
        if not callable(begin_generation):
            return
        try:
            parameters = signature(begin_generation).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "clear_rows" in parameters:
            begin_generation(
                self._catalog,
                generation=generation,
                clear_rows=bool(clear_rows),
            )
        else:
            begin_generation(self._catalog, generation=generation)

    def _show_invalid_query(self, error: ValueError) -> None:
        """Report an invalid filter/sort request without starting worker work."""
        if self.window is not None:
            self.window.set_status(f"Invalid MPC query: {error}")

    def _make_query_request(self, generation: int) -> _QueryRequest:
        self._load_shared_types()
        assert self._query_engine_type is not None
        return _QueryRequest(
            generation=generation,
            catalog=self._catalog,
            spec=self._build_query_spec(),
            query_engine_type=self._query_engine_type,
        )

    def _submit_query(self, request: _QueryRequest) -> None:
        if not self.is_active or request.generation != self._query_generation:
            return
        if self._active_future is not None and not self._active_future.done():
            self._pending_request = request
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="MpcExplorerQuery",
            )
        future = self._executor.submit(_execute_query_request, request)
        self._active_future = future
        self._active_query_generation = request.generation

        def deliver(completed: Future[Any], *, generation: int = request.generation) -> None:
            try:
                result = completed.result()
                error: Optional[BaseException] = None
            except BaseException as exc:  # worker defects are reported on the GUI thread
                result = None
                error = exc
            try:
                self._query_bridge.finished.emit(generation, result, error)
            except RuntimeError:
                pass

        future.add_done_callback(deliver)

    def _on_query_finished(self, generation: Any, result: Any, error: Any) -> None:
        generation = int(generation)
        was_active = generation == self._active_query_generation
        if was_active:
            self._active_future = None
            self._active_query_generation = None

        if self.is_active and generation == self._query_generation and self._model is not None:
            if error is not None:
                logger.error(
                    "MPC Explorer query generation %s failed",
                    generation,
                    exc_info=(type(error), error, error.__traceback__),
                )
                if self.window is not None:
                    self.window.set_status(f"MPC query failed: {error}")
            else:
                apply_result = getattr(self._model, "apply_query_result", None)
                applied = bool(apply_result(result)) if callable(apply_result) else False
                if applied and self.window is not None:
                    visible_count = int(np.asarray(result.path_ids).size)
                    total_count = int(getattr(self._catalog, "path_count", 0))
                    self.window.set_counts(visible_count, total_count)
                    self.window.set_status(
                        f"Frame {self._frame_token.step if self._frame_token else '?'}"
                        f" - revision {self._presentation_revision}"
                    )
                    if self._selected_path_id is not None:
                        self._request_path_row_reveal(self._selected_path_id)
                if applied and self._selection_port is not None:
                    if (
                        self._frame_token is not None
                        and self._catalog is not None
                        and self._render_packet is not None
                    ):
                        self._call_selection_port(
                            self._selection_port,
                            "prepare_viewport_mapping",
                            self._frame_token,
                            self._catalog,
                            self._render_packet,
                        )
                    self._call_selection_port(
                        self._selection_port,
                        "refresh_selected_details",
                    )

        if was_active and self._pending_request is not None:
            pending = self._pending_request
            self._pending_request = None
            if self.is_active and pending.generation == self._query_generation:
                self._submit_query(pending)

    def _request_path_row_reveal(self, path_id: int) -> None:
        """Resolve one selected canonical ID without blocking the GUI thread."""
        if (
            not self.is_active
            or self.window is None
            or self._model is None
            or self._selected_path_id != int(path_id)
        ):
            return
        model = self._model
        cached_lookup = getattr(model, "cached_path_row", None)
        if callable(cached_lookup):
            try:
                resolved, row = cached_lookup(path_id)
            except (IndexError, TypeError, ValueError):
                resolved, row = False, None
            if resolved:
                serial = self._next_row_lookup_serial()
                if row is not None:
                    revision = int(getattr(model, "permutation_revision", 0))
                    self._continue_path_row_reveal(
                        int(path_id),
                        int(row),
                        revision,
                        serial,
                    )
                return

        snapshot = getattr(model, "path_lookup_snapshot", None)
        if not callable(snapshot):
            # Compatibility for small external/test models. The production
            # model always exposes the worker snapshot and never scans here.
            self.window.select_path(int(path_id))
            return
        try:
            permutation_revision, path_ids = snapshot()
        except (RuntimeError, TypeError, ValueError):
            return
        serial = self._next_row_lookup_serial()
        request = _RowLookupRequest(
            serial=serial,
            lifecycle_generation=self._lifecycle_generation,
            permutation_revision=int(permutation_revision),
            path_id=int(path_id),
            path_ids=path_ids,
        )
        future = self._active_row_lookup_future
        if future is not None and not future.done():
            self._pending_row_lookup_request = request
            return
        self._submit_row_lookup(request)

    def _next_row_lookup_serial(self) -> int:
        """Invalidate older scalar lookup and incremental-reveal callbacks."""
        self._row_lookup_serial += 1
        self._pending_row_lookup_request = None
        return self._row_lookup_serial

    def _invalidate_row_lookup(self) -> None:
        """Make every pending row result/reveal obsolete without blocking."""
        self._next_row_lookup_serial()

    def _submit_row_lookup(self, request: _RowLookupRequest) -> None:
        if (
            not self.is_active
            or request.serial != self._row_lookup_serial
            or request.lifecycle_generation != self._lifecycle_generation
        ):
            return
        if self._row_lookup_executor is None:
            self._row_lookup_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="MpcExplorerRowLookup",
            )
        future = self._row_lookup_executor.submit(_execute_row_lookup_request, request)
        self._active_row_lookup_future = future
        self._active_row_lookup_serial = request.serial

        def deliver(completed: Future[Optional[int]]) -> None:
            try:
                row = completed.result()
                error: Optional[BaseException] = None
            except BaseException as exc:
                row = None
                error = exc
            try:
                self._row_lookup_bridge.finished.emit(request, row, error)
            except RuntimeError:
                pass

        future.add_done_callback(deliver)

    def _on_row_lookup_finished(
        self,
        request: Any,
        row: Any,
        error: Any,
    ) -> None:
        if not isinstance(request, _RowLookupRequest):
            return
        was_active = request.serial == self._active_row_lookup_serial
        if was_active:
            self._active_row_lookup_future = None
            self._active_row_lookup_serial = None

        if (
            error is not None
            and request.serial == self._row_lookup_serial
            and request.lifecycle_generation == self._lifecycle_generation
        ):
            logger.error(
                "MPC Explorer row lookup failed",
                exc_info=(type(error), error, error.__traceback__),
            )
        elif (
            self.is_active
            and request.serial == self._row_lookup_serial
            and request.lifecycle_generation == self._lifecycle_generation
            and request.path_id == self._selected_path_id
            and self._model is not None
        ):
            normalized_row = None if row is None else int(row)
            cache_result = getattr(self._model, "cache_path_row", None)
            cached = False
            if callable(cache_result):
                cached = bool(
                    cache_result(
                        request.path_id,
                        normalized_row,
                        permutation_revision=request.permutation_revision,
                    )
                )
            if cached and normalized_row is not None:
                self._continue_path_row_reveal(
                    request.path_id,
                    normalized_row,
                    request.permutation_revision,
                    request.serial,
                )

        if was_active and self._pending_row_lookup_request is not None:
            pending = self._pending_row_lookup_request
            self._pending_row_lookup_request = None
            if (
                self.is_active
                and pending.serial == self._row_lookup_serial
                and pending.lifecycle_generation == self._lifecycle_generation
            ):
                self._submit_row_lookup(pending)

    def _continue_path_row_reveal(
        self,
        path_id: int,
        row: int,
        permutation_revision: int,
        serial: int,
    ) -> None:
        """Expose at most one fetch batch per event-loop turn, then scroll."""
        if (
            not self.is_active
            or serial != self._row_lookup_serial
            or path_id != self._selected_path_id
            or self.window is None
            or self._model is None
            or int(getattr(self._model, "permutation_revision", permutation_revision))
            != permutation_revision
        ):
            return
        model = self._model
        if row < int(model.rowCount()):
            self.window.select_path(path_id)
            return
        ensure_row = getattr(model, "ensure_row_loaded", None)
        if not callable(ensure_row):
            return
        loaded_before = int(model.rowCount())
        loaded = bool(ensure_row(row))
        if loaded and row < int(model.rowCount()):
            self.window.select_path(path_id)
            return
        if int(model.rowCount()) <= loaded_before:
            return
        QTimer.singleShot(
            0,
            lambda: self._continue_path_row_reveal(
                path_id,
                row,
                permutation_revision,
                serial,
            ),
        )

    def _build_query_spec(self) -> Any:
        assert self._query_types is not None
        scope_type = self._query_types["scope"]
        grouping_type = self._query_types["grouping"]
        query_spec_type = self._query_types["query_spec"]
        if self._sort_spec is None:
            self._sort_spec = self._sort_spec_for_preset(self._preset_key)
        kwargs: dict[str, Any] = {
            "scope": scope_type(self._scope_key),
            "grouping": grouping_type(self._grouping_key),
            "sort": self._sort_spec,
            **self._filters,
        }
        try:
            parameters = signature(query_spec_type).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "prewarm_columns" in parameters:
            kwargs["prewarm_columns"] = tuple(sorted(self._visible_optional_columns))
        if "include_pick_mapping" in parameters:
            kwargs["include_pick_mapping"] = True
        return query_spec_type(
            **kwargs,
        )

    def _sort_spec_for_preset(self, preset_key: str) -> Any:
        assert self._query_types is not None
        preset_type = self._query_types["sort_preset"]
        preset = preset_type(preset_key)
        return self._query_types["sort_spec_for_preset"](preset)

    def _on_scope_changed(self, scope: str) -> None:
        self._scope_key = str(scope)
        self._queue_query(debounce=True)

    def _on_grouping_changed(self, grouping: str) -> None:
        self._grouping_key = str(grouping)
        self._queue_query(debounce=True)

    def _on_sort_preset_changed(self, preset: str) -> None:
        self._load_shared_types()
        self._preset_key = str(preset)
        self._sort_spec = self._sort_spec_for_preset(self._preset_key)
        preset_value = self._query_types["sort_preset"](self._preset_key)
        grouping = self._query_types["grouping_for_preset"](preset_value)
        self._grouping_key = str(getattr(grouping, "value", grouping))
        if self.window is not None:
            with QSignalBlocker(self.window.grouping_combo):
                index = self.window.grouping_combo.findData(self._grouping_key)
                if index >= 0:
                    self.window.grouping_combo.setCurrentIndex(index)
            refresh_sort_context = getattr(self.window, "refresh_sort_context", None)
            if callable(refresh_sort_context):
                refresh_sort_context()
        self._queue_query(debounce=True)

    def _on_sort_clauses_changed(self, clauses: Any) -> None:
        self._load_shared_types()
        assert self._query_types is not None
        clause_type = self._query_types["sort_clause"]
        field_type = self._query_types["sort_field"]
        direction_type = self._query_types["sort_direction"]
        sort_spec_type = self._query_types["sort_spec"]
        typed_clauses = tuple(
            clause_type(
                field=field_type(str(field)),
                direction=direction_type("ascending" if ascending else "descending"),
            )
            for field, ascending in clauses
        )
        try:
            self._sort_spec = sort_spec_type(clauses=typed_clauses)
        except TypeError:
            self._sort_spec = sort_spec_type(typed_clauses)
        self._queue_query(debounce=True)

    def _on_filters_changed(self, filters: Any) -> None:
        """Store lightweight filter values and debounce the NumPy query."""
        self._filters = dict(filters)
        self._queue_query(debounce=True)

    def _on_column_visibility_changed(self, column_name: str, visible: bool) -> None:
        """Warm an optional column only after the user explicitly exposes it."""
        normalized = str(column_name)
        if visible:
            if normalized in self._visible_optional_columns:
                return
            self._visible_optional_columns.add(normalized)
            if self.window is not None:
                self.window.set_status(f"Preparing {normalized.replace('_', ' ')}...")
            self._queue_query(debounce=True)
        else:
            self._visible_optional_columns.discard(normalized)

    def _on_model_sort_requested(self, sort_spec: Any) -> None:
        """Treat a standard table-header click as one global custom order."""
        self._sort_spec = sort_spec
        self._grouping_key = "none"
        if self.window is not None:
            self.window.set_sort_spec(sort_spec, global_order=True)
        self._queue_query(debounce=True)

    def _select_path(
        self,
        path_id: int,
        *,
        origin: str,
        packet_identity: Optional[int] = None,
    ) -> None:
        if self._frame_token is None or self._catalog is None:
            return
        if origin == "viewport":
            expected_identity = id(self._render_packet) if self._render_packet is not None else None
            if expected_identity is None:
                return
            if packet_identity is None:
                # Compatibility for renderer test doubles and older optional
                # callback adapters. Production pygfx always supplies it.
                packet_identity = expected_identity
            elif packet_identity != expected_identity:
                return
        path_id = int(path_id)
        path_count = int(getattr(self._catalog, "path_count", 0))
        if path_id < 0 or path_id >= path_count:
            return
        self._selected_path_id = path_id
        port = self._selection_port
        if port is not None:
            selection_kwargs: dict[str, Any] = {"origin": origin}
            if origin == "viewport":
                selection_kwargs["packet_identity"] = packet_identity
            previous_origin = self._selection_signal_origin
            self._selection_signal_origin = origin
            try:
                self._call_selection_port(
                    port,
                    "select_path",
                    self._frame_token,
                    path_id,
                    **selection_kwargs,
                )
            finally:
                self._selection_signal_origin = previous_origin
        elif self.window is not None:
            if origin != "table":
                self._request_path_row_reveal(path_id)
            self.window.set_details(self._path_details(path_id))
        self.pathSelectionRequested.emit(self._frame_token, path_id, origin)

    def _path_details(self, path_id: int) -> list[tuple[str, Any]]:
        catalog = self._catalog
        details: list[tuple[str, Any]] = [("Path ID", path_id)]
        for label, column_names in (
            ("TX", ("tx", "tx_id")),
            ("RX", ("rx", "rx_id")),
            ("Path loss", ("path_loss", "path_loss_db")),
            ("Delay", ("delay", "delay_ns")),
            ("Interactions", ("interactions", "interaction_count")),
            ("Geometric length", ("geometric_length_m", "geometric_length", "path_length")),
        ):
            value = self._catalog_scalar(catalog, path_id, column_names)
            if value is not None:
                details.append((label, value))
        for label, method_name in (
            ("Interaction sequence", "interaction_sequence"),
            ("Material sequence", "material_sequence"),
        ):
            method = getattr(catalog, method_name, None)
            if callable(method):
                try:
                    sequence = method(path_id)
                except (IndexError, TypeError, ValueError):
                    continue
                details.append((label, " -> ".join(map(str, sequence)) or "None"))
        details.extend(self._scope_status_details(catalog, path_id))
        return details

    def _scope_status_details(self, catalog: Any, path_id: int) -> list[tuple[str, str]]:
        if self._query_types is None:
            return []
        scope_type = self._query_types["scope"]
        result: list[tuple[str, str]] = []
        for label, key in (("Filtered", "filtered"), ("Rendered", "rendered")):
            try:
                mask = np.asarray(catalog.scope_mask(scope_type(key)), dtype=bool)
                included = path_id < mask.size and bool(mask[path_id])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            result.append((label, "Yes" if included else "No"))
        return result

    @staticmethod
    def _catalog_scalar(
        catalog: Any,
        path_id: int,
        column_names: tuple[str, ...],
    ) -> Any:
        column = getattr(catalog, "column", None)
        if not callable(column):
            return None
        for name in column_names:
            try:
                values = column(name)
                value = values[path_id]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                return "Unavailable"
            return value
        return None

    def _install_viewport_callback(self) -> None:
        if self._selection_port is not None:
            return
        renderer = getattr(self.visualizer, "renderer", None)
        setter = getattr(renderer, "set_mpc_path_selection_callback", None)
        if not callable(setter):
            return
        if not self.is_active or self._frame_token is None:
            setter(None)
            self._viewport_callback = None
            return
        generation = self._lifecycle_generation
        token = self._frame_token

        def on_path(
            path_id: int,
            packet_identity: Optional[int] = None,
        ) -> None:
            if (
                self.is_active
                and generation == self._lifecycle_generation
                and token == self._frame_token
            ):
                self._select_path(
                    path_id,
                    origin="viewport",
                    packet_identity=packet_identity,
                )

        self._viewport_callback = on_path
        setter(on_path)

    def _remove_viewport_callback(self) -> None:
        renderer = getattr(self.visualizer, "renderer", None)
        setter = getattr(renderer, "set_mpc_path_selection_callback", None)
        if callable(setter):
            setter(None)
        self._viewport_callback = None

    def _clear_selection(self, *, reason: str) -> None:
        self._selected_path_id = None
        self._invalidate_row_lookup()
        if self.window is not None:
            self.window.clear_details()
        port = self._selection_port
        if port is not None:
            self._call_selection_port(port, "clear_presented_frame", reason=reason)
            return
        renderer = getattr(self.visualizer, "renderer", None)
        clear_overlay = getattr(renderer, "clear_mpc_path_inspection", None)
        if callable(clear_overlay):
            clear_overlay()

    def _clear_presented_data(self, *, reason: str) -> None:
        self._query_generation += 1
        self._query_debounce_timer.stop()
        self._pending_request = None
        self._debounced_request = None
        self._shutdown_executor()
        self._shutdown_row_lookup_executor()
        self._clear_selection(reason=reason)
        self._frame_token = None
        self._render_packet = None
        self._catalog = None
        self._catalog_input_signature = None

        model = self._model
        self._model = None
        if self.window is not None:
            self.window.release_model()
            self.window.set_counts(0, 0)
        if model is not None:
            for method_name in ("cancel_pending_work", "shutdown", "close"):
                method = getattr(model, method_name, None)
                if callable(method):
                    method()
                    break
            delete_later = getattr(model, "deleteLater", None)
            if callable(delete_later):
                delete_later()

    def _shutdown_executor(self) -> None:
        executor = self._executor
        self._executor = None
        self._active_future = None
        self._active_query_generation = None
        if executor is not None:
            # NumPy kernels cannot be interrupted safely. Waiting here makes
            # the hidden/closed state literal: after this method returns no
            # Explorer worker retains arrays or consumes CPU.
            executor.shutdown(wait=True, cancel_futures=True)

    def _shutdown_row_lookup_executor(self) -> None:
        """Stop the optional scalar-lookup worker and release its array refs."""
        self._invalidate_row_lookup()
        executor = self._row_lookup_executor
        self._row_lookup_executor = None
        self._active_row_lookup_future = None
        self._active_row_lookup_serial = None
        self._pending_row_lookup_request = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _call_selection_port(port: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
        method = getattr(port, method_name, None)
        if callable(method):
            try:
                method(*args, **kwargs)
            except (RuntimeError, ValueError, TypeError):
                logger.exception("MPC selection port failed during %s", method_name)

    def _clear_root_reference(self) -> None:
        if getattr(self.visualizer, "_mpc_explorer_session", None) is self:
            self.visualizer._mpc_explorer_session = None


def toggle_mpc_explorer(visualizer: Any) -> MpcExplorerSession:
    """Toggle the one lazy Explorer session owned by a visualizer root."""
    session = getattr(visualizer, "_mpc_explorer_session", None)
    if isinstance(session, MpcExplorerSession) and not session._finished:
        session.toggle()
        return session
    from .mpc_selection_service import MpcSelectionService

    selection_service = MpcSelectionService(visualizer)
    session = MpcExplorerSession(
        visualizer,
        selection_port=selection_service,
        selection_port_owned=True,
    )
    visualizer._mpc_explorer_session = session
    session.open()
    return session
